"""Run paired frame-budget experiments in isolated single-GPU processes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def check_gpus(gpu_ids, minimum_free_gib=16.0):
    """Check native BF16, actual CUDA arithmetic, and per-card free memory."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable; install a CUDA wheel")
    visible = torch.cuda.device_count()
    if len(set(gpu_ids)) != 2 or any(i < 0 or i >= visible for i in gpu_ids):
        raise ValueError("Select two distinct, visible CUDA GPU indices")
    cards = []
    for index in gpu_ids:
        with torch.cuda.device(index):
            properties = torch.cuda.get_device_properties(index)
            # All Ampere GPUs have native BF16; exclude software emulation.
            if properties.major < 8 or not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"GPU {index} needs native BF16 support")
            free, total = torch.cuda.mem_get_info(index)
            if free / 2**30 < minimum_free_gib:
                raise RuntimeError(
                    f"GPU {index} has only {free / 2**30:.2f} GiB free; "
                    f"need at least {minimum_free_gib:.2f} GiB to start. "
                    "This threshold does not guarantee the full run fits."
                )
            value = torch.ones(
                (256, 256), dtype=torch.bfloat16, device=f"cuda:{index}"
            )
            result = value @ value
            torch.cuda.synchronize(index)
            if not bool(torch.all(result == 256).item()):
                raise RuntimeError(f"BF16 matrix check failed on GPU {index}")
            del value, result
            torch.cuda.empty_cache()
            cards.append({
                "visible_index": index,
                "name": properties.name,
                "total_gib": total / 2**30,
                "free_gib_before_check": free / 2**30,
                "compute_capability": [properties.major, properties.minor],
                "bf16_matmul": "passed",
            })
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cards": cards,
        "note": "Each worker uses one card; VRAM is not pooled.",
    }


def worker_specs(args):
    """Build explicit CUDA visibility and disjoint outputs for both jobs."""
    if len(set(args.gpu_ids)) != 2 or min(args.gpu_ids) < 0:
        raise ValueError("Select two distinct nonnegative GPU indices")
    # Nested CUDA masks can silently select different physical GPUs in a child.
    # Require an unmasked parent so the printed nvidia-smi indices are clear.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "Run `unset CUDA_VISIBLE_DEVICES` before the suite, "
            "then select cards with --gpu-ids"
        )
    specs = []
    for budget, gpu in zip((16, 32), args.gpu_ids):
        output = args.output_root / f"{args.stage}_b{budget}"
        command = [
            sys.executable, "-u", "-m", "event_caption_pilot.portable_budget",
            "--config", str(args.config),
            "--manifest", str(args.manifest),
            "--output-dir", str(output),
            "--frame-budget", str(budget),
            "--split", "eval" if args.stage == "eval" else "dev",
            "--duration", args.duration,
        ]
        if args.stage == "smoke":
            command.extend(["--max-videos", "1", "--max-questions", "1"])
        specs.append({
            "budget": budget,
            "gpu": gpu,
            "command": command,
            "output": str(output),
            "log": str(args.output_root / f"{args.stage}_b{budget}.log"),
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "4",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        })
    return specs


def launch(args):
    specs = worker_specs(args)
    print(json.dumps(specs, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    # Do this in a subprocess: CUDA contexts must be released before workers
    # consume nearly all the free memory on each card.
    environment = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    subprocess.run([
        sys.executable, "-m", "event_caption_pilot.a4500_suite", "check",
        "--gpu-ids", *(str(i) for i in args.gpu_ids),
        "--minimum-free-gib", str(args.minimum_free_gib),
    ], env=environment, check=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    workers, logs = [], []
    try:
        for spec in specs:
            log = Path(spec["log"]).open("a", encoding="utf-8")
            logs.append(log)
            log.write("\nLaunching: " + json.dumps(spec) + "\n")
            log.flush()
            process = subprocess.Popen(
                spec["command"],
                env={**os.environ, **spec["environment"]},
                stdout=log, stderr=subprocess.STDOUT,
            )
            workers.append((process, spec))
        pending = set(range(len(workers)))
        failed = False
        while pending:
            for index in list(pending):
                process, spec = workers[index]
                code = process.poll()
                if code is not None:
                    pending.remove(index)
                    failed |= code != 0
                    print(
                        f"B{spec['budget']} GPU {spec['gpu']}: exit {code}; "
                        f"log: {spec['log']}", flush=True,
                    )
            if pending:
                time.sleep(1)
        if failed:
            print("A worker failed. Inspect its log; rerun to resume.")
            return 1
        print("Both budgets completed. Compare their paired results next.")
        return 0
    finally:
        for process, _ in workers:
            if process.poll() is None:
                process.terminate()
        for process, _ in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("check", "smoke", "dev", "eval"))
    result.add_argument("--gpu-ids", type=int, nargs=2, default=[0, 1])
    result.add_argument("--minimum-free-gib", type=float, default=16.0)
    result.add_argument(
        "--config", type=Path, default=Path("configs/real.a4500.pinned.json")
    )
    result.add_argument(
        "--manifest", type=Path,
        default=Path("data/videomme_a4500/manifest.json"),
    )
    result.add_argument(
        "--output-root", type=Path, default=Path("outputs/a4500")
    )
    result.add_argument(
        "--duration", choices=("all", "short", "medium", "long"),
        default="long",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.minimum_free_gib <= 0:
        raise ValueError("minimum-free-gib must be positive")
    if args.stage == "check":
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Unset CUDA_VISIBLE_DEVICES before GPU check")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(json.dumps(check_gpus(args.gpu_ids, args.minimum_free_gib),
                         indent=2))
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
