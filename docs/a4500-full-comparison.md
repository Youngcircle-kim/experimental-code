# A4500 두 장으로 전체 문항의 16·32·64프레임 비교

기존 `data/videomme_a4500/manifest.json`의 **dev와 eval, short·medium·long,
모든 문항**을 처리한다. 이전에 만든 60 dev 영상과 120 eval 영상 manifest라면
180영상·540문항이다. 기본 manifest는 공식 전체 900영상·2,700문항의 일부다.
다른 manifest는 `--manifest`로 지정할 수 있으며 실제 대상 수를 실행 전에 출력한다.

학습 없이 동일한 pinned CLIP + Qwen3.5-4B로 프레임 예산 16·32·64와
`uniform`, `frame_top`, `temporal_bin`, `C`, `D`의 다섯 선택 방법을 비교한다.
각 문항은 동일한 여섯 선택지 순서로 평가한다. 모델·QA 해상도 320×320·BF16·SDPA,
후보는 기본 2FPS·224×224다. 전체 길이를 포함하는 이번 실행은 아래의 별도 설정으로
**최소 후보 64개**를 모든 예산에 공통 적용한다.

## 두 GPU의 분담

영상 ID를 정렬해 번갈아 두 묶음에 배정하고, 묶음 0은 GPU 0, 묶음 1은 GPU 1에
고정한다. 각 영상의 모든 문항을 같은 GPU에서 처리한다. 모든 예산에서 같은 분할을
사용하며 두 GPU가 각 예산을 나눠 실행한다. GPU당 모델 프로세스는 최대 하나다.
GPU별 대기열에서 B64, B32, B16 순으로 처리하며, 한 GPU가 먼저 끝나면 자기 다음
작업을 시작한다. VRAM은 합쳐지지 않는다.

완료된 문항과 선택지 순서의 체크포인트를 재사용한다. 중간에 실패하면 같은 명령을
다시 실행한다. 코드·입력·설정·GPU 배치가 달라지면 재개를 거부한다. 기존 단일 실행의
`outputs/a4500_ready`와 `outputs/a4500_b64`를 섞지 않고 **새 출력 루트**를 사용한다.

## 1. 전체 대상 영상 준비

아래 명령은 업데이트된 코드가 있는 Ubuntu 서버의 저장소 루트에서 실행한다.
기존 `.venv`, pinned 모델 설정, manifest를 그대로 사용한다. 실행 중에는 코드나
패키지 버전을 바꾸지 않는다.

기존에 dev long 영상만 추출했다면 나머지 영상도 필요하다. 원본 ZIP을 이미
`data/videomme/archives`에 내려받은 경우:

```bash
./.venv/bin/python -m event_caption_pilot.prepare_a4500_videos \
  --manifest data/videomme_a4500/manifest.json \
  --archive-dir data/videomme/archives \
  --video-root data/videomme/videos \
  --split all --duration all
```

모든 MP4가 이미 있다면 추출을 생략한다. 빠진 영상을 자동으로 제외하지 않는다.

### 짧은 영상도 포함하는 공통 후보 설정

2FPS로 추출한 후보가 64개 미만이면 B64 실행이 실패한다. 원본에 서로 다른 프레임이
64개 이상 있는 영상은 `min_candidate_frames: 64`로 기존 후보를 유지하면서 실제
원본 프레임을 보충할 수 있다. 원본 전체에 균등한 64지점 격자를 만들고, 기존 후보와
겹치지 않는 지점에서 필요한 수만큼 균등한 순위로 선택한다. 보충은 픽셀 내용이나
정답을 사용하지 않는다. B16·B32·B64 모두 같은 후보 풀을 사용한다.
이미 64개 이상인 영상은 기존 후보의 인덱스·시간·픽셀을 그대로 유지한다.

기존 pinned 설정을 보존하고 새 설정을 만든다. 다음 블록 전체를 서버에서 실행한다.

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
from event_caption_pilot.config import Config

source = Path('configs/real.a4500.pinned.json')
target = Path('configs/real.a4500.full.pinned.json')
settings = json.loads(source.read_text(encoding='utf-8-sig'))
settings['min_candidate_frames'] = 64
Config(**settings).validate()
if target.exists():
    assert json.loads(target.read_text(encoding='utf-8-sig')) == settings, 'Existing target differs; inspect it first'
else:
    target.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
print(target)
PY
```

중복 프레임이나 보간으로 채우지 않는다. 원본 프레임 자체가 64개 미만이거나,
파일의 선언 프레임 수와 실제 디코딩 수가 다르거나 FPS/프레임 수 메타데이터가
유효하지 않으면 오류로 중단한다. 후자의 경우 영상 파일과 메타데이터부터 확인한다.
`min_candidate_frames` 기본값 0은 기존 2FPS 동작을 유지한다. 새로운 설정과 코드로
기존 결과를 재개하거나 섞을 수 없으므로 아래의 새 출력 루트를 사용한다.

아래 검사는 모델이나 GPU를 사용하지 않고 모든 요청 파일의 SHA256을 계산한 뒤,
실제 디코딩으로 B64에 필요한 고유 후보 64개 이상과 시간 순서 등을 확인한다.
영상 전체를 읽으므로 파일 수와 영상 길이에 따라 시간이 걸린다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -m event_caption_pilot.a4500_full_suite check \
  --config configs/real.a4500.full.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --decode-check
```

`--decode-check`를 생략한 `check`는 파일·설정·문항·해시 검사까지 수행한다.
후보가 부족하거나 손상된 영상이 있으면 영상 ID와 함께 실패한다. 부족한 예산을
몰래 줄이거나 같은 프레임으로 채우지 않는다. 원래의 정확한 예산 조건을 유지한다.

실제 GPU 작업 명령과 분할표만 보려면 다음을 사용한다. 출력 폴더를 만들지 않는다.

```bash
./.venv/bin/python -m event_caption_pilot.a4500_full_suite run \
  --config configs/real.a4500.full.pinned.json \
  --output-root outputs/a4500_full_min64 --dry-run
```

## 2. 두 GPU 전체 실행

```bash
tmux new -s a4500-full
```

새 tmux 안에서 저장소 루트로 이동한 다음:

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -u -m event_caption_pilot.a4500_full_suite run \
  --config configs/real.a4500.full.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-root outputs/a4500_full_min64 \
  --gpu-ids 0 1 \
  --budgets 16 32 64 \
  --splits dev eval \
  --duration all
```

Ctrl+B 다음 D로 분리하고 `tmux attach -t a4500-full`로 돌아온다.
다른 SSH 창에서 처음 실행 중인 두 작업과 GPU 상태를 확인할 수 있다.

```bash
tail -f outputs/a4500_full_min64/logs/dev_b64_s0.log \
        outputs/a4500_full_min64/logs/dev_b64_s1.log
# 별도 창
watch -n 2 nvidia-smi
```

각 작업의 로그 경로와 완료 여부는 런처 화면에도 출력한다. 실패하면 새 작업 시작을
멈추고 실행 중인 다른 작업을 종료한다. 저장된 체크포인트는 유지하며 같은 명령으로
재개한다. 재개 중 GPU 순서도 그대로 유지한다. 기본 GPU 검사 기준은 카드당 여유
16GiB이며, 실제 실행 메모리를 보장하는 수치는 아니다.

dev만 먼저 실행하려면 **새 출력 루트**와 `--splits dev`를 사용한다. eval은 별도
루트에서 `--splits eval`로 실행할 수 있다. 문항이나 영상 수 제한은 이 실행기에 없다.

## 3. 자동 집계와 결과

모든 작업이 성공한 뒤 완료·체크포인트·문항·보기 순서·원본 및 후보 해시·모델·코드·
runtime을 검증하고 자동으로 집계한다. 한 문항이라도 누락되거나 중복되면 완성된
보고서를 만들지 않는다. 영상 분할별 평균이나 신뢰구간을 평균하지 않고, 문항별
원시 결과를 합쳐 영상 단위 bootstrap 5,000회를 다시 계산한다.

| 파일 | 내용 |
|---|---|
| `outputs/a4500_full_min64/comparison.json` | dev/eval별 전체·길이별 정확도와 모든 예산 쌍 비교, QA 비용, 후보 보충 기록 |
| `outputs/a4500_full_min64/comparison.csv` | 표로 보기 쉬운 방법별 정확도·95% 신뢰구간 |
| `outputs/a4500_full_min64/question_metrics.csv` | 문항별·예산별·방법별 순서 평균 |
| `outputs/a4500_full_min64/suite_plan.json` | 전체 대상·분할·입력 및 코드 해시·고정 GPU 배치 |
| `outputs/a4500_full_min64/suite_status.json` | 전체 완료/실패 상태 |
| `outputs/a4500_full_min64/reports/dev/b64/handoff.json` | dev B64의 합산 결과; 다른 split·예산도 동일 구조 |
| `outputs/a4500_full_min64/reports/eval/comparison_b32_b64.json` | eval의 B32↔B64 짝 비교; 16↔32, 16↔64도 생성 |
| `outputs/a4500_full_min64/jobs/` | GPU 분할별 원본 프로토콜·추론 결과·재개용 체크포인트 |

이미 완료된 저장 결과를 GPU 없이 다시 집계할 때:

```bash
./.venv/bin/python -m event_caption_pilot.a4500_full_suite summarize \
  --output-root outputs/a4500_full_min64
```

이 명령은 저장된 suite plan과 모든 원본 결과·체크포인트를 읽는다. 원본 영상이나
모델 가중치를 다시 로딩하지 않는다. 예전 B16/B32/B64 handoff 파일만으로는
이 실행기의 엄격한 비교 검증을 대신할 수 없다.

## 해석과 작업량

dev와 eval 점수는 섞지 않는다. dev는 이미 설정 선택에 사용됐으며, eval도 과거에
확인한 적이 있다면 새 독립 검증으로 해석할 수 없다. 전체 길이의 점수 외에도
short·medium·long을 따로 보고한다. 신뢰구간은 다중 비교 보정 전 개별 구간이다.
이 실행은 여섯 선택지 순서를 평균하는 사용자 정의 평가이므로 공식 benchmark
점수와 동일하다고 주장하지 않는다.

후보를 정확히 64개로 보충한 영상에서 B64는 다섯 방법 모두 모든 후보를 선택한다.
따라서 이 영상의 방법 간 동점은 선택 여지가 없는 설계의 결과다. 합산 보고서의
`candidate_sampling.densified_video_ids`에서 보충 대상 ID를 확인할 수 있고,
`candidate_sampling.videos`에는 원본·기본 후보·최종 후보 수와 보충 정책이 남는다.
문항별 원본 결과에는 `decode_info.candidate_sampling`으로 같은 정보를 기록한다.
기존 2FPS 실험과 새 공통 최소 후보 실험은 별도 실험으로 해석한다.

현재 540문항 기준 **540 × 3예산 × 5방법 × 6순서 = 48,600 QA 호출**이다.
각 QA에는 선택지별 forward가 포함된다. 앞서 측정한 시간 중앙값을 단순 적용하면
이상적으로 두 GPU에 균등 분배했을 때 QA만 약 49시간 규모이고, 영상 디코딩·모델
로딩·디스크 접근 시간이 추가된다. 실제 소요 시간은 길이별 입력과 서버 부하에
따라 달라지며 이 값은 완료 시각 보장이 아니다.

QA 시간과 메모리는 선택 단계·CPU 처리 등을 포함한 전체 처리 비용과 구분한다.
각 예산은 프레임 선택을 다시 수행하므로, B64가 B32의 모든 프레임을 포함한다고
가정하지 않는다.
