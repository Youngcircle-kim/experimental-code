# Question-blind caption evidence probe

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.caption_evidence_probe
```

The probe targets Video-MME question 856-1 and candidate frames 1035 and
1036 (517.5 and 518 seconds). It verifies the original decoded candidate
hash, frozen QA implementation, and runtime/model metadata.

The same Qwen backend generates one caption from only these two frames.
The caption prompt contains no question, options, or gold answer. The
generated caption is retained unchanged and reused for all six orders.
An empty or truncated caption stops the run.

Two conditions use the same images and six paired option orders:

- `frames_only`: original QA prompt and two images.
- `frames_plus_generated_caption`: same QA prompt and images, with the
  generated caption added as quoted descriptive data.

Scoring uses the original option likelihood procedure. Temporary model
settings are restored after generation/scoring, including on exceptions.
Caption generation, exact QA instructions, scores, inputs, and protocol
hashes are saved under `outputs/caption_pair_856_1`.
Use `--resume` to reuse verified checkpoints after an interrupted run.

This is a post hoc diagnostic on one human-selected question. Six option
orders are not six independent questions. The treatment adds both a
caption and a text wrapper/token budget; it does not isolate caption
semantics from all possible prompt effects. Neither outcome estimates
general performance across videos or validates automatic frame selection.
