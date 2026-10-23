# Dev Long B32: 배분과 구간 내부 선택의 2×2 비교

완료된 `a4500_full_suite`의 **dev Long 전체 20영상·60문항**을 사용한다.
프레임 예산은 32, QA 입력은 320×320, 보기 순서는 기존과 같은 여섯 개다.
학습은 하지 않는다. 기존 B32의 모델 revision, 설정, 후보 RGB·timestamp,
D2 구간 경계와 D 배정 수를 고정하고 아래 세 조건만 새로 평가한다.

| 조건 | 구간별 프레임 배분 | 구간 내부 프레임 선택 | QA |
|---|---|---|---|
| D | 기존 질문 관련도 배분 | 기존 균등/중앙 샘플 | 기존 결과 재사용 |
| D_topk | **D와 같은 배정 수** | 질문–프레임 cosine 상위 k개 | 새로 실행 |
| B | 구간의 **후보 프레임 개수에 비례** | 기존 균등/중앙 샘플 | 새로 실행 |
| B_topk | **B와 같은 배정 수** | 질문–프레임 cosine 상위 k개 | 새로 실행 |

여기서 B는 선택 방법 이름이고, B32의 32는 프레임 예산이다. B의 배분은 기존
`allocate_frames(log(capacities), capacities, 32, 1.0)`과 동일하며 구간당
최소 1장 보장은 없다. top-k는 구간 내부에서만 적용하고, 동점이면 이른 후보를
우선한다. 최종 32장은 중복 없이 시간순으로 정렬한다. 프레임 선택에는 질문만
사용하며 선택지나 정답은 사용하지 않는다.

추가 QA는 **60문항 × 3조건 × 6순서 = 1,080회**다. 선택이 우연히 같아도
각 조건을 별도로 실행한다. 기존 uniform·frame_top·temporal_bin·C·D의
1,800회 결과는 원본 체크포인트를 검증한 뒤 재사용한다.

## 실행

업데이트 파일을 Ubuntu 서버 저장소 루트에 놓고 기존 가상환경을 사용한다.
기존 실험 이후 패키지나 기존 Python 모듈을 바꾸지 않는다. 새 실행기는 새로운
모듈만 추가하며 기존 실험 코드를 덮어쓰지 않는다.

`--source-root`는 `suite_plan.json`과 `jobs/dev_b32_s0`, `jobs/dev_b32_s1`이
있는 **완료된 전체 비교 루트**다. `outputs/a4500_ready`의 단일 실행 결과나
집계 JSON 하나만으로는 실행할 수 없다. 이전에 쓴 config·manifest·원본 영상과
원본 체크포인트가 서버에 그대로 있어야 한다. 설정은 저장된 B32 결과에서 읽으므로
새 pinned config를 만들거나 `--config`를 지정할 필요가 없다.

먼저 원본 체크포인트, 코드, 대상 수, 영상 파일 SHA256을 확인한다. 이 단계는
GPU 추론을 하지 않으며 긴 영상의 파일 해시 계산 때문에 시간이 걸릴 수 있다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -m event_caption_pilot.a4500_long_ablation check \
  --source-root outputs/a4500_full_min64 \
  --output-root outputs/a4500_long_ablation \
  --gpu-ids 0 1
```

`n_videos: 20`, `n_questions: 60`, `expected_new_qa_trials: 1080`을 확인한다.
실행은 tmux 안에서 권장한다.

```bash
tmux new -s long-ablation
```

저장소 루트에서:

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -u -m event_caption_pilot.a4500_long_ablation run \
  --source-root outputs/a4500_full_min64 \
  --output-root outputs/a4500_long_ablation \
  --gpu-ids 0 1
```

기존 dev B32의 영상→GPU 배치를 그대로 사용한다. 각 GPU에 모델 프로세스 하나만
띄우고, 한 영상의 모든 문항·새 조건을 같은 GPU에서 실행한다. 두 카드의 VRAM을
합치지 않는다. 기본 사전검사는 GPU당 사용 가능한 메모리 16GiB 이상을 요구한다.
B32·320·기존 BF16 모델 설정을 유지하지만 실제 peak는 실행 중 별도로 기록한다.

Ctrl+B 다음 D로 tmux에서 빠져나오고 `tmux attach -t long-ablation`으로 복귀한다.

```bash
tail -f outputs/a4500_long_ablation/logs/dev_long_b32_s0.log \
        outputs/a4500_long_ablation/logs/dev_long_b32_s1.log
```

중단되면 **같은 run 명령**으로 재개한다. 완료된 보기 순서별 QA를 다시 호출하지
않는다. 코드·런타임·모델·입력이 달라지면 재개를 거부한다. 실패한 작업이 있으면
완료된 비교처럼 표시하지 않는다. 재시도 전 오류 로그를 확인한다.

## 저장 결과와 해석

- `handoff.json`: 전체 완료 여부, 정확도, 짝 비교 CI, QA 시간·메모리 요약.
- `comparison.json`: 위 요약과 모든 문항·조건의 프레임 선택 및 여섯 QA 원자료.
- `question_accuracy.csv`: 문항별 여섯 순서 평균 정확도, 기존/새 실행 구분.
- `source_snapshot.json`: 검증한 기존 B32 설정·기준군·계획의 변경 감지용 복사본.
- `suite_plan.json`: 실행 코드 해시, 대상, GPU 배치와 예정 추가 QA 수.
- `jobs/dev_long_b32_s*/`: 원본 trial/문항/계획 체크포인트, 런타임·backend·진행 정보.

정확도는 먼저 각 문항의 여섯 순서를 평균한 뒤 문항을 동일 가중치로 평균한다.
**영상 단위 짝 bootstrap 5,000회, 95% percentile CI**를 사용한다. 영상이
뽑히면 그 영상의 모든 문항과 모든 조건을 함께 뽑으므로 여섯 순서를 독립 표본
360개로 취급하지 않는다. 정확도와 차이는 0–1 단위이며 차이에 100을 곱하면 pp다.

우선 확인할 비교는 다음과 같다.

1. `D_topk_minus_D`: 배정 수가 같을 때 구간 내부 top-k의 효과.
2. `B_minus_D`: 기존 내부 샘플러에서 후보 개수 비례 배분의 효과.
3. `B_topk_minus_B`, `B_topk_minus_D_topk`: 다른 축을 고정한 추가 효과.
4. `B_topk_minus_D`: 두 변경을 합친 전체 효과.
5. `interaction`: `(B_topk − B) − (D_topk − D)`의 짝 비교 CI.

각 새 조건과 uniform·temporal_bin·C의 비교도 함께 저장한다. 이 결과는 작은
개발셋에서 수행하는 **다중 비교 보정 없는 탐색적 진단**이다. 이미 확인한 eval
결과를 보고 후속 조건을 골랐으므로 최종 효과 검증에는 별도 미사용 영상이 필요하다.

QA 시간은 각 보기 순서 호출의 동기화된 시간이며 peak CUDA allocated/reserved
메모리를 함께 기록한다. 기존 D/비교군 시간은 과거 실행 측정값이고 새 조건 시간은
이번 실행 측정값이다. 합계는 저장된 성공 trial의 측정값이며 재시도 실패, 디코딩,
특징 추출, 모델 로딩, 두 GPU 동시 실행을 포함하는 실제 경과 시간과 다르다.

완료 후 GPU 없이 다시 집계할 수 있다. 이때 새 출력 루트 전체가 있으면 기존 영상과
원본 실험 폴더는 필요하지 않다.

```bash
./.venv/bin/python -m event_caption_pilot.a4500_long_ablation summarize \
  --output-root outputs/a4500_long_ablation
```

결과 전달용 압축은 저장소 루트에서 실행한다. 영상이나 모델은 포함하지 않는다.

```bash
tar -czf a4500-long-ablation-results.tar.gz -C outputs a4500_long_ablation
```
