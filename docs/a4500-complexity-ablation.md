# A4500 두 GPU: 복잡도 × 관련도 × 구간 내부 선택

개발셋 Long 20영상·60문항, 프레임 예산32, QA 해상도320에서 복잡도 배분의
효과를 확인한다. 기존 SigLIP 2 기준 실험의 후보 프레임·D2 경계·모델 revision·
질문·여섯 보기 순서·영상별 GPU 배치를 고정한다. CLIP 기준군도 같은 인코더를
유지하면 사용할 수 있지만, 서로 다른 인코더의 결과를 섞지 않는다.

## 복잡도의 정의

이 실험의 복잡도는 **시각 특징의 시간적 변화와 다양성에 대한 대리지표**다.
광학 흐름, 객체 수, 행동의 난이도나 정답 증거의 양을 직접 측정하지 않는다.
legacy의 픽셀 분산 기반 복잡도를 그대로 재현한 것도 아니다.

이벤트의 단위 길이 프레임 특징을 `x_1, ..., x_n`이라고 할 때:

1. 시간 변화 `T`: 이벤트 내부의 인접 프레임에 대한 `(1 - cosine) / 2` 평균.
2. 다양성 `V`: 서로 다른 모든 프레임 쌍의 `(1 - cosine) / 2` 평균.
   전체 쌍을 만들지 않고 `n / (n - 1) * (1 - ||mean(x)||²) / 2`로 계산한다.
3. 복잡도 `c = (T + V) / 2`, 배분 배수 `g = 1 + c`.

각 성분은 0–1 범위이고, 한 프레임 구간은 두 성분을 모두0으로 둔다.
정지 프레임의 수치 오차가 배분 동점을 깨지 않도록 설정의 정규화 허용 오차
이내인 변화량은0으로 처리하고, 사용한 허용 오차도 함께 기록한다.
구간 경계를 가로지르는 변화는 포함하지 않는다. 합계 대신 평균을 사용하므로
구간 길이를 명시적으로 곱하지 않지만, 관측 길이와 통계적으로 독립이라는 뜻은
아니다. 복잡도는 질문·보기·정답을 받지 않고 영상마다 한 번 계산한다.

배수는1–2 범위로 고정했다. 작은 변화량을 영상별 최댓값으로 확대하거나
검증셋 결과에 맞춰 배수를 조정하지 않는다. 모든 구간의 복잡도가 같으면
복잡도 사용 여부가 배분에 영향을 주지 않는다.

## 비교 조건

`r`은 기존 D와 같은 정규화 평균 특징–질문 cosine이고 `tau`는 기존 배분
temperature다. 다음 가중치에 동일한 용량 제한·정수 배분을 적용한다.
최소1장 보장은 추가하지 않으며 총32개의 서로 다른 프레임을 선택한다.

| 배분 | 구간 가중치 | 내부 균등 선택 | 내부 관련도 top-k |
|---|---|---|---|
| 동일 가중치 | `1` | `N` | `N_topk` |
| 관련도만 | `exp(r / tau)` | 기존 `D` | `D_topk` |
| 복잡도만 | `g` | `K` | `K_topk` |
| 관련도 × 복잡도 | `exp(r / tau) * g` | `KD` | `KD_topk` |

`N`은 **이벤트별 동일 가중치**다. 전체 영상의 균등 프레임 선택이나 기존
후보 개수 비례 배분 `B`와 다르다. `K_topk`와 `N_topk`도 내부 top-k에는 질문을
사용한다. 따라서 '관련도 없음'은 **배분 점수에 질문 관련도를 사용하지 않음**을
뜻하며, 전체 선택 과정이 질문과 무관하다는 뜻이 아니다.

기존 D는 기준군에서 재사용한다. 나머지7조건은 동일한 실행에서 새로 평가한다.
이전 D_topk 후속 실험이 있더라도 이번 실행에서는 새 D_topk를 계산한다.
추가 QA는 `60 × 7 × 6 = 2,520회`이며, 각 QA는 네 보기의 별도 점수 계산을
포함한다. 기존5조건 기준군부터 준비하면 `1,800 + 2,520 = 4,320회 QA`다.

## 실행

설치는 기존 SigLIP 2 안내와 같다. 새 의존성이나 추가 모델은 필요하지 않다.
모든 명령은 Ubuntu 서버 저장소 루트의 같은 가상환경에서 실행한다.

완료된 `outputs/a4500_siglip2_long_baseline`이 없으면 먼저 기준군을 만든다.
이 단계에는 기존 pinned 설정·manifest·개발셋 Long 원본 영상이 필요하다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -u -m event_caption_pilot.a4500_full_suite run \
  --config configs/real.a4500.siglip2.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-root outputs/a4500_siglip2_long_baseline \
  --gpu-ids 0 1 --budgets 32 --splits dev --duration long
```

기준군이 완료됐으면 복잡도 실험을 확인하고 실행한다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -m event_caption_pilot.a4500_complexity_ablation check \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_complexity \
  --gpu-ids 0 1

./.venv/bin/python -u -m event_caption_pilot.a4500_complexity_ablation run \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_complexity \
  --gpu-ids 0 1
```

`check`의 대상은20영상·60문항, 추가 QA2,520회여야 한다. 기존 D_topk 실험의
출력 루트가 아니라 **원래5조건 기준군**을 `--source-root`로 지정한다.
각 영상의 모든 조건은 원래 기준군과 같은 GPU에서 실행한다. GPU마다 모델
프로세스 하나를 사용하며 두 장의 VRAM을 합치지 않는다.

새 기능은 별도 모듈이므로 기존 기준군 코드의 변경 검사를 우회하지 않는다.
원래 기록된 코드·모델·런타임·입력이 달라져 검증에 실패하면 원인을 해결하거나
동일한 새 환경에서 기준군부터 다시 실행해야 한다. 기존 결과 파일을 수정해서
검사를 통과시키지 않는다. 실험이 시작된 뒤에는 코드를 변경하지 않는다.

tmux 안에서 실행하고, 중단되면 동일한 `run` 명령을 다시 실행한다.
완료된 보기 순서별 체크포인트를 재사용한다. 결과가 일부 누락되거나 실패하면
완료 비교로 집계하지 않는다.

## 결과 해석

`handoff.json`은 요약, `comparison.json`은 원자료를 포함한 비교 결과,
`question_accuracy.csv`는 문항별 정확도다. 계획에는 구간별 복잡도 성분,
배분 점수·장수, 선택 프레임과 진단 정보를 보존한다.

먼저 **`KD_topk_minus_D_topk`**를 확인한다. 같은 관련도·top-k 선택에 복잡도만
추가한 효과다. `KD - D`는 내부 균등 선택에서의 같은 효과이며, `K - N`은
관련도 배분이 없을 때의 복잡도 효과다. `(KD - D) - (K - N)`의 짝 비교는
두 신호의 상호작용을 평가한다. top-k 조건에서도 같은 상호작용을 계산한다.

각 문항의 여섯 보기 순서를 평균하고 문항에 동일 가중치를 준다. 영상 단위
짝 bootstrap5,000회의95% 신뢰구간을 사용하며 여러 조건의 탐색적 비교에는
다중 비교 보정을 적용하지 않는다. 동일 프레임 선택이 발생하면 그 사실도
확인해야 한다. 복잡도 배수가 달라도 정수 배분 결과는 같을 수 있다.

QA 시간과 peak CUDA 메모리는 기록하지만 디코딩·인코더·선택·모델 로딩을
포함한 전체 지연시간으로 해석하지 않는다. 기존 D 시간은 과거 실행값이며
재개한 trial은 원래 측정값을 유지한다. 시각 변화 점수가 높다는 것만으로
정답에 필요한 증거가 더 포함됐다고 단정하지 않는다.

완료 후 GPU 없이 재집계하거나 결과를 전달할 수 있다.

```bash
./.venv/bin/python -m event_caption_pilot.a4500_complexity_ablation summarize \
  --output-root outputs/a4500_siglip2_complexity

tar -czf a4500-siglip2-complexity-results.tar.gz -C outputs \
  a4500_siglip2_complexity
```
