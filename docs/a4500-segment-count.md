# A4500 두 GPU: 이벤트 수를 통제한 분할·배분 비교

개발셋 Long 20영상·60문항에서 프레임 예산32, QA 해상도320,
기존 인코더·QA 모델·후보 프레임·여섯 보기 순서·영상별 GPU 배치를 유지한다.
완료된 원래5조건 기준군 `outputs/a4500_siglip2_long_baseline`을 사용한다.
복잡도 실험 출력이나 handoff.json 단독 파일은 입력 기준군이 아니다.

## 검증할 질문

이전 결과에서는 대부분 영상이32개 이벤트에 한 장씩을 배정했다.
이번에는 이벤트 수를 줄여 여러 장 배분이 실제로 일어나는지 먼저 확인하고,
다음 효과를 분리해서 측정한다.

- 같은 구간 개수에서 변화 기반 이벤트 경계가 고정 구간 경계보다 나은가?
- 같은 경계·내부 선택에서 관련도 배분에 복잡도를 더하면 나아지는가?
- 구간당 여러 장이 들어갈 때 내부 균등 선택과 관련도 top-k가 달라지는가?

| 요인 | 조건 |
|---|---|
| 이벤트 수 상한 | 16, 24, 32 |
| 경계 | D2 이벤트 / 실제 이벤트 수와 같은 고정 구간 |
| 배분 | 관련도 D / 관련도 × 복잡도 KD |
| 구간 내부 선택 | 균등 / 질문 관련도 top-k |

총24조건으로 추가 QA는 **60 × 24 × 6 = 8,640회**다.
각 QA에는 네 보기의 점수 계산이 포함된다. 기존5조건1,800회는 재사용한다.
같은 영상의 모든 조건은 원래 기준군과 같은 GPU에서 실행한다.
GPU마다 모델 프로세스 하나만 사용하고 두 장의 VRAM을 합치지 않는다.

## 상한과 실제 구간 수를 구분한다

`event_k16`은 이벤트를 정확히16개로 강제하는 조건이 아니라 **상한16**이다.
기존 D2의 특징·변화 임계값·윈도·최소 구간 길이는 그대로 두고 상한만 바꾼다.
실제 이벤트가11개면 `fixed_k16`도11개 고정 구간으로 만든다.
고정 구간은 기존 D0 방식대로 후보 인덱스를 균등하게 나눈다. 후보 시간이
거의 균일한 경우 시간 구간도 거의 균등하지만 초 단위 완전 동일 길이는 아니다.

16·24·32 상한에서 같은 분할이 나오는 영상도 제외하지 않는다.
실제 구간 수와 조건 간 동일 선택 비율을 보고하며, 이를 독립 표본으로
추가 계산하지 않는다. 모든 비교는 같은20영상·60문항을 사용한다.

복잡도는 기존과 동일하게 구간 내부 특징의 시간 변화와 다양성의 평균이고,
배수는 `1 + complexity`다. D는 `exp(relevance / temperature)`, KD는 여기에
복잡도 배수를 곱한다. 기존 용량 제한 정수 배분을 그대로 사용하고,
최소1장 보장·복잡도 배수 확대·해상도 변경은 추가하지 않는다.
각 조건은 정확히32개의 서로 다른 프레임을 시간순으로 QA에 전달한다.

## 서버 실행

기존 실험과 같은 Ubuntu 서버, 가상환경, 저장소 루트에서 실행한다.
새 패키지나 모델을 설치할 필요는 없다. 실행 중인 다른 실험이 없는 상태에서
코드를 업데이트한다. 실험 시작 후에는 코드를 변경하지 않는다.

```bash
cd /workspace/experimental-code
git pull --ff-only
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES

python -m event_caption_pilot.a4500_segment_count check \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_segment_count \
  --gpu-ids 0 1
```

저장소 위치가 다르면 첫 번째 경로만 실제 경로로 바꾼다.
`check`는 원래 결과·영상 파일·모델 설정·코드의 호환성을 검증하며,
새 모델 추론이나 QA는 실행하지 않는다. 출력은20영상·60문항·8,640회여야 한다.

전체 영상의 선택 계획과 배분 진단만 먼저 계산하려면:

```bash
python -u -m event_caption_pilot.a4500_segment_count prepare \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_segment_count \
  --gpu-ids 0 1

cat outputs/a4500_siglip2_segment_count/selection_preflight.json
```

`prepare`는 두 GPU에서 특징과 질문 임베딩을 계산하고 선택 계획을 저장하지만,
새 답변 QA는 실행하지 않는다. 전체20영상의 계획과 원래 D 입력 픽셀까지
검증해야 준비 완료가 된다. 모델 로딩에는 GPU 메모리가 필요하다.
진단에서 실제 구간 수, 0·1·2장 이상 배정 비율, D→KD의 배분·입력 변경률,
이벤트→고정 구간의 입력 변경률, 상한별 동일 선택을 확인한다.
변경률이 낮아도 설정을 자동 조정하지 않는다.

전체 QA 실행은 tmux 안에서 다음 명령을 사용한다.

```bash
python -u -m event_caption_pilot.a4500_segment_count run \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_segment_count \
  --gpu-ids 0 1
```

`run`은 준비 단계가 없으면 자동으로 수행한다. **두 GPU의 전체 선택 계획이
검증되기 전에는 어느 영상의 새 QA도 시작하지 않는다.** 준비 단계가 이미
완료됐으면 저장된 계획을 검증해 재사용한다. 중단되면 같은 명령으로 재개한다.
완료된 보기 순서별 QA는 재사용하고, 빠진 문항을 제외한 채 완료로 집계하지 않는다.
원래 기준군 파일을 수정하거나 덮어쓰지 않는다.

## 결과와 해석

- `selection_preflight.json`: QA 전 전체 선택·배분 진단과 계획 해시.
- `handoff.json`: 정확도, 짝 비교 신뢰구간, 선택 진단, QA 시간·메모리 요약.
- `comparison.json`: 요약과 문항별 조건·선택·trial 원자료.
- `question_accuracy.csv`: 문항별·조건별 정확도.
- `logs/*_prepare.log`, `logs/*.log`: GPU별 준비·QA 실행 로그.

같은 상한·같은 내부 선택에서 **event − fixed**와 **KD − D**를 먼저 본다.
상한 변경에 따른 효과와 내부 top-k 효과도 함께 보고한다.
기존 Uniform·Frame Top·Temporal Bin·C·D는 역사적 기준군으로 재사용된다.
상한만 줄인 `event_k16_D`·`event_k24_D`·`event_k32_D`와 원래 D의 짝 비교도
포함하므로, 기존의 많은 이벤트를 줄인 효과를 직접 확인할 수 있다.

각 문항의 여섯 보기 순서를 평균하고, 영상 단위 짝 bootstrap5,000회로
95% 신뢰구간을 계산한다. 여섯 순서를 별도 문항으로 세지 않는다.
이 실험은 이미 살펴본 개발셋의 탐색적 비교이며 다중 비교 보정은 하지 않는다.
개발셋 최고점을 확정적 우위로 해석하거나 같은 결과를 최종 평가로 재사용하지 않는다.
시간은 QA 호출 범위이며 디코딩·인코더·분할·선택·모델 로딩을 제외한다.
기존 기준군과 시간·메모리의 직접적인 효율 우위를 주장하지 않는다.

완료 후 GPU 없이 재집계하거나 결과를 전달할 수 있다.

```bash
python -m event_caption_pilot.a4500_segment_count summarize \
  --output-root outputs/a4500_siglip2_segment_count

tar -czf a4500-segment-count-results.tar.gz -C outputs \
  a4500_siglip2_segment_count
```

이전 실행의 코드·런타임·모델·입력 검증에 실패하면 오류 원인을 해결한다.
검사를 통과시키기 위해 이전 결과나 해시를 수정하지 않는다.
