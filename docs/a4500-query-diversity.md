# 질문 조건부 다양성 A/B/C 비교

완료된 `outputs/a4500_legacy_complexity_k16`의 개발셋 Long 20영상·60문항,
SigLIP2·Qwen3.5-4B 모델과 revision, 후보 프레임, 최대16 D2 이벤트 경계,
32프레임·320 해상도·6가지 보기 순서·GPU별 영상 배치를 그대로 사용한다.
기존 코드와 원본 결과는 수정하지 않고 새 모듈과 출력 디렉터리를 사용한다.

| 조건 | 이벤트 배분 점수 | 내부 선택 |
|---|---|---|
| A_R_topk | 기존 Legacy R 점수 그대로 | 기존 질문 cosine Top-K |
| B_R_Cvis_topk | R × (1 + 0.5 × Cvis) | 동일 Top-K |
| C_R_Cq_topk | R × (1 + 0.5 × Cq) | 동일 Top-K |

단위 SigLIP2 특징 z에 대해 Cvis = 1 - ||mean(z)||²,
Cq = 1 - ||sum(w_i z_i)||²이다.
질문 가중치 w는 이벤트 안의 기존 질문 cosine을 온도 0.07로 softmax한 값이다.
B와 C 모두 이벤트의 모든 후보를 사용한다. 질문 가중치 유무만 다르다.
특징 공간의 분산은 정답 근거·실제 정보량의 대리 지표이며 직접 측정값이 아니다.
복잡도에는 min-max 정규화를 추가하지 않고 [0,1]의 원래 스케일을 사용한다.
곱한 후 점수를 다시 정규화하지 않는다. lambda=0.5, tau=0.07은 실행 전에
고정하며 정답이나 정확도를 보고 조정하지 않는다.

기존 R은 최대4개 대표 프레임의 질문 cosine 최대값, 온도0.07 softmax와
Legacy 정규화로 계산한 기존 점수를 그대로 쓴다. 배분기의 온도0.3,
이벤트당 최소1장, 정수 반올림 및 용량 처리는 동일하다. 특징 재추출 과정에서
관련도나 이벤트 경계를 다시 계산해 교체하지 않는다.

## 검증과 재사용

1. 완료된 Legacy source, protocol, 모든 질문·trial 체크포인트를 검증한다.
2. 기존 두 GPU에 동일한 영상들을 배정해 후보 프레임을 다시 디코딩한다.
3. SigLIP2 특징을 원래 float64 정규화 상태로 저장하고, 이전 실험의 특징
   해시와 정확히 일치하는지 확인한다. 캐시는 `features/*.npy`에 저장한다.
4. 후보 픽셀·타임스탬프와 기존 D 및 A의 QA320 입력 해시를 확인한다.
5. 모든20영상의 선택 계획이 검증된 후에만 두 GPU의 QA를 시작한다.
6. A는 반드시 `legacy_R_topk`의 동일한 360개 trial을 재사용한다.
   B/C도 같은 질문·프레임·QA 픽셀·보기 순서일 때만 기존/현재 결과를 재사용한다.
   조건별 신규·재사용 수와 출처, 저장된 QA 시간과 신규 추론 시간을 분리한다.

총 평가 기록은60×3×6=1,080개이며 신규 QA는 최대720개다.
특징 재추출은 별도 비용이며 QA 횟수에 포함하지 않는다.
같은 영상의 모든 조건은 같은 GPU에서 실행한다. GPU당 모델 하나를 사용하고
준비 프로세스가 종료된 뒤 QA 프로세스를 시작하므로 GPU 메모리를 합치지 않는다.

## 실행

서버의 기존 Python 환경을 그대로 사용한다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -u -m event_caption_pilot.a4500_query_diversity run \
  --source-root outputs/a4500_legacy_complexity_k16 \
  --output-root outputs/a4500_query_diversity_abc \
  --gpu-ids 0 1
```

`prepare` 단계로 실행하면 특징 재추출과 선택 진단만 수행한다.
같은 코드·source·출력 경로로 재실행하면 검증된 체크포인트를 재사용한다.
실행 중 코드·환경·입력·점수 파라미터를 바꾸면 안 된다.

상태는 `suite_status.json`, `preparation/s*/progress.json`,
`jobs/dev_long_query_diversity_b32_s*/progress.json`에 기록한다.
진단과 로그는 `selection_preflight.json`, `logs/prepare_s*.log`,
`logs/qa_s*.log`에 있다. 완료 후 `handoff.json`, `comparison.json`,
`question_accuracy.csv`를 생성한다.

## 사전에 고정한 분석

주 비교는 C−B(질문 가중치 효과)이며 B−A(의미적 다양성), C−A(전체 효과)를
함께 보고한다. 문항별6순서 평균을 낸 뒤 영상 단위 짝 bootstrap5,000회,
seed42의95% 신뢰구간을 사용한다. 360회 반복 답변을360개 독립 문항으로
취급하지 않는다. 개발셋 반복 탐색이고 다중비교 보정 없는 주변 신뢰구간이므로
독립 평가셋에서의 확정적인 개선으로 해석하지 않는다.

정확도 외에 배분 변경 문항 수, 교체 프레임 수, 선택 특징 중복 진단,
QA 시간·메모리, 디코딩·특징 재추출·선택 시간, 전체 실행 시간을 함께 기록한다.
사람이 정답 근거를 표시한 데이터가 없으므로 근거 포함률을 자동으로 만들어
보고하지 않는다. 효과가 없더라도 사후에 가중치를 바꿔 같은 실행으로 취급하지 않는다.
