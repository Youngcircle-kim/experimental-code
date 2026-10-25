# Legacy 픽셀 복잡도·배분식 비교

참조는 [Legacy 7a711d2의 실제 영상 코드](https://github.com/Youngcircle-kim/Event-Aware-Token-Allocation-for-Efficient-Long-Video-Reasoning/tree/7a711d2/legacy)다.
기존 SigLIP2 개발셋 Long 20영상·60문항, 후보 프레임, D2 이벤트 경계,
Qwen3.5-4B, 32프레임·QA320, 여섯 보기 순서와 두 GPU 배치를 유지한다.
**Legacy의 픽셀 복잡도·관련도 집계·배분식을 이식한 비교**이며, 원래의 픽셀
경계 검출기·전체 원본 프레임 후보·CLIP·Qwen2-VL을 포함한 전체 재현은 아니다.

## 계산

1. 이벤트마다 후보 인덱스 기준 양끝 포함 균등 최대 4장을 고른다.
   원본 RGB에서 Pillow 기본 RGB 리사이즈로 64×64, float32/255로 변환한다.
   원본 영상은 OpenCV로 읽으며 기존 224 후보 전체의 픽셀·시각 해시를 확인한다.
2. `motion = mean(abs(frame[t+1] - frame[t]))`,
   `variance = var(frames)`를 모든 표본·공간·색상 축에 대해 계산한다.
   `raw_complexity = motion + variance`. 단일 후보 이벤트는 원본대로 1e-6이다.
   광학 흐름·객체 수·행동 난이도 지표가 아니다. 정지 무늬도 분산이 높을 수 있다.
3. 복잡도는 원본 호출 순서대로 영상 내 min-max를 두 번 적용한다.
   범위가 1e-6 미만이면 전부 1, 아니면 `(x-min)/(max-min+1e-6)`이다.
4. 저장된 SigLIP2 질문–후보 cosine 중 대표 4장의 최대치를 이벤트 관련도로
   사용한다. 온도 0.07 softmax 후 min-max 정규화한다. 질문의 보기·정답은
   선택에 사용하지 않는다. NumPy float32 계산이므로 원래 CLIP/Torch와의
   비트 단위 동일성을 주장하지 않는다.
5. `importance = minmax((complexity+0.01)*(relevance+0.01))`.
   온도 0.3 softmax, 가능한 경우 이벤트마다 최소 1장, 남은 장수의 정수
   배분·용량 초과 재분배를 적용한다. 정확히 32개의 서로 다른 후보를 고른다.
6. Legacy 균등 선택은 양끝 포함 `linspace`이며 **1장이면 첫 프레임**이다.
   현재 코드의 중앙 프레임 규칙과 다르다. Top-K는 같은 배분에서 구간의 모든
   후보를 질문 cosine으로 순위화하고, 동점이면 앞선 프레임을 고른다.

원본 배분기의 극단적인 용량 제한 상황에서 용량을 초과하는 결함은 수정했다.
해당 경우의 발생 횟수(`capacity_remainder_resets`)를 기록한다. 정상 용량에서
원본과 일치하는지는 원본 함수를 직접 추출해 검증한다.

## 네 조건

| 조건 | 복잡도 | 구간 내부 선택 |
|---|---|---|
| `legacy_R_uniform` | 정규화 복잡도를 모두 1로 고정 | Legacy 균등 |
| `legacy_R_topk` | 정규화 복잡도를 모두 1로 고정 | 질문 Top-K |
| `legacy_CR_uniform` | Legacy 픽셀 복잡도 | Legacy 균등 |
| `legacy_CR_topk` | Legacy 픽셀 복잡도 | 질문 Top-K |

R은 다른 정규화·관련도·배분 절차를 CR과 동일하게 유지한다. 따라서
`CR - R`이 이 배분식에서 픽셀 복잡도를 추가한 효과다. 이전 D와의 비교는
관련도 평균/최대, 정규화, 온도, 최소 장수, 대표 위치까지 함께 바뀐 비교다.

기본 최대 이벤트 수는 16으로, Legacy의 B32 기본값 `min(B//2,16)`에 맞춘다.
실제 개수는 저장된 D2 경계를 사용하므로 상한보다 작을 수 있다. 준비 단계는
16·24·32 세 상한의 복잡도와 배분 변화를 CPU로 계산하지만, QA는 선택한
`--cap` 한 조건만 실행한다. 최소 1장을 보장하면 32구간/B32에서 남는 예산은
0이므로 중요도가 달라도 배분은 모두 1장이 된다.

## 서버 실행

완료된 `outputs/a4500_siglip2_segment_count`가 필요하다. 기존 모듈의 해시,
두 shard의 모든 원본 QA·계획·완료 기록을 검증하고 별도 출력에 고정한다.
기존 모델·의존성이 설치된 가상환경에서 새 의존성 추가 없이 실행한다.

```bash
git pull --ff-only
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -m event_caption_pilot.a4500_legacy_complexity check \
  --source-root outputs/a4500_siglip2_segment_count \
  --output-root outputs/a4500_legacy_complexity_k16 \
  --cap 16 --gpu-ids 0 1

./.venv/bin/python -u -m event_caption_pilot.a4500_legacy_complexity run \
  --source-root outputs/a4500_siglip2_segment_count \
  --output-root outputs/a4500_legacy_complexity_k16 \
  --cap 16 --gpu-ids 0 1
```

장시간 실행은 tmux 안에서 수행한다. 중단 시 같은 명령으로 완료된 trial을
재사용한다. 영상의 모든 조건은 원래와 같은 GPU에 배정하며 GPU당 모델 하나를
사용한다. 먼저 20영상의 준비를 모두 마쳐야 QA를 시작한다.
총 조건별 평가 기록은 `60×4×6=1,440`개다. 같은 질문·순서·선택 인덱스·320
픽셀 해시가 기존 결과나 이번 실행의 다른 조건과 같으면 QA를 재사용한다.
실제 새 추론 수와 재사용 수, 출처를 따로 기록한다.

CPU 계산만 하고 멈추려면 `run`을 `prepare`로 바꾼다. 다른 상한은 새 출력
디렉터리에 `--cap 24` 또는 `--cap 32`로 준비한다. 비교 보고서 생성은 다음과 같다.

```bash
./.venv/bin/python -m event_caption_pilot.a4500_legacy_complexity summarize \
  --output-root outputs/a4500_legacy_complexity_k16
cat outputs/a4500_legacy_complexity_k16/selection_preflight.json
cat outputs/a4500_legacy_complexity_k16/handoff.json
```

`plans/`에 이벤트별 motion·variance·정규화 복잡도·관련도·중요도·장수·선택
인덱스를 저장한다. `selection_preflight.json`은 정확도와 별개로 복잡도가
실제 배분·입력을 바꿨는지 보여준다. `handoff.json`은 비교 요약,
`comparison.json`은 QA 원자료 포함 결과다. 미완료 shard는 완료로 집계하지 않는다.

보고서는 문항 내 여섯 순서 평균, 영상 단위 bootstrap 5,000회와 95% 신뢰구간을
사용한다. 개발셋 탐색 결과이며 다중비교 보정 없는 구간이다. QA 시간·메모리는
원자료에 보존되고 신규 계산과 재사용 기록을 나눈다. QA 시간에 영상 디코딩과
특징 추출 등이 포함되지 않으므로 전체 실행 효율과 구분해서 해석한다.
