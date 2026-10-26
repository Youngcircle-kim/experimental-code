# RTX 3060: 고정 이벤트에서 배분과 내부 선택 비교

기존 개발셋 Long 20영상·60문항의 이벤트 경계(최대16), 후보 시각,
SigLIP2 특징과 질문 cosine을 그대로 사용한다. 프레임 예산은32다.
질문·선택지·정답을 새로 만들지 않으며 정답과 과거 QA 결과는 선택기에
전달하지 않는다. 기존 A4500 실험 결과는 새 QA 결과로 재사용하지 않는다.

## 사전에 고정한 네 조건

| 조건 | 구간별 장수 배분 | 구간 내부 선택 |
|---|---|---|
| R_topk | 기존 Legacy 관련도 배분 | 질문 관련도 Top-K |
| G_topk | 이벤트당1장 보장 후 전체 후보의 질문 관련도 Top-K로 장수 결정 | 질문 관련도 Top-K |
| R_diverse | R_topk와 정확히 같은 장수 | 관련도 상위 후보 안에서 중복 억제 |
| G_diverse | G_topk와 정확히 같은 장수 | 관련도 상위 후보 안에서 중복 억제 |

G 배분은 각 이벤트의 최고 관련도 프레임1장을 먼저 확보한다. 나머지는
영상 전체의 아직 선택하지 않은 후보를 질문 cosine 내림차순으로 채우고,
이벤트별 선택 개수를 배분량으로 삼는다. 이는 프레임별 관련도를 합한 값이
최대가 되는 단순 기준선이다. 실제 정답 정보량을 측정하는 공식은 아니다.

중복 억제 선택은 이벤트에 k장이 배정되었을 때 질문 관련도 상위
min(2k, 후보수)장만 후보로 사용한다. 최고 관련도1장은 보존하고, 이후에는
이미 선택한 프레임들과의 최대 특징 cosine이 가장 작은 후보를 추가한다.
동점에서는 질문 관련도, 이른 시각 순서로 선택한다. 서로 다른 이벤트 사이의
중복은 억제하지 않는다. 최종32장은 시간순으로 QA 모델에 전달한다.
후보 풀 배수2는 실행 전에 고정하며 결과를 보고 조정하지 않는다.

## 3060 실행 조건

- 기존 Qwen3.5-4B revision과 BF16/SDPA/선택지 likelihood 평가를 유지한다.
- 저장된 특징으로 CPU에서 선택하므로 SigLIP2를 GPU에 로드하지 않는다.
- QA는 GPU 한 장에서 순차 실행하고 기준 조건도 새로 측정한다.
- 320 해상도에서 가장 긴 질문·선택지의 smoke를 먼저 실행한다.
  메모리가 부족하면 별도 출력에서224로 다시 smoke한다. 본 실험 해상도는
  모든 조건에서 동일하게 고정하며, 설정이 다른 결과를 합치지 않는다.
- 여섯 보기 순서는 기존 문항별 순서를 그대로 유지한다. 전체 평가 기록은
  60문항 × 4조건 × 6순서 = 1,440개다.
- 동일 실행의 동일 질문·시각·픽셀·보기 순서는 재사용할 수 있다. 다른 GPU나
  기존 A4500의 trial은 재사용하지 않는다. 신규/재사용 수와 시간을 분리한다.
- 원본 영상 SHA256, 후보 픽셀/시각 해시, 특징 해시를 검증한다. 중간 실패를
  정확도 분모에서 제외하지 않으며 미완료는 미완료로 기록한다.

실행 모듈은 `event_caption_pilot.rtx3060_fixed_events`다. `--help`로
`prepare`, `smoke`, `run`, `aggregate` 명령과 경로 인자를 확인할 수 있다.
실제 실행 경로·해상도·코드와 런타임 해시는 출력의 protocol에 고정한다.
smoke와 본 실험은 별도 출력 폴더를 사용한다.

2026-10-27 실제 시험에서는320에서 할당11.43GiB/예약17.80GiB,
QA당18.3~25.1초였다. 224에서는 할당9.76GiB/예약11.51GiB,
QA당7.5~8.1초로 네 조건이 모두 완료되었다. 메모리와 실행 시간을 근거로
본 실험은 **224 해상도**를 채택했다. 정확도를 보고 해상도를 선택하지
않았으며, 기존320 결과와 새224 결과의 점수를 직접적인 방법 효과로
비교하지 않는다.

현재 Windows 디코더는 원본 영상 파일 SHA256이 같아도 기존 Linux 후보
픽셀 해시를 재현하지 못했다. 따라서 기존 서버에서 GPU를 사용하지 않고
같은 디코더로 필요한 프레임의 합집합을 추출했다. `--pixel-cache-dir`는
이 캐시의 원본·후보·파일·배열 해시, 선택 요청, 인덱스·시각·해상도를 모두
검증한다. 검증을 우회하거나 다른 픽셀을 같은 입력으로 취급하지 않는다.

준비된 로컬 프로젝트에서 PowerShell 실행 예시는 다음과 같다.

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONHASHSEED='42'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_PROGRESS_BARS='1'

.\.venv\Scripts\python.exe -u -m event_caption_pilot.rtx3060_fixed_events smoke `
  --pixel-cache-dir outputs/rtx3060-fixed-events-20261027/pixels224 `
  --output-root outputs/rtx3060-fixed-events-20261027/smoke224 `
  --qa-side 224 --timeout-seconds 1800

.\.venv\Scripts\python.exe -u -m event_caption_pilot.rtx3060_fixed_events run `
  --pixel-cache-dir outputs/rtx3060-fixed-events-20261027/pixels224 `
  --output-root outputs/rtx3060-fixed-events-20261027/full224 `
  --qa-side 224 --timeout-seconds 28800
```

기본 source-root는 이전 A/B/C 로컬 다운로드이고, feature-dir는 이번
campaign의 `features` 폴더다. 둘 다 검증된 기존 원자료가 필요하다.
`progress.json`으로 진행률을 확인한다. 완료 후 `handoff.json`은 요약,
`comparison.json`은 문항별 원자료를 포함한다. 자동 재시작은 하지 않는다.

## 사전 분석과 해석

주 비교는 G_topk−R_topk(배분)와 R_diverse−R_topk(내부 선택)다.
G_diverse−R_topk(전체), 각 요인의 평균 효과와 상호작용도 함께 보고한다.
문항의 여섯 순서 정확도를 평균한 뒤 영상 단위 짝 bootstrap5,000회,
seed42로95% 신뢰구간을 계산한다. 반복 탐색 개발셋이며 두 주 비교 및
추가 분석을 확정적 검정이나 독립 평가셋 개선으로 제시하지 않는다.

QA 전에 전60문항의 배분 변화·프레임 교체·관련도·특징 중복·시간적 분포를
검사한다. 입력이 동일한 조건을 별도 알고리즘 효과처럼 해석하지 않는다.
특징 유사도 감소는 정답 근거 회수율이 아니다. 근거 annotation 없는
정보량/근거 포함률 수치를 생성하지 않는다.

설계의 목적은 복잡도 항을 추가하는 것이 아니라, 이벤트 경계를 고정한 채
배분과 내부 선택 중 어느 단계가 실제 입력 및 정확도를 바꾸는지 확인하는
것이다. 결과가 좋더라도 이벤트 분할 자체의 우위를 입증하는 실험은 아니다.
