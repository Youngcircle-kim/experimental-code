# 고정한 D의 독립 긴 영상 평가

이번 실험의 질문은 **개발셋에서 정한 D가 새로운 긴 영상에서도 Uniform보다
정확한가**이다. D의 선택 규칙과 QA 설정을 고정하고, 기존 manifest의
`eval` 중 긴 영상 **39개·117문항**에 네 조건을 적용한다.

## 이 실험을 선택한 이유

기존 개발셋의 D 정확도는 44.17%, Uniform은 35.56%였다. 주 비교인
D−Uniform은 **+8.61%p**, 영상 단위 95% 신뢰구간은 **+1.39~+16.11%p**다.
그러나 D의 최대 구간 수와 QA 해상도 등을 조정한 뒤 같은 개발 영상에서 얻은
결과이므로, 이 구간만으로 새 영상에서의 개선을 확정할 수 없다.

따라서 다음 단계에서는 알고리즘을 더 조정하지 않고, 원본 영상이 겹치지 않는
보류 표본에서 재현되는지 확인한다. Uniform을 주 비교로 고정하면 간단한 균등
추출보다 복잡한 질문 기반 선택이 실제로 도움이 되는지 직접 판단할 수 있다.
Top-k와 Temporal-bin은 높은 관련성 점수만 고르는 방식과 시간 범위를 유지하는
방식에 비해 D가 어떤 차이를 보이는지 설명하는 보조 비교다.

39영상·117문항은 기존에 정해진 보류 표본이며, 유의한 결과가 나오도록 새로
계산한 표본 크기가 아니다. 결과의 방향과 신뢰구간 폭을 함께 보고한다.

## 기존 실험과 달라지는 점

| 항목 | 완료된 개발셋 비교 | 이번 보류셋 비교 |
|---|---|---|
| 대상 | dev 긴 영상 20개·60문항 | eval 긴 영상 39개·117문항 |
| 목적 | 조정한 D의 탐색적 성능 확인 | 고정한 D의 새 영상 성능 검증 |
| D 설정 | 개발 과정에서 결정 | 기존 설정 그대로 고정 |
| D의 QA | 완료된 320 결과 재사용 | 새 영상에서 새 QA |
| 비교군 QA | 세 조건 새 QA | 네 조건 모두 새 QA |
| 최대 새 QA | 60 × 3 × 6 = 1,080회 | 117 × 4 × 6 = 2,808회 |
| QA 시간 | D는 과거 실행 측정 | 네 조건 모두 이번 실행 측정 |
| 동일하게 유지 | 320×320·16장, 모델·프롬프트·채점·순서 6개 | 왼쪽과 동일 |

기존 factorial 실험의 D−C는 **질문 관련성 배분을 유지했을 때 구간을 나누는
방식의 효과**를 살펴보는 비교였다. 이번 D−Uniform은 D 전체 선택 절차의
효과를 측정하므로, 개선이 구간 분할만의 효과라고 해석하지 않는다.

이 단계에서 새 알고리즘을 제안하는 것은 아니다. 또한 Global Top-k와
Temporal-bin은 이 저장소에서 정의한 비교 규칙이며, 외부 논문 전체를 재현한
모델이 아니다. 이번 실험만으로 선행연구 대비 신규성이나 최고 성능을 주장하지
않는다. 외부 연구와의 직접 비교에는 해당 구현·데이터·QA 프로토콜을 맞춘
별도 실험이 필요하다.

## 고정하는 비교 조건

| 결과 이름 | 16장 선택 규칙 |
|---|---|
| `D` | D2 특징 변화로 구간 분할, 질문 관련성으로 장수 배분, 구간 안에서 균등 선택 |
| `uniform` | 전체 후보에서 양 끝을 포함해 균등 선택 |
| `frame_top` | 전체 후보의 질문–CLIP cosine 상위 16장, 간격 제한 없음 |
| `temporal_bin` | 후보 수가 비슷한 시간순 16구간에서 최고 점수 한 장씩 선택 |

모든 조건은 서로 다른 16장을 시간순으로 QA에 넣는다. 비교군의 점수가 같으면
먼저 나온 후보를 선택한다. Temporal-bin은 시간 길이 자체가 아니라 후보 수를
기준으로 나눈다.

- D: 최대 128구간, 기존 D2 임계값·최소 구간 길이·배분 온도 유지.
- 선택용 후보: 기존 2 FPS, 최대 8,192장, **224×224** 및 고정 CLIP 설정 유지.
- QA 입력: 선택된 원본 프레임을 **320×320**으로 다시 추출, 시각 토큰 1,600개.
- QA 모델: 기존 Qwen/가중치 revision, BF16, SDPA, 프롬프트 및 길이 정규화한
  선택지 로그우도 채점 유지.
- 선택지 순서: 기존과 같은 고정된 6개 순서. 한 번의 QA는 한 순서에서 모든
  선택지를 채점하는 실행이며, 모델 forward 한 번과 같은 단위가 아니다.
- 선택기 입력: 질문 문장과 영상 특징만 사용. 선택지·정답·근거 주석은
  선택에 사용하지 않는다. 선택지와 정답은 QA 채점 단계에서 사용한다.
- 자막·오디오·새 캡션을 추가하지 않는다.

## 대상 고정과 로컬 영상 준비

대상은 `data/videomme_pilot/manifest.json`의 `split=eval`, `duration=long`
전체다. 파일이 없거나 특정 방법의 성능이 낮다는 이유로 영상을 대체하거나
제외하지 않는다. manifest의 원본 `source_id`가 dev와 겹치지 않는지 검증한다.
소스 ID 검사는 편집본·재인코딩본의 의미적 중복을 완전히 증명하지는 못한다.

구현 시점 읽기 전용 점검에서는 대상 경로에 0/39개가 있었지만,
`data/videomme/archives/videos_chunked_01.zip`부터 `videos_chunked_20.zip`까지의
로컬 아카이브 안에서 **39개 모두** 찾았다. 같은 파일명의 중복은 없었으며,
합계 비압축 크기는 9,599,309,631바이트(약 8.94 GiB)였다. 이 목록 확인은 ZIP
전체의 CRC 검증이나 영상 디코딩 검증을 대신하지 않는다. 영상 다운로드 없이
대상 파일만 로컬에서 추출할 수 있다.

dev 결과 파일과 보류 평가 산출물은 별도로 유지한다. 저장소에 남아 있는 이전
실험의 사용 표본을 점검하되, 기록에 남지 않은 외부 평가나 모델 학습 데이터에
이 영상이 사용됐는지까지 확인했다고 주장하지 않는다. 보류 점수를 본 뒤
설정을 바꾸면 이 표본은 더 이상 그 변경안의 독립 검증셋이 아니다.

## 실행

PowerShell에서 작업 폴더를 연다.

```powershell
Set-Location C:\Users\dudgh\experimental-code
```

먼저 로컬 아카이브의 대상 목록만 확인한다. 모델 실행·다운로드·파일 추출은
하지 않으며 출력 폴더도 만들지 않는다. 이미 추출한 파일이 있으면 크기와 CRC가
아카이브와 일치하는지도 읽기 전용으로 확인한다.

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.prepare_holdout_videos --check-inputs
```

실제 영상 준비는 다음 명령이다. eval 긴 영상만 지정된 manifest 경로에
추출한다. 같은 파일이 이미 있으면 크기와 CRC를 확인하고 재사용하며,
다른 내용이면 덮어쓰지 않고 중단한다. ZIP 내부 경로를 출력 경로로 사용하지
않는다. 새 파일은 임시 파일에 추출해 CRC를 검증한 뒤 원자적으로 설치한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.prepare_holdout_videos
```

다른 경로를 사용할 때는 `--manifest`, `--archive-dir`, `--video-root`를
명시한다. 대상 manifest 경로가 `--video-root` 밖이면 추출을 거부한다.
이 도구는 추출 무결성을 확인하며, 영상 내용·프레임 디코딩은 다음 단계에서
검사한다.

실행기 입력 검사:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.selector_comparison_holdout --check-inputs
```

선택 계획을 준비한다. 이 단계에서는 새 영상의 후보와 CLIP 특징을 사용해
네 조건의 선택을 고정하며 QA 정답률을 산출하지 않는다. 기본 phase도
`prepare`이므로 단계 인자를 생략해도 QA가 자동으로 시작되지는 않는다.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_holdout --phase prepare --output-dir outputs/selector_compare320_holdout_long
```

준비가 완료되면 같은 폴더에서 네 조건의 QA를 실행한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_holdout --phase evaluate --output-dir outputs/selector_compare320_holdout_long
```

중단되면 해당 단계의 **같은 명령을 다시 실행**한다. 완료된 체크포인트를
검증하고 재사용한다. 전체 결과를 다시 집계하려면:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.selector_comparison_holdout --phase summarize --output-dir outputs/selector_compare320_holdout_long
```

기본 `--source-dir`은 `outputs/selector_compare320_long`이며, 기존 dev의
동결 설정과 출처를 확인하는 데 사용한다. dev 정확도나 D의 과거 QA 결과를
holdout 문항의 결과로 복사하지 않는다. `--split eval`이 기본이며 전체
39영상·117문항으로 고정한다. 대상 수를 줄이는 옵션은 dev smoke에서만 허용한다.

구현 점검을 위한 dev 한 영상·한 문항 smoke는 보류 실행과 다른 폴더를 쓴다.
준비와 QA 양쪽에 같은 대상 옵션을 전달한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_holdout --split dev --max-videos 1 --max-questions 1 --phase prepare --output-dir outputs/selector_holdout_dev_smoke
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_holdout --split dev --max-videos 1 --max-questions 1 --phase evaluate --output-dir outputs/selector_holdout_dev_smoke
```

최종 전달 파일은 다음 경로다.

```text
C:\Users\dudgh\experimental-code\outputs\selector_compare320_holdout_long\handoff.json
```

고정 프로토콜, 문항별 결과 및 선택 계획도 같은 실행 폴더에 보존한다.
성능 결과를 확인하기 전에 코드·대상·설정을 확정해야 한다.

## 결과를 읽는 방법

주 비교는 **`D_minus_uniform`**, 보조 비교는 `D_minus_frame_top`과
`D_minus_temporal_bin`이다. 문항별 6개 선택지 순서의 정답 여부를 평균한 뒤
문항마다 같은 가중치를 준다. 순서 6개를 독립 문항으로 세지 않는다.

정확도 및 조건 간 차이의 95% 신뢰구간은 동일한 영상을 묶어 재표집하는
**paired video bootstrap 5,000회**로 계산한다. 같은 영상에 속한 문항의
상관관계와 조건 간 짝을 유지한다. 개선·하락·동일 문항 수는 D와 비교군의
문항별 순서 평균을 비교해 센다. 한 번의 대표 선택지 순서만으로 세지 않는다.

미리 고정한 주 비교에서 95% 신뢰구간 하한이 0보다 크면 이 보류 표본에서
D의 개선을 뒷받침하는 결과로 보고한다. 0을 포함하면 개선이 없다고 단정하지
않고, 이번 표본으로 개선을 확정하기 어렵다고 보고한다. 보조 비교의 구간은
다중비교 보정 전 개별 구간이므로 세 비교를 같은 강도의 확증으로 제시하지
않는다. 전체 Video-MME 공식 점수나 다른 데이터셋 성능으로 일반화하지 않는다.

## 비용 해석

보고하는 시간·최대 GPU 메모리·시각 토큰은 **QA 비용**이다. 이번에는 새
영상의 후보와 CLIP 특징을 준비해야 하지만, 특징 추출·프레임 선택 시간은
QA 비용에 합산하지 않는다. 문항/영상 내에서 공유하거나 저장한 특징을
재사용하더라도 그 비용이 0이라는 뜻은 아니다.

Uniform은 CLIP 특징이 없어도 실행할 수 있으므로, QA 비용만 보고 D와
Uniform의 전체 처리 비용이 같다고 결론 내리지 않는다. 같은 문항에서 실제
입력 픽셀·모델·프롬프트·선택지 순서가 같은 경우만 QA 결과를 공유한다.
최대 2,808회는 공유 전 상한이며 실제 새 실행 수는 더 적을 수 있다.

정확도·차이·개선/하락 수와 함께 QA 시간 중앙값, 최대 GPU 메모리, 문항당
시각 토큰을 전달한다. dev의 과거 D 측정값과 이번 holdout 측정값을 합쳐
속도 우위를 주장하지 않는다.

## 구현 검증 기록 (2026-09-29)

- 로컬 ZIP에서 보류 긴 영상 39개를 실제 추출하고 크기·CRC를 검증했다.
  실행기 입력 검사는 `ready=true`, 39영상·117문항, 최대 2,808회 QA를
  확인했다. holdout의 CLIP 준비 및 전체 QA는 아직 실행하지 않았다.
- 전체 테스트 **191개**가 통과했다. 신규 테스트는 실행기 24개, 영상 준비
  도구 10개다. 신규 Python 파일 네 개는 Ruff 검사도 통과했다.
- 기존 dev 비교 실행기의 입력 검사도 통과했다. 기존 알고리즘과 완료된 dev
  결과 파일은 변경하지 않았다.
- 실제 개발 영상 `895`의 문항 `895-1`에서 후보 5,081개를 디코딩하고
  CLIP으로 네 선택기를 다시 계산했다. 모든 방법의 16장 선택이 기존 결과와
  정확히 일치했다.
- 같은 문항에서 네 조건 × 여섯 순서, **새 QA 24회**를 실행했다. 선택된
  픽셀·타임스탬프·모든 순서의 예측이 기존 dev 결과와 일치했으며, 선택지
  점수의 최대 절대 차이는 네 조건 모두 **0**이었다.
- 완료 후 `--phase summarize`로 재집계했고 추가 QA는 0회,
  `handoff.json`의 파일 해시도 그대로였다.

점검 산출물은 `outputs/selector_holdout_dev_smoke/`에 있다.
`qa_reproduction.json`은 조건별 재현 확인 기록이고, `handoff.json`에는
`evaluation_role=dev_smoke`가 명시된다. 이 한 문항 결과를 보류셋 성능으로
사용하지 않는다.
