# 중앙 한 장의 근거 충분성 검토

`event_caption_pilot.midpoint_review`는 실제 D 입력과 같은 구간의 다른 후보를
비교하는 오프라인 HTML 검토 도구다. 새 모델 호출이나 QA 재추론은 하지 않는다.
영상은 원 실험과 같은 설정으로 CPU 디코딩하고, 후보 픽셀과 시각의 해시가
원 기록과 일치하는지 확인한다. 따라서 긴 영상은 생성에 시간이 걸릴 수 있다.

## 1. 먼저 검토할 문항 목록 확인

저장소 루트 `C:\Users\dudgh\experimental-code`에서:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.midpoint_review export --output-dir outputs/midpoint_review_long --limit 12 --list-only
```

기본값은 긴 영상, D−C 비교다. 개선·악화·양쪽 정확도 0·그 외 동일 점수의
네 집단을 순환하며 문항을 고른다. 각 집단 안에서는 정확도 차이 절댓값이 큰
문항부터 고른다. 집단이 비면 나머지에서 채운다. 이 표본은 원인 탐색용이며,
전체 문항의 병목 발생률 추정에 사용하지 않는다. 목록 확인은 디코딩하지 않는다.

## 2. 검토 화면 만들기

빠른 시작: 특정 문항 하나의 모든 후보를 저장한다.

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.midpoint_review export --output-dir outputs/midpoint_review_856 --video-id 856 --question-id 856-1 --all-candidates
```

기본 표본 12문항의 구간별 미리보기를 만들려면:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.midpoint_review export --output-dir outputs/midpoint_review_long --limit 12 --preview-count 7
```

- 결과 `index.html`을 브라우저에서 연다. 인접한 `frames` 디렉터리도 필요하다.
- `--all-candidates`: 선택한 영상의 후보 전체를 PNG로 저장하고 구간별로 펼쳐 볼 수 있다.
  긴 영상 한 개에 수천 개 파일이 생길 수 있다. 같은 영상의 여러 질문은 이미지를 공유한다.
- 기본 미리보기: 구간마다 최대 7개 균등 후보에 실제 선택 프레임을 추가한다.
  구간 길이가 길어도 7개만 보므로 **여기에 없다는 이유로 근거가 없다고 판단할 수 없다.**
- `--control B`: 검토 문항의 선정과 QA 차이 표시를 D−B로 변경한다.
  화면의 분석 대상 입력은 계속 D이다.
- `--duration all`: 짧은 영상도 포함한다. 특정 짧은 영상 ID를 지정할 때도 필요하다.
- 새 출력 디렉터리를 지정해야 한다. 기존 검토 파일을 덮어쓰지 않는다.
- 실패한 export 디렉터리에는 일부 PNG가 남을 수 있다. 다른 출력 이름으로 다시 실행한다.

## 3. 화면에서 기록하기

먼저 질문·선택지와 D가 실제로 받은 전체 프레임을 본다. 정답과 QA 점수는
접힌 영역에 있으며 필요한 시점에 펼친다. 프레임은 원 후보 해상도를 유지한다.
확대 표시해도 원 입력에서 소실된 정보가 복원되는 것은 아니다.

두 수준의 판단을 구분한다.

1. **전체 선택 입력 충분성:** 실제 D 프레임들을 함께 봤을 때 답을 판단할 근거가 충분한가?
2. **구간 중앙 프레임 충분성:** 해당 구간에서 필요한 근거를 중앙 한 장이 담고 있는가?
   모든 구간이 단독으로 질문 전체에 답할 수 있어야 한다는 의미가 아니다.

각 구간에는 실제 선택 프레임, 비교 후보, 배정 장수, 후보 인덱스와 시각이 표시된다.
1장인 구간에는 실제 정수 중앙 인덱스인지도 기록된다. 0장·다중 프레임 구간도
함께 표시해 다른 실패 원인과 구분할 수 있게 했다.

중앙 프레임이 불충분하고 다른 후보에서 추가 근거가 보이면:

- 중앙 프레임 충분성: `no`
- 다른 후보의 추가 근거: `yes`
- 근거 인덱스: 실제 확인한 후보 인덱스를 쉼표로 입력
- 메모: 무엇이 빠졌는지, 한 장으로 대체 가능한지, 전후 장면이 모두 필요한지 기록

추가 근거가 `yes`라면 실제 D 선택에 포함되지 않은 후보 인덱스가 최소 하나 필요하다.
검토하지 않았거나 판단하기 어렵다면 `unknown`으로 둔다. 질문과 관련 없는
구간의 중앙 판단은 `na`로 둘 수 있다. 같은 구간의 후보만 입력할 수 있다.
다른 구간의 근거는 그 구간에 기록한다.

**화면에서 판단했다고 로컬 파일에 자동 저장되지는 않는다.** JSON 내보내기
버튼으로 `midpoint_annotations_<보고서ID>.json`을 다운로드한다.
아래 집계 명령 예시에서는 이 파일을 `annotations.json`으로 이름을 바꿔 사용한다.
내보낸 JSON은 같은 검토 화면에서
다시 불러올 수 있다. 검토 도중에도 주기적으로 내보내면 진행 내용을 보존할 수 있다.

## 4. 사람이 기록한 결과 집계

다운로드한 JSON을 원하는 위치에 저장한 뒤 실제 경로를 사용한다.

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.midpoint_review assess --report outputs/midpoint_review_856/review.json --annotations outputs/midpoint_review_856/annotations.json --output-dir outputs/midpoint_review_856_assessed
```

`assessment.json`, `assessment.csv`를 생성한다. 역시 새 출력 폴더를 사용한다.

| 상태 | 해석 |
|---|---|
| `human_identified_midpoint_miss` | 사람이 중앙 한 장의 불충분성과 미선택 후보의 추가 근거를 기록함 |
| `human_identified_omitted_event_evidence` | 0장 구간의 후보에서 사람이 근거를 발견함 |
| `human_judged_midpoint_sufficient_for_event` | 사람이 해당 구간의 중앙 프레임을 충분하다고 판단함 |
| `midpoint_insufficient_alternative_unconfirmed` | 중앙은 불충분하지만 대안 근거는 아직 확인되지 않음 |
| `human_review_pending` | 나머지 경우; 자동으로 성공/실패를 결정하지 않음 |

추가 프레임에서 근거가 발견돼도 QA가 정답을 낼지는 별개의 질문이다.
현재 도구는 사람이 확인한 사례를 모으고, 다음 통제 실험의 대상을 정하는 데 쓴다.
모델의 정확도, 근거 recall, 인과적인 성능 개선을 자동 산출하지 않는다.

## 검증 범위

기존 선택 계획·프로토콜 연결, 질문/선택지/정답과 trial 대응, D 후보 인덱스·시각,
원 디코딩 픽셀 해시, 검토 JSON의 보고서 ID와 근거 인덱스 범위를 검사한다.
다른 보고서에서 내보낸 판단 파일은 거부한다. manifest는 현재 로컬 버전을
읽으며, 해시를 검토 보고서에 함께 기록한다. 원본 영상이나 QA 결과는 수정하지 않는다.
