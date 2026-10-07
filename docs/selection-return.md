# 동일 쌍 재분석과 선택 실험 복귀

## 재분석

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.paired_reanalysis --input outputs/videomme_dev_frozen/results.json --output-dir outputs/videomme_paired_reanalysis_new
```

생성 답변을 판독할 수 있는 문항/순서 쌍에서 생성과 채점 모두 평가합니다.
그 다음 각 문항의 순서 평균을 구하고 원본 영상 단위로 paired bootstrap
5000회를 수행합니다. 전체 6순서가 모두 판독된 문항 분석도 별도 저장합니다.
미판독을 모두 오답/정답으로 놓은 값은 누락 응답에 대한 민감도 경계이며
새 추정 정확도가 아닙니다. 판독 가능한 부분집합 선택 편향은 남습니다.

현재 재분석: 1080쌍 중 892쌍 판독, 173문항/60영상. 문항 평균 생성
62.36%, 동일 쌍 채점 49.81%, 차이 +12.55%p, 영상 단위 95% CI
[+6.27,+18.99]%p. 전체 순서 판독 118문항에서도 차이는 +11.86%p입니다.
공식 benchmark 결과나 생성 방식의 보편적인 우월성으로 해석하지 않습니다.

## dev 선택 실험

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.selection_dev --baseline-dir outputs/videomme_dev_frozen --output-dir outputs/videomme_selection_dev_new
```

- QA는 기존 전체 선택지 로그우도, 길이 정규화, 모델/프롬프트 그대로 고정.
- uniform/TOP/BIN 각 16장. TOP/BIN은 질문 텍스트만 사용합니다.
- 같은 후보 풀, 같은 선택지 순서 6가지. 보류 eval은 읽지 않습니다.
- uniform은 기존 결과의 후보 해시·선택 인덱스·순서·config·관련 코드·모델
  메타데이터가 일치할 때 재사용합니다. 생성 응답은 새 선택 비교에 섞지 않습니다.
- 영상별 CLIP 특징을 한 번 계산하고 질문별로 TOP/BIN을 선택합니다.
- 조건/문항 완료마다 checkpoint 저장, 진행률은 progress.json에 기록합니다.
- 방법별 순서 평균을 문항 안에서 구한 뒤 paired 영상 bootstrap으로
  TOP−uniform, BIN−uniform을 분석하고 영상 길이별 결과도 제공합니다.

중단 후 재개:

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.selection_dev --baseline-dir outputs/videomme_dev_frozen --output-dir outputs/videomme_selection_dev_v2 --resume
```

현재 실행 디렉터리는 `outputs/videomme_selection_dev_v2`입니다.
새 실행은 비어 있는 새 디렉터리를 사용하고, 실행 중에 같은 디렉터리로
재개 명령을 동시에 실행하지 않습니다.
관련 코드/config/입력 변경 후에는 같은 실행으로 재개하지 않습니다.
완료 결과는 results.json과 summary.md에 저장됩니다. 이 실험은 질문 기반
프레임 선택(H1)부터 다시 확인하는 단계이며 이벤트/캡션 비교(H2/H3)는
아직 포함하지 않습니다. 생성 QA도 유망하지만 답 판독 누락 문제를 가진
상태에서 primary QA까지 동시에 바꾸면 선택 효과와 혼동되므로, 이번에는
기존 primary QA를 유지합니다. 이는 QA 방식에 조건부인 선택 효과입니다.
