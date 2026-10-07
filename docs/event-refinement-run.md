# 이벤트 할당 × 이벤트 내부 top-k 실험

기존 D2 이벤트 경계와 QA 모델/프롬프트/선택지 순서를 고정하고,
모든 조건에서 서로 다른 프레임 16장을 시간순으로 입력합니다.
캡션은 생성하거나 입력하지 않습니다.

| 조건 | 이벤트별 할당 | 이벤트 내부 선택 |
|---|---|---|
| `D` | 기존 D 그대로 | 기존 균등 선택, 1장이면 중앙 |
| `D_topk` | 기존 D 그대로 | 질문–프레임 CLIP cosine top-k |
| `focus_uniform` | 상위 8개 이벤트에 집중 | 균등 선택 |
| `focus_topk` | 상위 8개 이벤트에 집중 | 질문–프레임 CLIP cosine top-k |

`D`는 검증된 기존 결과를 재사용합니다. 나머지 조건도 같은 질문에서
16장의 인덱스가 완전히 같으면 QA 결과를 재사용합니다.
프레임 선택에는 질문 문장만 사용하며 선택지, 정답, 사람의 근거 표시는
사용하지 않습니다. 선택지와 정답은 기존 QA 평가 단계에서 처리합니다.

## 할당과 내부 선택의 정의

- 중요도는 기존 D에 저장된 **이벤트 평균 특징과 질문의 cosine**입니다.
  이번 실험에서는 이벤트 점수 정의를 바꾸지 않습니다.
- 집중 조건은 상위 8개 이벤트만 할당 후보로 남긴 뒤 기존 temperature와
  softmax/용량 제한/최대 나머지 정수 배분을 그대로 적용합니다.
  후보 밖 이벤트는 0장입니다. 후보 이벤트 수가 8보다 적으면 모두 사용합니다.
- 상위 이벤트의 총 프레임 용량이 16 미만이면 다음 순위 이벤트를 추가합니다.
  실제 후보 이벤트와 용량 때문에 추가된 이벤트 수를 계획 파일에 기록합니다.
  후보로 남아도 최종 배분이 0장일 수 있습니다.
- top-k의 k는 해당 이벤트에 배정된 장수입니다. 이벤트 할당을 유지하는
  `D_topk`에서는 k=1인 이벤트가 여전히 1장입니다. `focus_topk`에서
  여러 장을 배정받은 이벤트의 다중 프레임 선택을 시험합니다.
- 동점은 앞선 이벤트/프레임부터 선택합니다. 모델 입력은 항상 시간순입니다.
- `--min-gap-seconds 0`이 기본 순수 top-k입니다. 양수를 지정하면 같은
  이벤트에서 점수순으로 선택하되 선택된 프레임과의 시간 간격을 확인합니다.
  간격 제한 때문에 k장을 채우지 못하면 나머지를 점수순으로 채워
  총 16장을 유지하고 `gap_fallback_frames`에 기록합니다.
  시간 간격은 시각적 다양성을 보장하지 않습니다.

## 실행

PowerShell 작업 경로는 `C:\Users\dudgh\experimental-code`입니다.
모든 명령은 기존 `.venv`를 사용합니다. 새 모델이나 설치가 필요하지 않습니다.

입력 검사만 수행(영상 디코딩·GPU 추론·결과 폴더 생성 없음):

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.event_refinement_dev --output-dir outputs/event_refinement_top8 --check-inputs
```

전체 실행(기존 60영상/180문항, 조건마다 동일한 6개 선택지 순서):

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_refinement_dev --output-dir outputs/event_refinement_top8
```

중단 후 같은 설정으로 재개:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_refinement_dev --output-dir outputs/event_refinement_top8 --resume
```

기존 D는 재사용하므로 새 QA 호출은 최대 `180 × 3 × 6 = 3,240`회입니다.
동일 선택 재사용으로 실제 호출 수는 줄어들 수 있습니다.
QA 한 순서마다 저장하며 완료한 문항/영상은 재개 시 건너뜁니다.
중단된 영상은 다시 디코딩하지만 저장된 선택 계획과 완료된 QA는 재사용합니다.
설정·소스 결과·코드가 달라지면 같은 폴더의 재개를 거부합니다.

먼저 856 영상의 3문항으로 실행을 확인하려면 별도 폴더를 사용합니다:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_refinement_dev --video-id 856 --output-dir outputs/event_refinement_smoke856
```

선택적 후속 비교: 최소 2초 간격을 적용한 top-k. 이것은 위 기본 실험과
별도 설정이며, 결과를 섞지 않습니다. 2초는 최적화된 값이 아닙니다.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_refinement_dev --min-gap-seconds 2 --output-dir outputs/event_refinement_top8_gap2
```

`--top-events`로 후보 이벤트 수를 바꿀 수 있지만 기본 8은 사전 고정한
탐색적 설정입니다. 결과를 보고 최적값을 고른 뒤 같은 dev 성능을
확증적 결과로 보고하지 않습니다. 값이 달라지면 새 출력 폴더를 사용하세요.

## 전달할 결과

실행 완료 후 **`outputs/event_refinement_top8/handoff.json`**을 전달하면 됩니다.
프로토콜, 전체/영상 길이별 결과, 문항별 변화, 선택 진단,
856-1의 알려진 근거 후보 1035/1036 포함 여부를 담습니다.
이 두 인덱스 검사는 선택이 모두 끝난 뒤 보고에만 사용합니다.
모든 문항의 근거 정답률이나 근거 재현율을 의미하지 않습니다.

- `summary.md`, `summary.json`: 전체 결과와 영상 단위 bootstrap CI.
- `question_metrics.csv`: 문항별 선택지 순서 평균 정확도와 조건 차이.
- `selections.csv`: 조건별 이벤트 할당, 프레임 인덱스/시간, 중복도 등.
- `results.json`: 모든 조건의 선택 계획과 선택지별 QA 점수.
- `plans/`: 각 후보 프레임의 질문 cosine, 선택 계획, 체크포인트 해시.
- `progress.json`: 완료 문항 수. `status=complete`가 최종 완료입니다.

주 비교는 `both = focus_topk − D`입니다.
`within_only = D_topk − D`, `allocation_only = focus_uniform − D`로
개별 효과를 보고, `interaction`으로 집중 할당에 따라 내부 선택 효과가
달라졌는지 봅니다. 6개 순서는 독립 문항이 아닙니다. 먼저 문항별 평균을
구한 뒤 영상 단위로 5,000회 bootstrap합니다. 영상 하나의 smoke 실행에는
CI가 없습니다. 이 실험은 실패 사례를 확인한 뒤 설계한 dev 탐색 실험입니다.

선택 cosine 상승이나 프레임 중복도 감소는 근거 포착의 대리 지표일 뿐입니다.
실제 근거 포착 개선은 별도 사람 검토로 확인해야 하며, 근거를 골라도
QA가 오답일 수 있으므로 정확도와 프레임 선택 결과를 함께 확인합니다.
