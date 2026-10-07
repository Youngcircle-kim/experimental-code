# 이벤트 중요도 점수 비교 + 긴 구간 검토

소스는 완료된 `outputs/segment_cap128_long`입니다. 기본 범위는 그 실험과
같은 긴 영상 20개·60문항입니다. 기존 D의 구간 경계, 총 16장, QA 설정,
선택지 순서, 배분 온도, 배분 알고리즘, 이벤트 내부 균등 선택을 고정합니다.

| 조건 | 이벤트 점수 |
|---|---|
| `pooled_mean` | 기존: 프레임 특징들을 평균한 뒤 정규화한 벡터와 질문의 cosine |
| `frame_max` | 구간 내 프레임–질문 cosine의 최댓값 |
| `frame_topn_mean` | 구간 내 프레임–질문 cosine 상위 3개의 평균 |

상위 3장은 **중요도 점수 계산용**입니다. QA에 그 3장을 직접 넣는 것이
아닙니다. QA 프레임은 모든 조건에서 배분 장수에 따른 균등 선택이며,
1장일 때 중앙 프레임입니다. 후보가 3장 미만인 구간은 있는 후보를 모두
평균합니다. 기존 평균 특징 cosine은 프레임 cosine의 산술평균과 다릅니다.

원래 D 점수 재현을 수치 허용오차 내에서 확인하고, 원래 할당과 최종
인덱스는 정확하게 재현되어야 합니다. 기존 D QA를 대조군으로 재사용합니다.
새 조건도 같은 문항에서 인덱스가 동일하면 QA 결과를 재사용합니다.

## 1. 준비 및 긴 구간 검토 화면 생성

PowerShell 작업 경로: `C:\Users\dudgh\experimental-code`.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_score_dev --phase prepare --output-dir outputs/event_score_cap128
```

이 단계는 영상 디코딩과 CLIP만 실행합니다. **QA 모델을 로드하지 않고,
QA나 캡션 추론도 하지 않습니다.** 질문 임베딩에는 질문 문장만 사용합니다.

생성 파일:

- `outputs/event_score_cap128/index.html`: 로컬 브라우저에서 열 검토 목록.
  전체 대상 중 60초를 넘는 구간을 길이순으로 최대 10개 선택합니다.
  각 구간마다 12장 균등 미리보기와 모든 후보 경계의 D2 변화 점수,
  임계값 0.15를 표시합니다. 질문·정답·모델 예측은 표시하지 않습니다.
- `long_intervals.csv`: 검토 구간의 시작·끝 시각, 길이, 후보 범위.
- `event_ranks.csv`: 문항·조건·이벤트별 점수, 순위, 배정 장수,
  실제 선택 인덱스와 점수가 가장 높은 후보.
- `prepare_summary.json`: 준비 완료 통계.
- `plans/`: 검증용 해시가 포함된 전체 선택 계획과 점수.

미리보기 프레임 사이에서 일어난 변화는 놓칠 수 있습니다. 영상에서 변화가
없다고 결론 내리기 전에 필요 구간을 원 영상에서도 확인합니다. 구간 번호와
후보 인덱스는 0부터, 순위는 1부터 시작합니다.

준비 중 중단 시 같은 명령에 `--resume`을 추가합니다. 완료 영상의 디코딩,
CLIP, 선택은 반복하지 않습니다. 변경한 검토 설정도 프로토콜에 기록되므로
`--review-limit`, `--preview-count`, `--long-seconds` 등을 바꾸면 새 폴더를
사용하세요. 예: 영상 하나의 점검은 `--video-id 816`과 별도 출력 폴더.

## 2. 알려진 근거 프레임의 구간 순위 확인 (선택적, 추론 없음)

이전에 사람이 확인한 856-1 후보 1035·1036을 예제로 제공합니다.
준비가 끝난 뒤 다음 명령을 실행합니다:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.event_score_dev --phase evidence --output-dir outputs/event_score_cap128 --evidence-file examples/event_evidence_856.json
```

`evidence_audit_<주석해시>.json/.csv`에 각 조건의 근거 구간 순위, 구간 할당,
정확히 그 후보가 선택됐는지, 같은 구간에서 실제로 선택된 프레임을 기록합니다.
주석 해시별로 파일을 분리하므로 기존 검토를 덮어쓰지 않습니다.

주석은 이 단계에서만 읽고 **선택·배분·QA 입력에 사용하지 않습니다**.
다른 문항을 추가하려면 아래 형식의 JSON을 별도 파일로 만들면 됩니다:

```json
{
  "annotations": [
    {"video_id": "856", "question_id": "856-1", "candidate_indices": [1035, 1036]}
  ]
}
```

이는 지정된 후보의 진단일 뿐 전체 근거 재현율이 아닙니다. 다른 후보도
동등한 근거를 포함할 수 있으며, 해당 후보를 포함해도 QA가 틀릴 수 있습니다.
여러 후보가 함께 필요한지 또는 각각 대체 가능한지 자동 판단하지 않습니다.

## 3. 준비된 계획으로 QA 비교

검토 후 QA를 진행하려면:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_score_dev --phase evaluate --output-dir outputs/event_score_cap128
```

새 QA 호출은 최대 `60문항 × 2새조건 × 6순서 = 720회`입니다.
영상은 다시 디코딩하지만 구간·점수·프레임 선택은 준비 결과를 재사용합니다.
중단되면 evaluate 명령을 그대로 다시 실행하면 됩니다. 선택지 순서별로
저장하므로 완료된 QA와 문항을 재사용합니다.

완료 후 전달할 파일은 **`outputs/event_score_cap128/handoff.json`**입니다.
근거 순위 검토를 했다면 `evidence_audit_*.json`도 함께 전달하세요.
원시 결과는 `results.json`, 문항별 요약은 `question_metrics.csv`에 저장합니다.

주 비교는 `frame_topn_mean − pooled_mean`, 보조 비교는
`frame_max − pooled_mean`, `frame_topn_mean − frame_max`입니다.
먼저 문항별 6개 순서의 정확도를 평균한 뒤 영상 단위 bootstrap으로 CI를
계산합니다. 반복 선택지 순서는 독립 문항이 아닙니다.

최댓값과 상위 N개 평균은 긴 구간에서 우연히 높은 점수를 얻기 쉬울 수
있습니다. 또한 점수 분산이 바뀌면 고정 온도에서도 할당이 달라집니다.
이 실험은 그러한 효과를 포함한 점수 정책 비교이며 별도 점수 교정을 하지
않습니다. 근거 구간 순위 개선과 QA 정확도 개선은 분리해 해석합니다.

이번 실행기는 긴 구간을 자동으로 추가 분할하지 않습니다. 실제 경계 누락을
검토한 뒤 필요할 때 별도 실험으로 수행합니다.

## 입력 검사만 하기

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.event_score_dev --phase prepare --output-dir outputs/event_score_cap128 --check-inputs
```

모델을 로드하거나 출력 폴더를 만들지 않습니다. 모든 단계는 같은 소스,
대상 영상과 설정을 사용해야 하며, 소스/코드 변경 시 기존 폴더의 재개를
거부합니다. `--top-n` 기본값은 3이며 변경 실험은 별도 폴더를 사용하세요.
