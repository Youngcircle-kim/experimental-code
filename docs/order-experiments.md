# 후속 실험 1·2·3

모든 명령은 프로젝트 루트 PowerShell에서 실행합니다. 가중치는 기존의
고정된 버전을 사용합니다. 출력 디렉터리는 새 이름이어야 합니다.

## 1. 잘린 sparse 생성만 길이 상한 증가

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.order_experiments length --results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --previous-comparison outputs/qa_compare_sparse_dense/comparison.json --max-new-tokens 256 --output-dir outputs/order_length_256
```

이전 comparison.json의 sparse 중 output_token_limit_reached=true인 항목만
재실행합니다. 문장이 길어 보여서 잘렸다고 추정하지 않습니다. 해당 항목이
없으면 중단합니다. 프롬프트 문자열/입력 토큰을 이전 실행 및 현재 채점과
대조하며, 후보 RGB/timestamp와 문항도 원래 결과와 대조합니다.
생성 상한만 늘리고 같은 순서의 로그우도를 다시 계산합니다.

설명문은 선택지 전체 텍스트와 일치하지 않으므로 자동 답 판독이 null일 수
있습니다. results.json의 trials[].generation.text에서 명시적 최종 답을
직접 확인하세요. null은 오답으로 계산하지 않습니다. limit 도달은 길이
상한 도달 표시이며, 정확히 그 지점에서 EOS가 나온 가능성까지 배제하지는 않습니다.

## 2. dev 문항의 선택지 순서 민감도

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.order_experiments dev --config configs/real.rtx3060.pinned.json --manifest data/my_vqa_manifest.json --max-new-tokens 256 --output-dir outputs/order_dev
```

우선 기존 예제로 동작 확인하려면 manifest를
`data/perception_smoke/manifest.json`으로 바꾸세요. 이 예제의 dev는 영상
1개/문항 2개이므로 유형별 평가에는 부족합니다. 새 dev 데이터는 사용자가
준비해야 하며 자동 다운로드하거나 eval을 dev로 옮기지 않습니다.
기존 manifest 검증 규칙 때문에 dev/eval을 모두 포함해야 하지만, 이 명령은
dev 문항에만 추론합니다. 모든 split의 파일을 읽고 누출 검증은 수행합니다.

문항별 균등 16프레임(설정의 frame_budget)을 고정합니다. 3지선다는 순서
6가지 전부, 4지선다는 원래 순서+시드로 고정한 다른 5가지를 사용합니다.
2지선다는 2가지이며 그 밖의 선택지 수는 명시적으로 거부합니다.

질문 유형을 따로 기록하려면 아래 형식의 JSON을 만들고
`--question-types data/dev_question_types.json`을 추가하세요.

```json
{
  "video_7766/0": "action",
  "video_7766/4": "temporal_order"
}
```

위 유형은 형식 예시이며 실제 문항 내용을 보고 작성해야 합니다.
유형 미지정은 unlabeled로 표시하며 글자/물체/행동/순서 분류를 추측하지
않습니다. 결과에는 다음 지표의 문항별 값, 문항 평균, 유형별 평균을 기록합니다.

- scoring_answer_changed: 순서에 따라 답변 내용이 바뀌는가
- scoring_order_mean_accuracy: 순서 평균 정확도
- generation_parse_rate: 생성의 정확 일치 판독률
- generation_accuracy_on_parsed: 판독 가능한 생성만의 정확도
- generation_answer_changed_on_parsed: 판독 가능 응답이 2개 이상일 때 변경 여부
- method_disagreement_on_parsed: 판독 가능 생성과 채점 간 불일치율
- generation_limit_rate: 생성 상한 도달 비율

각 평균의 eligible_questions도 제공합니다. 판독률을 함께 보고해야 하며
일부만 판독된 생성 정확도를 전체 채점 정확도와 단순 비교하지 마세요.

## 3. 동일 순서 묶음으로 8개 프레임 선택 조건 비교

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.order_experiments ablation --results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --output-dir outputs/order_ablation
```

기존 실험에서 저장한 각 조건의 selected_indices를 재사용합니다. 이벤트
분할·캡션·dev 정규화·선택을 다시 계산하지 않으며 QA의 선택지 순서만
바꿉니다. 모든 조건에서 문항별 같은 순서 묶음을 사용하고 원래 선택지
인덱스로 답을 복원합니다. 동일한 frame_budget과 원본 후보 해시를 검사합니다.
새 데이터 실험은 먼저 기존 run_pilot으로 8조건 결과를 만든 뒤 그 결과를
이 명령에 전달하세요. 이번 D0 예제로는 D2 경계의 효과를 판단할 수 없습니다.

QA 방식은 기존 full-option 로그우도 채점으로 고정됩니다. dev 결과를 보고
자동으로 평가 방식이나 유리한 순서를 고르지 않습니다.

순서별 정오 → 문항별 순서 평균 → 문항 가중 정확도 순으로 집계합니다.
paired 차이의 신뢰구간은 원본 영상 단위 bootstrap으로 계산합니다.
같은 문항의 순서 반복이나 여러 방법의 호출을 독립 표본으로 세지 않습니다.
영상 1개면 신뢰구간은 null이며 다중 비교 보정은 하지 않습니다.

각 실행의 summary.md와 results.json을 확인하세요. 실제 모델 추론 비용은
dev의 경우 문항×순서×(생성+선택지 채점), ablation의 경우 문항×순서×8조건의
QA에 해당합니다. 결과를 얻기 위해 eval 임계값이나 선택지 순서를 튜닝하지 마세요.
