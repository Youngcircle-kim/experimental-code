# 오답 근거 진단

## 동일 프롬프트 채점 비교 + 촘촘한 프레임 진단

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.qa_comparison_probe --results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --video-id video_1726 --question-id 12 --indices 6 10 --output-dir outputs/qa_compare_sparse_dense
```

`sparse`는 #6/#10, `dense`는 #6/#7/#8/#9/#10입니다. 후보를 원본과
같은 해상도로 복원하고 해시를 확인합니다. 추가 디코딩 밀도나 해상도 변경은
없습니다. 각 구성에서 선택지 순서 6가지를 모두 실행합니다.
각 순서마다 동일한 질문·선택지·이미지로 자유 생성과 기존 선택지 로그우도
채점을 실행하며, 두 경로의 실제 프롬프트 및 입력 토큰 일치를 검증합니다.
정답은 두 모델 입력에 들어가지 않으며 응답은 서로 독립적입니다.

`comparison.md`는 12행 비교표입니다. sparse에서 생성/채점이 같은지,
순서를 바꾸어도 같은 선택지 내용을 답하는지, dense에서 결과가 달라지는지
차례로 확인하세요. 생성 답변은 선택지 전체 텍스트와 대소문자/바깥 공백을
제외하고 정확히 일치할 때만 자동 판독합니다. 설명이나 번호로 답하면 null로
표시하므로 원문을 확인해야 합니다. null을 오답으로 집계하지 않습니다.

`comparison.json`은 실제 프롬프트, 입력/생성 토큰 ID, 선택지별 target 토큰,
target/predictor 위치(0부터 시작), 토큰별 로그우도, 합계/평균 점수를 포함합니다.
실제 평가 점수는 기존 qa_length_normalize 설정을 유지합니다.
토큰 접합으로 프롬프트가 바뀌거나 두 경로의 입력이 다르면 오류로 종료합니다.

이는 평가 후 선택한 한 문항의 진단입니다. 2장과 5장은 프레임 예산이 다르며,
선택지 순서 6회는 독립적인 평가 문항 6개가 아닙니다. 생성과 teacher-forcing은
계산 경로가 달라 동일 프롬프트여도 수치 차이가 발생할 수 있습니다.
결과만으로 채점 버그나 알고리즘 성능 향상을 확정하지 마세요.

## 두 프레임 Qwen 진단 (#6, #10)

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.frame_probe --results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --video-id video_1726 --question-id 12 --indices 6 10 --output-dir outputs/frame_probe_6_10
```

기존 가상환경과 다운로드한 모델을 사용해 GPU 추론을 수행합니다.
각 프레임의 독립 글자 판독, 두 장의 변화 설명, 원래 질문의 자유 응답,
기존 객관식 로그우도 채점을 순서대로 실행합니다. 각 호출은 독립적이며
이전 응답이나 정답을 다음 입력에 주지 않습니다. 생성 상한은 256토큰이고
실제 후보 해상도와 모델 버전은 기존 실행 설정을 유지합니다.

`probe.md`에서 설명을 읽고 `probe.json`에서 프롬프트·입력 인덱스·점수·
모델 정보를 확인하세요. 단일 이미지부터 오독하면 글자 인식을, 단일 판독은
맞지만 변화 설명이 틀리면 두 시점 비교를 점검합니다. 자유 응답과 객관식
답변이 다르면 프롬프트/채점 방식에 대한 추가 진단이 필요합니다.
이 테스트만으로 원인을 확정하지 않습니다. 사람이 사후 선택한 두 장과
변경된 프롬프트를 사용하므로 16프레임 벤치마크 정확도와 직접 비교하지 않습니다.
출력 폴더가 이미 있으면 새로운 이름을 지정하세요.

프로젝트 루트의 PowerShell에서 실행합니다. Qwen/CLIP 추론을 다시 하지
않습니다. MP4 재디코딩에는 기존 video 의존성(opencv-python)이 필요합니다.
결과의 Config로 후보 풀을 복원하고 RGB+timestamp 해시를 대조합니다.
해시가 다르면 다른 입력을 같은 실험으로 분석하지 않도록 실패합니다.

## 1. 기존 results.json으로 보고서 생성

아래 결과 파일 경로를 실제 D0 실행 폴더로 바꾸세요.

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.qa_diagnostics --results outputs/real_rtx3060/실행폴더/results.json --output-dir outputs/qa_review_d0
```

입력 manifest 위치가 바뀌었다면 `--manifest data/.../manifest.json`을
추가할 수 있습니다. 영상은 그 manifest 기준으로 읽으며 해시가 일치해야 합니다.
출력 폴더는 새 이름이어야 합니다. 기존 검토 파일을 덮어쓰지 않습니다.

`outputs/qa_review_d0/qa_review.html`을 브라우저에서 엽니다.

- 문항별 예측·선택지 점수·정답 점수와 최고 오답 점수의 차이
- 전체 후보 프레임과 해당 프레임을 선택한 조건 표시
- V/T/V+T별 시간순 선택 프레임

PNG는 모델에 들어간 후보 해상도 그대로입니다. 원본 MP4의 고해상도
프레임이 아닙니다. 글자를 못 읽겠다면 원본 MP4와 비교해 촬영 문제인지,
리사이즈/후보 추출에서 정보가 소실됐는지 따로 확인하세요.

## 2. 사람이 근거 판독 결과 기록

같이 생성한 `review.json`에서 각 문항을 검토합니다.

```json
{
  "video_id": "video_1726",
  "question_id": "실제 생성된 question_id 유지",
  "question_video_mapping_verified": true,
  "candidate_evidence_readable": true,
  "required_evidence_groups": [[7, 8, 9], [12, 13]],
  "notes": "예시 인덱스이며 실제 프레임을 확인해서 작성할 것"
}
```

- `question_video_mapping_verified`: 영상과 질문/정답 연결을 확인했으면
  true, 잘못 연결되었다고 확인했으면 false, 미검토는 null.
- `candidate_evidence_readable`: 전체 후보에서 정답에 필요한 근거를
  판독할 수 있으면 true, 없거나 읽을 수 없으면 false, 미검토는 null.
- `required_evidence_groups`: 한 묶음 안은 서로 대체 가능한 프레임입니다.
  위 예는 7/8/9 중 하나와 12/13 중 하나가 **모두** 필요하다는 뜻입니다.
  마지막 글자 한 장만으로 답할 수 있으면 묶음 하나를 사용합니다.
  순서 판단에 두 시점이 필요하면 두 묶음을 사용합니다.
- 판독 여부가 false/null이면 근거 묶음은 `[]`로 둡니다.

모든 문항 항목과 report_sha256은 유지하세요. 검토하지 않은 문항은
null로 남겨도 됩니다. 이 annotation은 사후 진단에만 사용하며
프레임 선택·캡션·정규화에 전달되지 않습니다.

## 3. 근거 포함 여부와 오답 분류 계산

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.qa_diagnostics --results outputs/real_rtx3060/실행폴더/results.json --review outputs/qa_review_d0/review.json --output-dir outputs/qa_review_d0_assessed
```

`diagnosis.json`에 조건별 결과가 생성됩니다.

| status | 해석 |
|---|---|
| human_review_pending | 판독 또는 영상 연결 확인이 아직 없음 |
| check_question_video_mapping | 질문/영상 연결 문제부터 확인 |
| candidate_evidence_not_readable | 후보 자체에 판독 가능한 충분한 근거 없음 |
| selection_misses_required_evidence | 후보에는 근거가 있지만 선택이 일부 필수 묶음을 놓침 |
| inspect_qa_recognition_reasoning_or_scoring | 근거를 모두 선택했으나 오답; QA/채점 점검 대상 |
| covered_and_correct | 표시된 근거를 모두 선택했고 정답 |

이 분류는 사람이 표시한 근거에 의존하며 QA 구현 버그를 자동 확정하지
않습니다. 마지막 오답 유형에서는 글자 인식, 시간 순서 추론, 정답 annotation,
현재 전체 선택지 로그우도 채점 방식을 구분해서 점검해야 합니다.
선택지 점수는 확률이 아니고, 점수 개선만으로 성능 개선을 주장하지 않습니다.

이번 단계에서는 dev/eval을 추가하거나 D2 임계값을 바꾸지 않습니다.
오답 근거 점검이 끝난 뒤 dev 데이터에서 분할 설정을 정하고 고정합니다.
