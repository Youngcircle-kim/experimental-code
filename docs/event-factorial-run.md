# A/B/C/D 이벤트 실험 실행

## 실험 정의

| 이름 | 구간화 | 프레임 수 배분 | 구간 내부 선택 |
|---|---|---|---|
| A | D0 균등 구간 | 후보 수 비례 | 균등 |
| B | D2 내용 변화 구간 | 후보 수 비례 | 균등 |
| C | D0 균등 구간 | 질문 관련성 softmax | 균등 |
| D | D2 내용 변화 구간 | 질문 관련성 softmax | 균등 |

D2는 기존 `segment()`의 윈도 평균 특징 변화 방식이다. D0는 영상별 D2와 같은 구간 수를 사용한다. 후보 수 비례는 고정 FPS에서 지속시간 비례의 근사이며, 이벤트마다 같은 수를 배정하는 방식과 다르다. 모든 조건에서 capacity를 지키며 정확히 16개 고유 프레임을 시간순으로 QA에 전달한다. K>16이면 일부 구간에 0장이 배정될 수 있다.

**주 비교: D-C.** 보조 비교 B-A, C-A, D-B, D-Uniform, D-BIN과 상호작용 `(D-B)-(C-A)`를 출력한다. D-C는 구간 길이와 평균 특징의 변화까지 포함한 구간화 정책의 비교다.

기존 Qwen 모델·후보 풀·QA·선택지 순서 6개를 그대로 사용한다. caption이나 정답은 선택기에 들어가지 않는다. 정답은 QA 출력 채점에만 사용한다. dev만 실행하며 보류 eval 영상은 추론하지 않는다.

## 1. 실행 전 확인 (GPU 추론 없음)

저장소 루트 `C:\Users\dudgh\experimental-code`에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.event_factorial_dev --baseline-dir outputs/videomme_dev_frozen --selection-dir outputs/videomme_selection_dev_v2 --output-dir outputs/videomme_event_factorial --check-inputs
```

`inputs_validated_no_inference`, 60영상·180문항이 출력되어야 한다. 이 단계는 파일 존재·주석·프로토콜·소스 해시·기존 QA 순서/채점 일관성을 확인한다. 영상 픽셀 해시는 실제 디코딩 때, 실행 라이브러리/모델 메타데이터는 모델 로딩 때 추가 검증한다.

## 2. 전체 dev 실행

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_factorial_dev --baseline-dir outputs/videomme_dev_frozen --selection-dir outputs/videomme_selection_dev_v2 --output-dir outputs/videomme_event_factorial
```

최대 4,320개 새 QA 호출(180문항 × 4조건 × 6순서)이며, 각 QA 호출은 선택지별 로그우도 계산을 포함한다. 선택 인덱스가 같은 조건은 같은 문항·순서·프로토콜 내에서 결과를 재사용한다. Uniform/TOP/BIN 기존 결과도 검증 후 재사용한다. 따라서 실제 새 호출 수는 줄어들 수 있다.

기본 관련성은 mean-pooled event feature와 질문 사이의 raw cosine이다. V/T 점수 교정이나 캡션을 적용하지 않는다. softmax 온도는 기존 config에서 상속하며, 현재 1.0이다. 이는 이전 event pipeline의 z-score 기반 V 조건과 동일한 점수 규칙은 아니다. `protocol.json`에 명시적으로 고정한다.

먼저 한 영상만 실행하려면 **별도 출력 디렉터리**와 `--max-videos 1`을 사용한다. 이 결과는 smoke test이며 60영상 결과로 취급하지 않는다. 전체 실행은 `--max-videos` 없이 새 디렉터리에서 시작한다.

## 3. 중단 후 재개

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.event_factorial_dev --baseline-dir outputs/videomme_dev_frozen --selection-dir outputs/videomme_selection_dev_v2 --output-dir outputs/videomme_event_factorial --resume
```

같은 디렉터리로 두 프로세스를 동시에 실행하지 않는다. 코드·config·주석·원본 결과가 바뀌면 재개를 거부한다. 온도를 명시했거나 `--max-videos`를 사용했다면 재개할 때도 같은 값을 전달한다. 완료 후 재개는 저장된 결과로 재집계하며 이미 계산된 모델 호출을 반복하지 않는다. 후보 픽셀 검증을 위한 영상 디코딩은 반복한다.

## 출력 확인

- `progress.json`: 완료 문항 수와 실행 상태.
- `protocol.json`: 입력·모델·코드 해시, 조건 정의, 온도, 주 비교.
- `plans/*.json`: 영상별 D0/D2 구간, 원래 관련성 점수, 배분 가중치, 정수 배분, 선택 인덱스·시간, 준비 비용. 질문 간 배분 변화 여부와 max_segments 도달 여부도 포함.
- `trials/*.json`: 새로 수행된 선택지 순서별 원자적 체크포인트. 같은 입력의 조건들이 공유한다. QA 실행 시간과 제공되는 processor grid/입력 토큰 정보도 기록한다.
- `questions/*.json`: 완료된 문항의 모든 조건별 결과.
- `results.json`: 전체 결과와 영상별 진단.
- `summary.md`: 정확도, 차이의 95% CI, 길이별 결과, 개선/악화/동일 문항 수.

첫 해석 순서:

1. `D_minus_C`가 주 비교다. CI가 0을 포함하면 이벤트 구간화의 개선 근거 미확인으로 해석한다.
2. `D_minus_B`로 이벤트 내부에서 질문 기반 배분이 도움이 됐는지 확인한다.
3. `allocation_varies_across_questions`가 C/D에서 거짓인 영상이 많은지 확인한다. raw cosine 차이가 작으면 정수 배분이 거의 같아질 수 있다.
4. 평균 특징 중복도, 고정 시간 bin 점유율, 최대 후보 시간 간격을 함께 살핀다. 이 값은 정답 근거 recall이 아니다.

온도를 바꾸려면 `--temperature 0.1`처럼 명시하고 별도 출력 디렉터리를 사용한다. QA 정확도가 가장 높은 온도를 자동 선택하지 않는다. 현재 dev에서 변경한 설정은 이후 보류 평가 전에 고정해야 한다.

문항 안에서 순서별 정답률을 평균한 뒤 영상 단위 paired bootstrap 5,000회를 수행한다. 결과는 탐색적 dev 분석이며 다중 비교 보정은 적용하지 않는다. 시간 기록은 캐시 재사용 여부와 함께 해석한다. 완전한 GPU 메모리/visual-token 효율 벤치마크, 근거 시간 주석 평가는 이번 실행기에 포함되지 않는다.
