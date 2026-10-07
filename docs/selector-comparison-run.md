# D vs Uniform · Global Top-k · Temporal-bin

긴 영상 20개·60문항에서 **모든 조건을 320×320·16장**으로 비교한다.
QA 모델·가중치·BF16·SDPA·프롬프트·선택지 채점·선택지 순서 6개를 고정한다.
최대 구간 수 128인 D의 완료된 320 결과를 재사용하고, 세 비교군의 QA를 추가한다.

| 결과 이름 | 선택 방식 |
|---|---|
| `uniform` | 전체 후보에서 양 끝을 포함해 균등하게 16장 |
| `frame_top` | 전체 후보 중 질문–CLIP cosine 상위 16장; 간격 제한 없음 |
| `temporal_bin` | 후보를 시간순으로 동일 개수에 가깝게 16구간으로 나누고, 각 구간에서 최고 점수 1장 |
| `D` | 기존 D2 구간 분할·질문 기반 배분·구간 내부 균등 선택 결과 |

모든 조건은 16개의 서로 다른 프레임을 시간순으로 입력한다. 점수 동률이면
더 이른 후보를 선택한다. Temporal-bin의 구간은 후보 개수 기준이며, 이 기법을
AKS 등 외부 논문 전체의 재현으로 간주하지 않는다.

## 실행

PowerShell:

```powershell
Set-Location C:\Users\dudgh\experimental-code
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_dev --output-dir outputs/selector_compare320_long
```

기존 선택 계획이 있으므로 별도의 준비 단계는 없다. **새 QA는 최대 1,080회**다.
`60문항 × 새 조건 3개 × 선택지 순서 6개`이며, QA 한 번 안에서 선택지별 모델
forward를 수행한다. 같은 질문에서 선택 프레임과 320 픽셀이 완전히 같은 조건은
QA를 공유하므로 실제 호출 수는 더 적을 수 있다. D를 다시 추론하지 않는다.

중단되면 **같은 명령을 다시 실행**한다. 선택지 순서마다 원자적으로 저장하므로
완료된 순서부터 재사용한다. 완료된 질문은 디코딩과 QA를 모두 생략한다.
소스 파일, 코드, 설정, 대상 범위가 바뀌면 기존 폴더의 재개를 거부한다.

실행 전 입력 검사만 하려면:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.selector_comparison_dev --check-inputs
```

이 검사는 모델 로딩·영상 디코딩·출력 폴더 생성 없이 수행한다.
첫 영상의 첫 질문만 먼저 실행하려면 별도 출력 폴더를 사용한다:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.selector_comparison_dev --max-videos 1 --max-questions 1 --output-dir outputs/selector_compare320_smoke
```

## 비교가 같은 조건인지 확인하는 방식

- D 소스는 기본 `outputs/resolution320_cap128_long`이다. 여기서 연결된
  `segment_cap128_long` 및 `videomme_event_factorial`의 프로토콜·결과 해시,
  후보 픽셀, 질문 집합, 모델 설정을 확인한다.
- 기존 프레임 선택 실험에 저장된 CLIP 점수로 Uniform·Top-k·Temporal-bin
  규칙을 다시 적용하고, 저장된 선택 인덱스와 정확하게 일치해야 진행한다.
- **선택용 후보 및 CLIP 특징은 기존 224 설정으로 고정**한다. D도 같은
  후보·인코더를 사용했다. QA용 이미지는 모두 원본에서 320으로 추출한다.
- 후보 전체의 224 픽셀·타임스탬프 해시가 기존 실험과 일치해야 한다.
  선택된 원본 이미지만 320으로 보관하므로 후보 전체의 고해상도 배열을 만들지 않는다.
- 저장된 D의 320 입력 픽셀 해시까지 다시 확인한 뒤 D 결과를 재사용한다.
- **기존 비교군의 224 QA 결과는 재사용하지 않는다.** 새 QA와 체크포인트의
  실제 처리 크기, 이미지 격자, 1,600개 시각 토큰을 검사한다.
- 선택기는 질문 문장만 사용하며, 선택지·정답·캡션·사람이 찾은 근거를 추가하지 않는다.

## 결과

전달할 파일:

```text
C:\Users\dudgh\experimental-code\outputs\selector_compare320_long\handoff.json
```

주 비교는 **`D_minus_uniform`**이다. 보조 비교는
`D_minus_frame_top`, `D_minus_temporal_bin`이다.

문항별 선택지 순서 평균을 먼저 계산하고, 영상 단위 5,000회 bootstrap으로
95% 신뢰구간을 계산한다. 선택지 순서를 독립 문항으로 세지 않는다.
보조 비교 CI는 개별 CI이며 다중비교 보정을 적용하지 않는다.
같은 개발 집합에서 D를 조정한 뒤의 비교이므로 최종 주장은 독립 영상에서 재검증한다.

생성 파일:

- `handoff.json`: 네 조건 정확도, 쌍별 차이·CI, 개선/하락 문항 수, QA 자원 요약.
- `results.json`: 선택 인덱스·시각·픽셀 해시·선택지 점수 등 원시 결과.
- `question_metrics.csv`: 문항별 조건 평균과 D 대비 차이.
- `selected_frames.csv`: 조건별 실제 선택된 프레임·시각.
- `selection_plans.json`: 실행 전 고정한 선택 계획과 출처 해시.
- `trials/`, `questions/`, `progress.json`: 재개용 체크포인트와 진행 상태.

완료된 결과만 다시 집계하려면 같은 전체 명령에 `--summarize-only`를 붙인다.

## 비용 지표의 범위

이번 실행기는 정확도 비교를 위해 저장된 선택 결과와 CLIP 점수를 재사용한다.
따라서 기록하는 시간·최대 GPU 메모리는 **QA 추론 비용**이며, 특징 추출과
선택 비용을 포함한 전체 처리 비용은 아니다. 측정하지 않은 선택 비용은 `null`로
표시한다. Uniform은 원리상 CLIP 특징이 필요 없으므로 특징 추출 비용을 0으로
취급한 D와 Uniform의 총비용 비교를 하면 안 된다.

D의 QA 시간은 과거 320 실행 값이다. 새 비교군이 D와 같은 입력이라 D의 QA를
공유한 경우에도 과거 측정으로 표시한다. `timing_origins`에서 구분할 수 있다.
이번 결과만으로 전체 속도나 자원 효율의 우위를 주장하지 않는다.

## 검증

현재 완료된 20영상·60문항의 소스 연결과 세 비교군의 선택 재현을 확인했다.
선택 규칙·동률 처리, D 결과 보존, 동일 입력 공유, 224 결과 혼입 방지,
중단 후 재개 및 원본 픽셀 불일치 검사를 포함한 관련 테스트 16개를 통과했다.
전체 GPU QA 비교는 위 명령으로 실행한다.
