# 최대 구간 수 완화: 분할 확인 후 C/D QA

기본 설정은 **최대 128개 구간, 긴 영상 20개·60문항, 최종 16장**입니다.
128개를 강제하지 않습니다. 기존 D2 변화점 임계값(0.15), 윈도(3),
최소 구간 길이(후보 2장), 프레임 후보, 질문 관련성 배분, 내부 균등 선택,
QA 모델과 프롬프트, 선택지 순서를 유지합니다. 값은 소스 프로토콜에서
가져오고 `max_segments`만 변경합니다. 캡션이나 top-k는 사용하지 않습니다.

새 C는 영상별로 새 D가 실제 만든 구간 수와 같은 개수의 균등 구간을
사용합니다. 여기서 C/D는 새 상한 조건이고 `C_original`, `D_original`은
기존 상한 24 결과입니다. 동일한 16장 입력은 같은 문항/순서에서 재사용합니다.

## 1. 분할과 선택 계획 준비

PowerShell 작업 경로: `C:\Users\dudgh\experimental-code`.

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.segment_cap_dev --phase prepare --max-segments 128 --output-dir outputs/segment_cap128_long
```

이 단계는 영상을 디코딩하고 CLIP을 실행합니다. **QA 모델은 로드하지
않으며 QA/캡션 추론도 하지 않습니다.** 질문 임베딩은 질문 문장만 사용합니다.
원본 후보 프레임 해시와 기존 상한에서의 분할 경계 재현을 확인한 뒤,
새 경계와 문항별 C/D 할당·16장 선택 계획을 저장합니다.

먼저 전달하거나 확인할 파일:

- `outputs/segment_cap128_long/segmentation_summary.json`
- `segmentation_videos.csv`: 영상별 기존/새 구간 수, 상한 도달 여부,
  각 구간의 길이와 후보 수.
- `prepared_selections.csv`: 문항·조건별 0장 구간 수, 다중 프레임 구간 수,
  미배정 후보 비율, 할당 벡터, 최종 인덱스.

요약의 `hit_max_segments_fraction`이 높으면 여전히 상한이 구간 수를
제한할 가능성이 있습니다. 구간 수뿐 아니라 길이 분포와 짧은 구간 비중도
확인하세요. 구간 길이는 후보 인덱스의 반개방 시간 구간 기준이며,
사람이 표시한 의미적 이벤트 길이가 아닙니다. 근거 포착률도 아닙니다.

중단된 준비 작업은 동일 명령에 `--resume`을 추가해 재개합니다.
완료한 영상은 디코딩/CLIP을 다시 실행하지 않습니다.

## 2. 준비된 선택으로 QA 평가

분할 통계를 확인한 후 같은 폴더에서 실행합니다:

```powershell
.\.venv\Scripts\python.exe -B -u -m event_caption_pilot.segment_cap_dev --phase evaluate --max-segments 128 --output-dir outputs/segment_cap128_long
```

준비 단계가 완료되어야 실행됩니다. 저장된 모든 계획의 해시를 검증한 후,
영상만 다시 디코딩하고 준비된 인덱스로 QA를 수행합니다. 분할·질문 임베딩·
프레임 선택은 다시 계산하지 않습니다. 새 QA는 최대 `60 × 2 × 6 = 720`회,
같은 선택이 재사용되면 실제 호출 수는 더 적습니다.

QA는 선택지 순서별로 저장합니다. 중단 시 **evaluate 명령을 그대로
다시 실행**하면 완료한 순서와 문항을 재사용합니다. `--resume`을 추가해도
동일하게 동작합니다. 완료된 영상은 디코딩하지 않습니다.

완료 후 전달할 파일: **`outputs/segment_cap128_long/handoff.json`**.

- `summary.json`, `summary.md`: 새 C/D, 기존 C/D, 쌍별 정확도 차이와 CI.
- `question_metrics.csv`: 문항별 선택지 순서 평균 정확도 및 차이.
- `results.json`: 선택 계획과 선택지별 QA 점수.
- `prepare_progress.json`, `qa_progress.json`: 각 단계 진행 상태.

주 비교는 새 상한에서 `D_minus_C`입니다. 보조 비교는
`D_minus_original_D`, `C_minus_original_C`, 그리고 기존 대비 D−C 차이의
변화인 `change_in_D_minus_C`입니다. 문항별 6개 순서를 평균한 뒤 영상 단위
5,000회 bootstrap합니다. 상한 완화는 경계뿐 아니라 구간 특징·할당까지
바꾸므로, 이를 경계 정확도만의 효과로 해석하지 않습니다.

## 입력 검사와 범위 변경

모델을 로드하거나 출력 폴더를 만들지 않는 입력 검사:

```powershell
.\.venv\Scripts\python.exe -B -m event_caption_pilot.segment_cap_dev --phase prepare --max-segments 128 --output-dir outputs/segment_cap128_long --check-inputs
```

전체 60영상·180문항은 **두 단계 모두** `--duration all`을 붙이고 별도 폴더
`outputs/segment_cap128_all`을 사용합니다. 한 영상 점검에는 `--video-id 856`,
별도 출력 폴더를 사용합니다. 선택한 범위와 설정은 두 단계에서 같아야 합니다.
상한, 모델/소스, 코드 또는 대상 영상이 달라지면 기존 폴더를 재사용하지 않습니다.

이 실행기는 사람이 검증한 근거 구간 주석을 새로 만들지 않습니다. 실제
근거 구간 선택 및 내부 근거 포착 개선은 별도 검토가 필요합니다.
