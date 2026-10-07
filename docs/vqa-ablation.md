# VQA 선택 실험 프로토콜

## 실행

```powershell
python -m event_caption_pilot --config configs/demo.json
python -m event_caption_pilot --config configs/real.rtx3060.json --manifest data/my_vqa_manifest.json
```

데모는 합성 데이터로 파이프라인을 확인합니다. 실제 성능 근거가 아닙니다.
실제 실행은 모델 의존성/가중치와 영상 및 정답이 있는 manifest가 필요합니다.
RTX 3060 프리셋의 4B 모델도 실제 VRAM/추론 실행 검증이 필요합니다.
실행 폴더의 `summary.md`를 먼저 보고, 선택 인덱스·캡션·배분은
`results.json`에서 확인하세요. `pilot=a`는 경계 진단만 실행합니다.

## 조건 정의

| 키 | 선택 방법 |
|---|---|
| uniform | 전체 후보에서 B개 균등 추출, 질문을 사용하지 않음 |
| frame_top | 프레임–질문 cosine 유사도 상위 B개 |
| temporal_bin | B개 인덱스 균등 구간별 최대 유사도 1개 |
| uniform_event_v | 감지 이벤트와 구간 수가 같은 D0, V로 배분 |
| uniform_event_vt | 동일 D0, V+T로 배분 |
| control_v | 설정한 detector의 이벤트, V로 배분 |
| treatment_t | 동일 감지 이벤트, T로 배분; 진단용 |
| treatment_vt | 동일 감지 이벤트, V+T로 배분 |

V는 정규화한 프레임 특징을 구간별 평균한 뒤 다시 정규화한 특징과
질문의 cosine 유사도입니다. T는 캡션과 질문의 텍스트 유사도입니다.
이벤트 조건은 모두 같은 temperature·용량 제한 softmax 배분과
구간 내 균등 추출을 사용합니다. V+T는 dev에서 고정한 z-score를
`(1-text_weight)*V + text_weight*T`로 결합합니다.

모든 최종 선택은 B개 중복 없는 원본 후보 프레임이며 시간순으로 QA에
전달됩니다. 점수 동률에서는 앞선 인덱스를 우선합니다. BIN/D0는 후보
인덱스 기준으로 균등하므로 불규칙 timestamp에서는 동일 시간 길이가 아닙니다.
BIN은 AKS 등의 논문 알고리즘 재현이 아닌 단순 시간 커버리지 대조군입니다.

## 무엇을 분리해서 해석하는가

- H1: TOP/BIN 대 uniform은 질문 기반 선택의 효과를 측정합니다.
- H2: 감지 V 대 D0 V가 경계 위치의 주 비교입니다. 두 조건은 이벤트 수,
  특징 풀링, 배분, 구간 내 추출이 같고 경계 위치가 다릅니다.
  감지 V 대 TOP/BIN은 풀링·배분·추출법도 달라지므로 경계만의 효과로
  해석하지 않습니다. detector=D0이면 감지/D0 조건이 같아집니다.
- H3: 감지 V+T 대 V는 고정 경계에서 캡션을 **선택에 사용하는 효과**입니다.
  D0에서도 같은 비교를 하여 경계 의존성을 살펴봅니다. 캡션을 최종 QA
  프롬프트에 추가하는 효과는 이 실험에 포함되지 않습니다.

최종 QA 프레임 예산은 같지만 캡션 관측·생성 연산은 추가 비용입니다.
두 분할은 이벤트 수와 이벤트당 관측 상한이 같아도 짧은 이벤트 때문에
실제 캡션 입력 프레임 수가 다를 수 있습니다. 각 캡션 provenance를
확인해야 합니다. 전체 suite는 모든 조건의 준비를 공유하므로 baseline의
독립 실행 시간으로 전체 wall time을 보고하지 마세요. 감지 캡션 비용은
`caption_cost`, D0 추가 비용은 `matched_uniform_partition`에 저장됩니다.

## 데이터와 통계

Video-MME/LongVideoBench 등의 영상과 문항을 `examples/manifest.example.json`
형식으로 준비해야 합니다. 공식 데이터 다운로드·스키마 변환·공식 채점기는
포함하지 않습니다. 이 코드는 정답 인덱스가 있는 객관식 VQA를 평가합니다.
자막 입력이나 공식 벤치마크 전체 프로토콜을 재현했다고 주장하지 않습니다.
MP4 입력을 지원하며 NPZ는 미리 추출한 후보 RGB와 timestamp 캐시일 뿐입니다.

dev/eval은 원본 영상 단위로 분리합니다. dev 점수로만 공통 정규화를
맞추고, detector 임계값·text_weight·프레임 예산 등도 dev에서 정한 후
고정하세요. 같은 원본 영상/동일 콘텐츠를 독립 평가 영상으로 중복하지 않습니다.

정답률은 문항 가중이며 차이는 동일 문항의 paired 비교입니다.
신뢰구간은 영상을 묶어서 재표집하는 percentile bootstrap입니다.
결과 JSON의 정확도/차이는 0–1 단위, summary의 정확도는 %, 차이는 pp입니다.
구간은 다중 비교 보정이 없으며 자동으로 가설 채택/기각을 선언하지 않습니다.
근거 annotation이 있는 경우에만 evidence hit를 보조 지표로 제공합니다.
