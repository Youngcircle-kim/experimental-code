# Event caption 최소 파일럿

**Ubuntu + RTX A4500 20GB 두 장에서 실행:**
[A4500 실험 안내](docs/a4500-experiments.md)에 설치, 모델·데이터 준비,
GPU별 16/32프레임 병렬 실행, 재개와 결과 비교 명령을 정리했습니다.
새 실행기는 과거 `outputs/` 없이 원본 영상과 manifest에서 시작합니다.

질문 기반 프레임 선택, 이벤트 분할, 이벤트 캡션의 효과를 **8개 조건의 VQA 비교 실험**으로 검사합니다. D0/D1/D2 경계 진단도 제공합니다. 학습형 detector나 utility predictor를 재현한 코드는 아닙니다.

## 세 가설에 대응하는 비교

| 가설 | 주요 비교 | 추가 대조군 |
|---|---|---|
| 질문에 따른 선택이 균등 선택보다 좋은가? | `frame_top` − `uniform` | `temporal_bin` − `uniform` |
| 이벤트 단위로 묶으면 도움이 되는가? | `control_v` − `uniform_event_v` | `control_v` − TOP/BIN |
| 이벤트 캡션을 선택에 쓰면 좋아지는가? | `treatment_vt` − `control_v` | `uniform_event_vt` − `uniform_event_v`, T 단독 |

`control_v`는 감지 이벤트 V, `treatment_t`는 감지 이벤트 T,
`treatment_vt`는 감지 이벤트 V+T입니다. `uniform_event_*`는 감지한
이벤트와 **구간 수를 맞춘 D0**를 사용합니다. BIN은 후보 인덱스를 B개의
균등한 시간순 구간으로 나누어 각 구간에서 질문 유사도가 가장 높은 프레임을
1개 선택합니다. TOP은 전체 후보에서 상위 B개를 선택합니다.

모든 조건은 같은 후보 풀·QA 모델·최종 프레임 수 B를 사용합니다.
캡션은 프레임 선택에만 사용하며 최종 QA에는 넣지 않습니다.
정규화는 dev 영상의 감지/D0 구간 점수를 함께 사용해 한 번만 계산합니다.
구체적 통제 조건과 해석 범위는 [실험 설계](docs/vqa-ablation.md)를 참고하세요.

**기본 실행은 합성 RGB 영상과 규칙 기반 모델을 쓰는 기능 검증 데모입니다. 실제 영상 성능이나 caption의 효과를 입증하는 실험 결과가 아닙니다.** 실제 실행용 frozen CLIP + Qwen 연결은 Qwen3.8-27B와 Transformers 5.17.0 기준으로 갱신했습니다. RTX 3060 12GB용 Qwen3.5-4B 설정도 별도로 제공합니다. 모델 가중치 추론 검증은 별도로 필요합니다.

## 바로 실행

현재 Windows 작업 환경에서는 아래 인터프리터에 NumPy가 설치되어 있습니다.

```powershell
cd C:\Users\dudgh\experimental-code
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -B run_pilot.py --config configs/demo.json
```

다른 환경에서는 Python 3.10 이상을 사용합니다. 실제 모델 환경은 Python 3.11 또는 3.12의 별도 가상환경을 권장합니다.

```bash
python -m pip install -e ".[test]"
python -m event_caption_pilot --config configs/demo.json
python -W error -m pytest -q
python -m ruff check event_caption_pilot tests run_pilot.py
```

패키지 설치 없이 `pip install -r requirements.txt` 후 프로젝트 루트에서 실행해도 됩니다. 데모에는 NumPy와 표준 라이브러리만 필요합니다. 영상 디코딩과 실제 모델 라이브러리는 해당 모드에서만 가져옵니다.

## 실행 결과

Video-MME dev 60영상/180문항 확장과 고정 QA 실행·재개는
[Video-MME dev 안내](docs/videomme-dev.md)를 참고하세요.

생성 길이 재검사, dev 선택지 순서 진단, 순서 평균을 사용하는 8조건 비교는
[후속 실험 1·2·3](docs/order-experiments.md)을 참고하세요.

기존 실행의 오답을 프레임 단위로 점검하려면
[QA 근거 진단 안내](docs/qa-diagnostics.md)를 사용하세요.
추론을 다시 하지 않고 후보/선택 프레임 보고서와 수동 근거 포함 분석을 만듭니다.

각 실행은 `outputs/<UTC 시각>_<고유 ID>/`를 새로 만듭니다. 기존 실행 결과를 덮어쓰지 않습니다.

| 파일 | 내용 |
|---|---|
| `results.json` | 실행 시각, 전체 Config, 환경·모델 버전, 경계, caption, 문항별 예측, 통계, 비용을 모은 단일 딕셔너리 |
| `config.json` | 기본값까지 포함한 해당 실행의 설정 |
| `questions.csv` | 평가 문항별 8개 조건 정오와 이벤트 V→V+T 개선·악화 |
| `summary.md` | 조건별 정답률, H1/H2/H3별 정확도 차이와 영상 단위 bootstrap 신뢰구간 |
| `diagnostics.html` | 관측 밀도별 경계 타임라인, 경계 전후 원본 프레임, caption |
| `manual_review.json` | 과분할·미분할·sampling 누락·동일 배경 행동 변화 누락, caption 누락·환각·순서 오류의 수동 검토 양식 |

HTML은 인터넷 없이 브라우저에서 열 수 있습니다. 모든 경계는 JSON에 저장하고, HTML의 경계 전후 미리보기 수는 `diagnostic_max_boundaries`로 제한합니다. 수동 검토값 `null`은 **아직 확인하지 않음**을 뜻합니다. 문항별 최종 프레임과 근거 일치는 `results.json`의 `questions[].conditions`에서 확인합니다.

## 비교 설계와 구현

1. 동일한 frozen encoder로 원본 후보 프레임 특징을 계산합니다.
2. **D0:** 관측 인덱스상 균등 분할. **D1:** 인접 특징 간 cosine distance. **D2:** 전후 window 평균 특징 간 cosine distance. 임계값을 넘는 경계를 점수순으로 받아들이되 최소 구간 길이와 최대 구간 수를 지킵니다. 점수 동률은 앞선 위치를 선택합니다.
3. D1 및 D2 각각과 **구간 수가 정확히 같은 D0**를 추가합니다. `observation_strides`는 동일한 후보 풀을 1/2/4칸마다 관측하는 밀도 진단입니다. 이미 빠진 사건을 복원하거나 후보 풀 밖 프레임을 추가하지 않습니다.
4. 파일럿 B는 전체 후보 풀에서 `detector` 하나로 경계를 고정하고, 같은 구간 수의 D0를 추가합니다. 양쪽 분할 모두 이벤트당 최대 `caption_max_frames`개의 중복 없는 시간순 관측으로 caption을 만듭니다. 관측 수는 최종 배분 결과와 무관합니다. D0 추가 캡션과 비용은 `videos[].matched_uniform_partition`에 기록됩니다.
5. V는 시각 특징과 질문, T는 caption과 질문의 cosine relevance를 계산합니다. development 영상의 event-question 쌍에서 채널별 평균·표준편차를 구하고 고정합니다. 정답이나 평가 데이터로 정규화를 학습하지 않습니다.
6. `V+T = (1 - text_weight) * z_visual + text_weight * z_text`. 이벤트 조건 모두 같은 temperature의 softmax, 용량 제한 배분, largest-remainder 정수화를 사용합니다. 구간 내 선택은 같은 균등 샘플러입니다. 프레임은 중복 없이 **정확히 frame_budget개**를 선택하며, 후보가 부족하면 예산을 몰래 줄이지 않고 실패합니다. 이벤트별 배분은 0개일 수 있습니다.
7. 모든 조건의 최종 QA에는 선택된 **원본 RGB 프레임, timestamp, 질문과 선택지**만 전달합니다. Caption이나 정답·근거 annotation은 전달하지 않습니다. 조건 실행 순서는 질문별 시드로 섞되, 각 QA 호출에는 같은 시드를 다시 적용합니다.

이 파일럿의 relevance에는 질문 텍스트만 사용하며, 선택지는 QA 단계에서 사용합니다. Timestamp는 caption 관측과 QA 및 진단에 보존하고, 배분에는 동일한 구간 용량 제약을 적용합니다. 별도의 시간 prior나 학습된 utility curve는 사용하지 않습니다.

### 지표와 비용 해석

- 정답률은 평가 문항에 대해 8개 조건별로 계산합니다. 가설별 비교는 `metrics.hypotheses`에 저장됩니다. 신뢰구간은 다중 비교 보정을 하지 않은 탐색적 구간입니다.
- 쌍별 정답률 차이(`mean_difference`)와 percentile bootstrap 신뢰구간을 제공합니다. **원본 영상 단위로 복원추출**하고 영상의 모든 질문을 함께 포함합니다. 정답률은 질문 수 가중 평균을 유지합니다. 영상이 하나뿐이면 신뢰구간은 `null`입니다. 적은 영상으로 계산한 구간은 불안정할 수 있고, 데모에서 모든 정오가 같아 생기는 폭 0의 구간은 효과 입증이 아닙니다.
- `interval_hit`은 해당 질문의 annotation 구간 중 선택 timestamp가 하나 이상 들어가는 구간의 비율입니다. `all_interval_hit`은 모든 구간에 들어갔는지입니다. 근거 구간은 초 단위 폐구간 `[start, end]`이고, annotation이 없으면 지표는 `null`입니다. 분할 자체의 배열 인덱스 구간은 반개구간 `[start, stop)`입니다.
- Decode, feature, 분할·밀도 진단, caption 생성·cache 접근, caption 특징, development 정규화, relevance, 조건별 selection/QA, export 및 전체 시간을 나눠 기록합니다. CUDA/MPS 모델 호출은 동기화 후 시간을 반환합니다.
- Caption cache hit 시 현재 생성 시간·입력 프레임·토큰 수는 0이며, 최초 생성 시각·생성 시간·원래 관측량은 유지합니다. `historical_cold_generation_seconds`는 저장된 최초 측정값의 합으로, 이번 실행에서 다시 측정한 cold 시간과 다릅니다. 실제 tokenizer 수가 없는 데모는 토큰 수를 추정하지 않고 `null`로 기록합니다.
- 영상별 질문 수 `q`와 `C/q`를 기록합니다. 이는 caption 생성의 상각 비용입니다. 공유 특징 계산이나 모델 warm-up을 포함한 조건별 독립 시스템의 cold/warm 속도 비교는 아니므로 이 수치만으로 속도 우위를 주장하지 않습니다. 내장 백엔드는 로컬 모델이며 외부 API를 호출하지 않습니다.

## 설정과 재현성

모든 실험 설정은 `Config` dataclass에 모이고 `run_experiment(config)`로 전달됩니다. JSON은 필요한 값만 덮어쓸 수 있으며, 명시한 CLI 옵션이 JSON보다 우선합니다. 실제 기본값 전체는 다음 명령으로 내보냅니다.

```bash
python -m event_caption_pilot --write-config my_config.json
python -m event_caption_pilot --config my_config.json --frame-budget 32
python -m event_caption_pilot --pilot a
```

`fix_seed`는 `reproducibility.py` 상단에 정의되고 모듈 초기화 시 호출됩니다. 실행 시 Config 시드로 다시 초기화합니다. Python `random`, NumPy 전역 RNG를 고정하고 별도 `default_rng`에도 시드를 명시합니다. 실제 Transformers 백엔드는 PyTorch 시드, CUDA 시드, cuDNN deterministic/benchmark, deterministic algorithms, cuBLAS 설정을 적용합니다. TensorFlow는 `seed_tensorflow=true`인 경우 지원합니다.

각 caption은 관측 내용·원본 영상·구간·전역 시드에서 만든 별도 생성 시드를 사용하므로 일부 cache hit가 다른 caption의 난수 소비 순서를 바꾸지 않습니다. Cache key는 실제 관측 RGB/timestamp, 구간, 모델 메타데이터, prompt 본문·버전, 생성 설정·시드를 포함하고 **질문·선택지·정답은 포함하지 않습니다**.

`reproducible_result_sha256`는 경계·caption 텍스트·점수·선택·예측·통계의 반복 실행 일치 여부를 확인합니다. 실행 시각·시간·출력 경로·cache 상태는 제외합니다. 원본 후보 RGB/timestamp의 내용 hash도 영상별로 저장합니다.

시드만으로 다른 하드웨어·라이브러리 버전·모델 체크포인트 사이의 bitwise 일치를 보장하지는 않습니다. 실제 실험에서는 결과의 resolved model commit을 Config의 `encoder_revision`과 `vlm_revision`에 고정하고, 사용 환경을 별도로 잠그세요. Python hash seed가 필요하면 **프로세스 시작 전에** `PYTHONHASHSEED`를 설정해야 합니다. 실행 중 환경변수 변경이 현재 인터프리터 hash seed를 바꾼다고 주장하지 않습니다.

```bash
PYTHONHASHSEED=42 python -m event_caption_pilot --config my_config.json
python -m pip freeze > environment.lock.txt
```

공식 동작 참고: [NumPy RNG](https://numpy.org/doc/stable/reference/random/), [PyTorch 재현성](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 실제 영상 연결

실제 데이터와 정답 annotation은 제공되지 않았습니다. `examples/manifest.example.json`의 경로·질문·정답·근거는 **형식 예시**이므로 모두 연구 데이터로 교체해야 합니다. 예시에 해당하는 영상 파일은 포함하지 않습니다.

```bash
python -m pip install -e ".[real,test]"
python -m event_caption_pilot --config configs/real.example.json --pilot a
# 경계·관측 밀도를 검토하고 설정을 고정한 다음:
python -m event_caption_pilot --config configs/real.example.json
```

`real.example.json`은 **Qwen3.8-27B / CUDA / bfloat16** 예시입니다. 27B 언어 모델 가중치만 16비트 기준 약 54GB이므로 RTX 3060 12GB에서 이 설정을 사용하지 마세요. **파일럿 A는 CLIP만 로드하며 Qwen을 로드하지 않습니다.** 기본 JSON의 `main` revision은 가변 참조이므로 체크포인트 commit으로 고정해야 합니다.

RTX 3060 12GB용 후보 설정은 `configs/real.rtx3060.json`입니다. **Qwen3.5-4B**를 명시적으로 선택하며, Qwen3.8의 소형 버전이라는 뜻은 아닙니다. bfloat16, SDPA, 이미지당 최대 50,176 pixels로 메모리를 줄이되 QA 예산 16장과 caption 관측 4장은 유지합니다. 실제 메모리 사용은 질문 길이와 실행 환경에 따라 달라지며, 이 설정의 GPU 실행 성공을 보장하는 측정 결과는 아직 없습니다.

```bash
# 먼저 실제 MP4 경로와 annotation으로 manifest를 준비합니다.
python -m event_caption_pilot --config configs/real.rtx3060.json --pilot a
# 모든 실험 설정을 유지한 채 모델과 chat template을 내려받고 commit을 고정:
python -m event_caption_pilot.prepare_models --config configs/real.rtx3060.json --output-config configs/real.rtx3060.pinned.json
python -m event_caption_pilot --config configs/real.rtx3060.pinned.json
```

모델 준비 명령은 큰 가중치 파일을 다운로드합니다. 실행 환경의 PyTorch CUDA 빌드는 드라이버와 맞아야 하며, `requirements-real-mac.txt`는 Windows용 설치 파일이 아닙니다.

실제 백엔드는 CLIP의 frozen projected visual/text feature를 사용합니다. Qwen은 `AutoModelForMultimodalLM`으로 checkpoint의 아키텍처를 선택합니다. Qwen3.8-27B와 Qwen3.5-4B는 `Qwen3_5ForConditionalGeneration` 계열입니다. 두 작업 모두 `enable_thinking=False`로 채팅 템플릿을 만들고, 시간순 이미지로 greedy caption을 생성합니다. QA에서는 prompt 토큰을 제외한 **선택지 전체 continuation의 조건부 log likelihood**를 점수화합니다. 기본값은 선택지 토큰 수로 나눈 평균이며 `qa_length_normalize`로 고정합니다. Tokenizer가 prompt/선택지 연결 경계를 바꾸면 명시적으로 중단합니다. CLIP context를 넘는 긴 caption/질문은 경고 후 잘립니다. 이미지 크기 제한은 Transformers 5의 `images_kwargs.size`로 매 호출에 적용합니다. 모델 변경 시 새로운 메타데이터로 caption cache가 분리됩니다.

API 참고: [CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Transformers 5.17.0 구현](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py). 오프라인 테스트와 실제 checkpoint/GPU 추론 검증은 구분해야 합니다.

### 입력 계약

**MP4를 직접 입력할 수 있으며 NPZ 사전 추출은 필수가 아닙니다.** 기본 manifest 예제는 dev/eval 모두 MP4입니다. NPZ는 검증된 RGB 후보 풀을 디코딩 없이 반복 재사용하거나, 가변 FPS의 실제 timestamp를 보존할 때 선택하는 포맷입니다. 특징 벡터 파일이 아니며, NPZ를 읽은 뒤에도 CLIP 특징은 다시 계산합니다. MP4 입력도 실행 한 번 안에서는 같은 후보 풀을 8개 조건에서 공유합니다. NPZ를 채택한 원저자의 의도는 문서로 확인되지 않으므로 이는 코드 동작에 근거한 설계 해석입니다.

NPZ를 바꾸지 않고 `candidate_fps`만 바꿔도 후보 밀도는 변하지 않습니다. NPZ 경로에서는 추가 샘플링과 resize를 하지 않기 때문입니다. 경계 진단에서 후보 밀도를 바꾸려면 MP4를 다시 샘플링하거나 NPZ 자체를 다시 준비해야 합니다.

Manifest의 각 `videos` 항목은 `video_id`, `source_id`, `split`, `path`, `questions`를 갖습니다. `source_id`는 **원본 영상**의 ID입니다. 같은 원본의 여러 클립을 독립 영상으로 넣지 말고 하나로 통합하세요. 코드가 source ID 중복과 동일 후보 내용 복제를 검사하며, development/evaluation에 원본을 나누지 않습니다. 비슷한 내용을 편집·재인코딩한 영상의 중복 여부는 데이터 준비 시 별도로 관리해야 합니다.

- 일반 영상: `timestamp_mode: "constant_fps"`를 명시해야 합니다. OpenCV가 전체 영상을 순서대로 읽고 원본 frame index/FPS로 시간을 계산하며 `candidate_fps`에 따라 샘플링합니다. RGB로 바꾸고 Config의 해상도로 조정합니다.
- 가변 FPS 또는 검증된 후보 풀: `.npz`를 사용합니다. 키는 `frames` (`uint8`, `N×H×W×3`, RGB), `timestamps` (`float`, `N`, 엄격 증가·초 단위), `duration_seconds` (스칼라)입니다. `allow_pickle=False`로 읽습니다. NPZ는 이미 준비된 후보 풀로 간주하여 추가 샘플링·resize를 하지 않습니다. 실제 실행에서 동일한 후보 준비 정책을 유지하세요.
- 질문: `question_id`, `text`, `options`, 0부터 시작하는 `answer_index`. `evidence_intervals`는 선택 사항입니다. 같은 영상에서 질문 ID는 유일해야 합니다. 파일럿 A만 실행할 때는 `questions`를 생략할 수 있습니다.
- 경로는 manifest 파일 위치 기준입니다. `manifest_path`, output/cache 경로는 실행 작업 디렉터리 기준입니다.

NPZ를 생성하는 경우 NumPy의 `np.savez_compressed(path, frames=frames, timestamps=timestamps, duration_seconds=duration_seconds)` 형태로 기존 검증된 배열을 저장하면 됩니다. 모든 candidate는 최종 영상 안에 있어야 하며 `duration_seconds`는 마지막 timestamp보다 커야 합니다.

이미 사용 중인 encoder/captioner/QA가 있다면 `types.py`의 `Backend` 계약을 구현하고 `backend: "your_package.your_module:factory"`를 지정할 수 있습니다. Factory는 `Config`를 받아 frozen backend를 반환해야 합니다. 실제 backend 메타데이터에는 `synthetic: false`, 모델·processor 버전과 체크포인트 식별자를 포함하세요. 자체 RNG를 보유한 플러그인은 Config/관측 기반 시드로 명시적으로 초기화해야 하며, torch/tensorflow 플러그인은 해당 `seed_*` 옵션을 설정해야 합니다. Caption 생성은 프레임과 timestamp만, 최종 QA는 프레임·timestamp·질문·선택지만 받아야 합니다.

## 코드 위치

| 모듈 | 역할 |
|---|---|
| `run_pilot.py`, `cli.py` | 실행 진입점·JSON/CLI 설정 |
| `config.py`, `reproducibility.py` | 설정 검증·전역/프레임워크 시드 |
| `types.py`, `data.py` | 모델 입력 계약·영상/NPZ·데이터 품질 검증 |
| `algorithms.py` | D0/D1/D2·정규화·예산 배분·근거/통계 평가 |
| `backends.py`, `hf_backend.py` | 합성 fixture·선택적 실제 frozen 모델 |
| `cache.py` | 질문 독립 caption 캐시·생성 provenance |
| `pipeline.py`, `reporting.py` | 파일럿 실행·메타데이터/JSON/CSV/HTML 내보내기 |
| `tests/` | 경계·예산·누출 방지·영상 단위 재표집·캐시·통합 검증 |
이벤트 분할 데이터 선택과 평가 절차는 [분할 평가 가이드](docs/segmentation-evaluation.md)를 참고하세요. Kinetics-GEBD를 첫 대상으로 제안하며, 경계 정답 변환기와 정량 evaluator는 아직 포함되지 않습니다.
