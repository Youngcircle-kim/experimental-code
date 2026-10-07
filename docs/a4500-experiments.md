# A4500 20GB × 2: 프레임 선택과 입력 예산 실험

이 안내의 명령은 **Ubuntu 서버의 저장소 루트**에서 실행한다.
새 실행기는 과거 실험의 `outputs/`를 요구하지 않는다. Git으로 받는 코드 외에
실제 MP4 영상과 모델 가중치를 별도로 준비해야 한다.

## 무엇을 확인하는가

학습이나 fine-tuning 없이 frozen CLIP과 Qwen3.5-4B로 긴 영상의 객관식
질의응답을 평가한다. **같은 수의 프레임을 줄 때 어떤 선택 방법이 더 정확한지**,
**16장에서 32장으로 늘리면 그 차이가 어떻게 달라지는지**가 연구 질문이다.
이 실행에서는 캡션·음성·자막을 사용하지 않는다.

| 조건 | 프레임 선택 방법 | 확인하는 점 |
|---|---|---|
| `uniform` | 전체 후보에서 시간순 균등 추출 | 단순 기준선 |
| `frame_top` | 질문–프레임 CLIP 유사도 상위 B장 | 질문 관련성만으로 고르는 효과 |
| `temporal_bin` | 후보를 B개 시간순 구간으로 나눠 각 구간에서 최고 점수 1장 | 시간 범위를 유지하면서 관련성으로 선택하는 효과 |
| `C` | D2와 구간 수가 같은 균등 D0 분할 → 구간 관련성으로 장수 배분 → 내부 균등 | 내용 기반 구간화의 대조군 |
| `D` | CLIP 변화 기반 D2 분할 → 구간 관련성으로 장수 배분 → 내부 균등 | 내용 기반 구간화와 질문 기반 배분을 합친 방법 |

Temporal-bin과 D0는 **후보 개수** 기준이다. D2 구간을 사람이 정의한 사건
경계라고 가정하지 않는다. 모든 방법은 중복 없는 B장을 시간순으로 입력한다.
선택에는 질문 문장만 쓰고 선택지·정답·근거 주석을 넣지 않는다.

- **주 비교:** 각 예산에서 `D_minus_uniform`.
- **보조 비교:** `D_minus_C`, `D_minus_frame_top`, `D_minus_temporal_bin`.
- **예산 비교:** 각 방법의 32−16 정확도 변화와
  `(D32 − Uniform32) − (D16 − Uniform16)`.
- D−C는 구간화 정책의 비교다. 경계뿐 아니라 구간 길이·평균 특징·배분도
  함께 달라지므로 경계 검출 정확도만의 효과라고 해석하지 않는다.

## 두 GPU와 메모리 사용

| 항목 | GPU 0 | GPU 1 |
|---|---|---|
| 독립 프로세스 | B16 실험 | B32 실험 |
| 모델 | CLIP + Qwen3.5-4B | 동일 모델·revision |
| QA 입력 | 320×320, 16장 | 320×320, 32장 |
| 시각 토큰 | 1,600 | 3,200 |
| 정밀도·attention | BF16·SDPA | BF16·SDPA |
| 선택용 후보 | 2 FPS, 224×224, 최대 8,192장 | 동일 |
| 구간 설정 | D2, 최대 128구간, threshold 0.15 | 동일 |
| CLIP batch | 8 | 8 |

**VRAM은 카드마다 20GB씩 사용한다. 이 구성에서 40GB로 합쳐지지 않는다.**
`CUDA_VISIBLE_DEVICES`로 각 프로세스에 카드 하나만 보여 주며, 프로세스 내부
장치 이름은 둘 다 `cuda:0`이다. DDP나 모델 분할은 사용하지 않는다.

Qwen3.5-4B의 공개 체크포인트는 시각 모듈 등을 포함해 약 **9.32GB**의 텐서다.
실행에는 CLIP·activation·attention 작업 공간 등이 더 필요하다. 20GB 카드용
시작 설정이며 **이 문서 작성 과정에서 A4500 실물 추론을 검증하지는 않았다**.
아래 BF16 연산 검사와 smoke를 먼저 실행한다. 한 문항 smoke 성공은 더 긴 질문을
포함한 전체 실행의 최대 메모리를 보장하지 않는다.

후보는 영상 하나씩 CPU에서 처리하고, CLIP 입력만 batch 단위로 GPU로 보낸다.
두 프로세스의 CPU 메모리와 모델 로딩도 고려해 **RAM 64GB 이상을 권장**한다.
이는 측정된 최소 요구량이 아니다. 디스크에는 모델 cache와 영상 공간을 확보한다.
기존 27B 기본 설정은 이 실행기에 사용하지 않는다.

## 1. 코드와 Python 환경

이미 저장소를 받았다면 그 폴더에서 `git pull --ff-only`한다. 처음 받는 경우:

```bash
git clone https://github.com/Youngcircle-kim/experimental-code.git
cd experimental-code
```

Ubuntu 22.04/24.04의 Python 3.10/3.12 기준이다. 기존 서버 `.venv`를
재사용해도 되며, Windows `.venv`를 복사하지 않는다.

아래 OS 패키지는 이미 설치했다면 생략한다. `root` 기준이며 일반 계정에서는
`apt` 앞에 `sudo`를 붙인다. OpenCV import에 필요한 공유 라이브러리도 포함한다.

```bash
apt update
apt install -y python3 python3-venv python3-pip git tmux libgl1 libglib2.0-0
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 공식 CUDA 12.8 wheel의 대응 버전을 먼저 설치
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 프로젝트의 실제 모델·영상·데이터 의존성 (Transformers 5.17.0 포함)
python -m pip install -e '.[real,data]'
python -m pip check

unset CUDA_VISIBLE_DEVICES
python -m event_caption_pilot.a4500_suite check
```

`+cu128`까지 지정하므로 기존 환경에 같은 버전의 CPU wheel이나 다른 CUDA
wheel이 설치되어 있어도 이번 CUDA 12.8 빌드로 맞춘다.

마지막 명령은 두 카드에서 BF16 행렬 곱을 실제 실행하고 모델명·VRAM·여유
메모리를 출력한다. 기본 시작 기준은 카드별 여유 **16GiB 이상**이며 이것은
전체 추론 메모리 보장이 아니다. 다른 작업이 GPU를 쓰고 있다면 먼저 조정한다.
드라이버가 CUDA 13.0을 지원하는 환경에서도 CUDA 12.8 PyTorch를 사용할 수
있다. 이 경로에는 별도 CUDA Toolkit 설치가 필요하지 않다.

## 2. 모델 내려받기와 revision 고정

두 GPU를 동시에 시작하기 **전에 한 번** 실행한다. 공개 모델을 내려받고 실제
commit을 고정한 설정을 만든다. 이미 파일이 있으면 덮어쓰지 않고 중단하므로
재개할 때는 기존 pinned 설정을 그대로 사용한다.

```bash
python -m event_caption_pilot.prepare_models \
  --config configs/real.a4500.json \
  --output-config configs/real.a4500.pinned.json
```

이후 모델 로딩은 내려받은 cache를 사용하는 offline 설정이다.
`configs/real.a4500.pinned.json`과 `.cache/`는 Git에 올리지 않는다.

## 3. 서버 경로로 데이터 준비

기존 `data/videomme/videos/` MP4를 서버에 복사했다면 그대로 사용한다.
**Windows 절대 경로가 들어 있는 manifest는 그대로 사용하지 않는다.**
아래 명령은 공식 annotation에서 서버 경로의 새 manifest를 만들며 영상 파일을
다운로드하지 않는다. 생성 폴더가 있으면 재생성하지 말고 기존 파일을 사용한다.

```bash
python -m event_caption_pilot.prepare_videomme \
  --video-root data/videomme/videos \
  --base-config configs/real.a4500.pinned.json \
  --output-dir data/videomme_a4500 \
  --dev-videos 60 --eval-videos 120 --seed 42
```

전체 표본은 dev 60영상·180문항과 eval 120영상·360문항이다. 이번 기본 실행은
그중 **long 영상만** 사용한다. 실제 대상 수는 `--check-inputs`에서 확인한다.
같은 source ID가 두 split에 겹치면 실행을 거부한다. 이 사용자 정의 표본의
점수는 공식 Video-MME 전체 benchmark 점수가 아니다.

MP4가 없고 ZIP도 없다면 공식 영상 ZIP을 받는다. **20개 ZIP 합계가 약 101GB**로
크다. 이미 받은 ZIP이 있다면 서버 `data/videomme/archives/`로 옮기고 이 다운로드를
생략한다. ZIP을 모두 보존하고 전체 영상을 풀면 모델 제외 200GB 이상이 필요할
수 있으며, 아래 도구는 지정한 표본만 푼다.

```bash
hf download lmms-eval/Video-MME --repo-type dataset \
  --include 'videos_chunked_*.zip' --local-dir data/videomme/archives

# ZIP에서 dev 긴 영상만 CRC/크기를 검증하며 추출; 기존 일치 파일은 재사용
python -m event_caption_pilot.prepare_a4500_videos \
  --manifest data/videomme_a4500/manifest.json \
  --archive-dir data/videomme/archives \
  --video-root data/videomme/videos --split dev --duration long
```

MP4가 이미 준비되어 있으면 ZIP 다운로드와 추출 명령 둘 다 생략한다.
다음 검사는 모델을 로딩하거나 출력 폴더를 만들지 않는다. 대상 영상의 SHA256을
계산하므로 디스크 읽기 시간은 필요하다.

```bash
python -m event_caption_pilot.portable_budget \
  --config configs/real.a4500.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-dir outputs/a4500/dev_b16 \
  --frame-budget 16 --split dev --duration long --check-inputs
```

## 4. 두 GPU smoke와 본 실험

SSH 연결이 끊겨도 계속 실행되도록 tmux에서 시작한다.

```bash
tmux new -s a4500
# 새 tmux 안에서, 저장소 루트 기준
source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES

# dev 첫 영상·첫 문항, 5개 방법, 6개 선택지 순서, 두 예산
python -m event_caption_pilot.a4500_suite smoke

# smoke가 성공한 것을 확인한 뒤 dev 긴 영상 전체
python -m event_caption_pilot.a4500_suite dev

# 두 예산 모두 성공적으로 완료된 뒤 짝지어 비교
python -m event_caption_pilot.compare_budgets \
  --baseline outputs/a4500/dev_b16 \
  --larger outputs/a4500/dev_b32 \
  --output outputs/a4500/dev_budget_comparison.json
```

Ctrl+B 다음 D로 분리하고, `tmux attach -t a4500`으로 복귀한다.
런처는 두 자식 프로세스가 끝날 때까지 기다리며, 한쪽이라도 실패하면 실패를
반환한다. 화면 대신 GPU별 로그에 상세 진행을 쓴다. 다른 SSH 창에서:

```bash
tail -f outputs/a4500/dev_b16.log outputs/a4500/dev_b32.log
# 별도 창에서 GPU 상태 확인
watch -n 2 nvidia-smi
```

GPU를 바꾸려면 시작 전에 `--gpu-ids 1 0`을 쓸 수 있다. 기존 실행 재개에서는
runtime 검증 때문에 GPU 배치와 환경을 그대로 유지한다. 명령만 확인하려면
`python -m event_caption_pilot.a4500_suite dev --dry-run`을 사용한다.

한 예산에서 문항 수가 Q이면 QA 최대 호출 수는 **Q × 5조건 × 6순서**다.
예를 들어 60문항이면 예산당 1,800회, 두 예산 총 3,600회다. 한 QA 안에서
선택지별 forward가 수행되므로 모델 forward 3,600회라는 뜻은 아니다.
이 실행기는 같은 프레임을 고른 조건도 각각 평가한다. 소요 시간은 smoke의
실제 로그로 판단하며 GPU 두 장으로 정확히 두 배 빨라진다고 가정하지 않는다.

## 5. 고정한 설정으로 eval 평가

dev 결과를 확인한 뒤 설정을 확정하고 eval 결과를 보기 전에 고정한다.
이미 과거에 평가에 사용한 영상이면 새 독립 검증이라고 부르지 않는다.
필요한 MP4가 없을 때만 아래 추출을 실행한다.

```bash
python -m event_caption_pilot.prepare_a4500_videos \
  --manifest data/videomme_a4500/manifest.json \
  --archive-dir data/videomme/archives \
  --video-root data/videomme/videos --split eval --duration long

python -m event_caption_pilot.a4500_suite eval

python -m event_caption_pilot.compare_budgets \
  --baseline outputs/a4500/eval_b16 \
  --larger outputs/a4500/eval_b32 \
  --output outputs/a4500/eval_budget_comparison.json
```

eval은 문항/영상 수를 임의로 줄이는 옵션을 거부한다. 예산별 D−Uniform의
개선과 예산 상호작용을 구분해서 보고한다. 16·32 양쪽을 확증적 주 검정으로
다룬다면 다중검정 방침을 미리 정해야 하며, 제공되는 CI는 보정 전 개별 구간이다.

## 재개, 메모리 문제와 결과 읽기

중단된 단계는 **같은 명령을 다시 실행**한다. 예를 들어 dev 재개는
`python -m event_caption_pilot.a4500_suite dev`다. 완료된 문항과 선택지 순서의
체크포인트를 검증해 재사용한다. 코드·설정·입력·runtime이 달라지면 재개를
거부한다. 실행 중 `git pull`이나 패키지 업그레이드를 하지 않는다.

OOM이 나면 로그에서 CLIP 단계인지 QA 단계인지 확인한다. CLIP batch를 8에서
4로 낮춰 재실행할 때도 별도 설정과 새 출력 루트를 사용한다. QA OOM이 계속되는
상태에서 실패 문항만 프레임 수나 해상도를 줄여 기존 결과에 섞지 않는다.
B16만 완주하면 그 다섯 방법의 비교는 가능하지만, B32가 완료되기 전에는 예산
비교를 할 수 없다. 서버 RAM이 부족하면 두 예산을 순서대로 실행할 수도 있다:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u -m event_caption_pilot.portable_budget \
  --config configs/real.a4500.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-dir outputs/a4500_sequential/dev_b16 \
  --frame-budget 16 --split dev --duration long
# 같은 명령의 frame-budget을 32, output-dir을 .../dev_b32로 바꿔 다음 실행
```

| 파일 | 내용 |
|---|---|
| `outputs/a4500/dev_b16/handoff.json` | B16 방법별 정확도와 짝 비교·95% CI, QA 비용 |
| `outputs/a4500/dev_b32/handoff.json` | B32의 동일 지표 |
| `outputs/a4500/dev_budget_comparison.json` | 방법별 32−16과 D의 상대 이득 변화 |
| 각 실행의 `question_metrics.csv` | 문항별 선택지 순서 평균·방법 간 차이 |
| 각 실행의 `results.json`, `protocol.json`, `runtime.json` | 원시 결과, 고정 설정·파일/코드 해시, 실행 환경 |

정확도는 먼저 문항 안에서 선택지 순서 6개의 정오를 평균한다. 이후 문항에 같은
가중치를 주고 **영상 단위 paired bootstrap 5,000회**로 CI를 계산한다.
6개 순서를 독립 문항 6개로 세지 않는다. CI가 0을 포함하면 개선을 확정하기
어렵다는 뜻이며, 두 방법이 동등하다는 증거는 아니다.

저장된 CUDA 최대 메모리는 **PyTorch allocator의 peak allocated/reserved**다.
드라이버 등 프로세스 전체 메모리는 `nvidia-smi`도 함께 확인한다. QA 비용에는
상주 모델이 포함되지만 후보 디코딩·CLIP 선택 비용은 별도다. Uniform에 CLIP
비용이 원래 필요하지 않으므로 QA 시간만으로 전체 처리 효율을 주장하지 않는다.

## 환경 선택의 공식 근거

- [PyTorch CUDA 12.8 대응 버전과 설치 명령](https://pytorch.org/get-started/previous-versions/): torch 2.11.0 / torchvision 0.26.0.
- [NVIDIA A4500 사양](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx/nvidia-rtx-a4500-datasheet.pdf): Ampere, 20GB GDDR6.
- [Ampere BF16 지원](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [NVIDIA CUDA 호환성](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
- [Qwen3.5-4B 가중치 메타데이터](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/model.safetensors.index.json): 텐서 총 9,319,737,856 bytes. 실제 로딩 dtype·임시 메모리에 따라 VRAM 사용은 다르다.
- [Transformers 5.17.0 Qwen3.5 구현](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py): SDPA 지원.
- [Video-MME 공식 배포 파일](https://huggingface.co/datasets/lmms-eval/Video-MME/tree/main), [Hugging Face 다운로드 CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
