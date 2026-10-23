# A4500 두 GPU에서 SigLIP 2 + D_topk 비교

프레임 선택용 이미지·질문 인코더를
`openai/clip-vit-base-patch32`에서
`google/siglip2-base-patch16-224`로 교체한다. 입력 후보 224×224,
기본 2 FPS, 최종 QA 320×320, Qwen3.5-4B, BF16·SDPA를 유지한다.
기존 pinned 설정에서 인코더만 교체하면 Qwen의 정확한 revision도 보존된다.

이 체크포인트는 고정 해상도 SigLIP 2이며 Transformers의 `SiglipModel`을
사용한다. NaFlex용 `Siglip2Model`과 구분한다. 이미지와 텍스트 모두 같은
SigLIP 공간에서 특징을 추출한다. 질문은 소문자로 정규화하고 최대 64토큰까지
고정 길이로 padding/truncation한다. 질문을 사진 분류용 문장으로 바꾸거나
선택지·정답을 추가하지 않는다. Qwen에 전달하는 질문 원문은 유지한다.

선택 점수는 정규화 특징의 cosine이다. SigLIP의 학습된 logit scale/bias나
sigmoid 확률로 배분 점수 정의까지 바꾸지는 않는다. D2 임계값 0.15와 배분
temperature 1.0을 유지하지만, 이 값이 SigLIP에 최적이라는 의미는 아니다.
새 특징으로 D2 경계·구간 점수·장수 배분을 다시 계산한다.

## 1. 기존 QA 모델 버전을 유지하면서 인코더 준비

Ubuntu 서버의 저장소 루트에서 업데이트된 코드를 사용한다. 모델 가중치 다운로드는
여기서 수행하며, 코드를 교체한 뒤 진행 중인 이전 실험을 같은 폴더에서 재개하지 않는다.
아래 명령은 기존 전체 실험에서 사용한 pinned 설정이 있는 경우다.

```bash
./.venv/bin/python -m event_caption_pilot.prepare_models \
  --config configs/real.a4500.full.pinned.json \
  --encoder-model google/siglip2-base-patch16-224 \
  --output-config configs/real.a4500.siglip2.pinned.json
```

인코더 ID가 바뀌면 기존 CLIP commit을 사용하지 않고 새 모델의 `main`을 조회해
정확한 commit으로 고정한다. Qwen 모델 ID·revision과 나머지 실험 설정은
입력 설정에서 보존한다. 기존 출력 설정 파일이 있으면 덮어쓰지 않는다.
새 모델의 특정 commit을 지정하려면 `--encoder-revision`을 추가한다.
Transformers는 기존 `5.17.0`을 유지한다.

기존 pinned 설정이 없는 새 환경에는 `configs/real.a4500.siglip2.json`을
`--config`로 전달하고 `--encoder-model`을 생략한다. 이 경우 Qwen도 현재
revision으로 새로 고정하므로 과거 실험과 동일한 가중치라고 가정하지 않는다.

## 2. 한 문항으로 실제 GPU 실행 확인

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
PYTHONHASHSEED=42 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./.venv/bin/python -u -m event_caption_pilot.portable_budget \
  --config configs/real.a4500.siglip2.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-dir outputs/a4500_siglip2_smoke \
  --frame-budget 32 --split dev --duration long \
  --max-videos 1 --max-questions 1
```

고정된 후보·프로세서·모델 로딩과 실제 QA를 확인한다. 5방법×6순서로 30 QA다.
실측 GPU 메모리와 속도는 이 smoke의 `handoff.json`과 `backend.json`에서 확인한다.
Base/16을 사용하지만 A4500에서의 peak VRAM은 실행 전 보장하지 않는다.

## 3. SigLIP 기준군을 두 GPU에서 새로 실행

Dev Long 전체 20영상·60문항, 예산32장만 실행할 수 있다.
원본 manifest와 영상은 기존 것을 사용하고 출력 폴더는 새로 지정한다.

```bash
unset CUDA_VISIBLE_DEVICES
./.venv/bin/python -u -m event_caption_pilot.a4500_full_suite run \
  --config configs/real.a4500.siglip2.pinned.json \
  --manifest data/videomme_a4500/manifest.json \
  --output-root outputs/a4500_siglip2_long_baseline \
  --gpu-ids 0 1 --budgets 32 --splits dev --duration long
```

각 영상의 모든 문항과 조건은 같은 GPU에서 수행한다. GPU당 모델 프로세스는 하나이며
두 카드의 VRAM을 합치지 않는다. 기준군은 uniform·frame_top·temporal_bin·C·D의
5조건, 60문항×5조건×6순서 = **1,800 QA**다. 예산이 하나이므로 예산 간 비교는
비어 있고, 방법 간 비교·정확도·신뢰구간·자원 측정은 정상 생성된다.

## 4. 같은 SigLIP 기준군으로 D_topk 후속 비교

기준군 완료 후 실행한다. `--source-root`에 예전 CLIP 출력이 아닌 바로 위의
새 SigLIP 출력 폴더를 지정한다.

```bash
./.venv/bin/python -u -m event_caption_pilot.a4500_long_ablation run \
  --source-root outputs/a4500_siglip2_long_baseline \
  --output-root outputs/a4500_siglip2_long_ablation \
  --gpu-ids 0 1
```

| 조건 | 구간별 배분 | 구간 내부 선택 |
|---|---|---|
| D | SigLIP 구간–질문 관련도 | 균등/중앙; 새 기준군 결과 재사용 |
| D_topk | D와 정확히 같은 배정 | SigLIP 프레임–질문 cosine 상위 k개 |
| B | 구간 후보 개수 비례 | 균등/중앙 |
| B_topk | B와 정확히 같은 배정 | SigLIP 프레임–질문 cosine 상위 k개 |

D2 경계·후보·모델·보기 순서는 새 SigLIP 기준군에서 고정한다. 구간 내부 선택에
보기나 정답을 사용하지 않는다. 추가3조건은 **1,080 QA**, 기준군을 합하면
**2,880 QA**다. smoke의 30 QA는 별도다. 4개 보기 점수를 각각 계산하므로
QA 한 시행에는 4회의 옵션 forward가 포함된다.

완료 후 `outputs/a4500_siglip2_long_ablation/handoff.json`에서
`D_topk_minus_D`를 먼저 확인한다. 영상 단위 짝 bootstrap 5,000회와 QA 시간,
peak CUDA allocated/reserved 메모리도 기록한다. 원자료와 선택 프레임은
`comparison.json`에 있다. 중단 시 같은 명령으로 완료 체크포인트부터 재개한다.

기존 CLIP D와 새 SigLIP D_topk만 비교하면 인코더와 내부 선택이 함께 바뀐다.
내부 선택의 효과는 같은 SigLIP의 D 대 D_topk로 판단한다. 인코더 비교는 같은
방법끼리 비교하되 경계와 점수 분포도 함께 바뀐 실험임을 명시한다. 자동 resume 및
짝 비교는 다른 인코더·revision·코드의 결과를 섞지 않는다.

## 전체 예산 실험

전체 문항을 다시 평가하려면 다른 새 출력 루트에서 `--budgets 16 32 64`,
`--splits dev eval`, `--duration all`을 사용한다. 최소 후보64가 적용된 설정과
전체 원본 영상이 필요하다. 후속 조건을 개발셋에서 선택한 뒤 확인하며, 이미 본
eval 결과를 새로운 독립 검증으로 해석하지 않는다.

## 공식 참고

- [Google SigLIP 2 Base/16 224 모델](https://huggingface.co/google/siglip2-base-patch16-224)
- [Transformers 5.17 SigLIP API](https://huggingface.co/docs/transformers/v5.17.0/model_doc/siglip)
- [SigLIP 2 전처리 안내](https://huggingface.co/docs/transformers/v5.17.0/model_doc/siglip2)
