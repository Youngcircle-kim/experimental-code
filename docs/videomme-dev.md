# Video-MME 확장 dev 실험

공식 annotation을 사용한 사용자 정의 표본입니다. 공식 benchmark test에서
개발 표본을 분리하므로 공식 전체 점수와 직접 비교하지 않습니다.
공식 출처: https://huggingface.co/datasets/lmms-lab/Video-MME

## 준비된 표본

`data/videomme_pilot`에 dev 60영상/180문항, holdout(eval) 120영상/360문항을
생성했습니다. 길이(short/medium/long)와 domain을 묶어 번갈아 추출하고,
각 묶음 안에서는 시드로 섞습니다. 파일 존재 여부나 정답률을 보고 표본을
바꾸지 않습니다. task_type은 원본 문항 정보를 보존하고 분포를 보고합니다.
유형별 개수가 균등하다는 보장은 없습니다.

영상은 아직 다운로드하지 않았습니다. preparation.json의 missing_videos에
필요한 파일의 전체 경로가 있습니다. 공식 배포 영상 중 선택된 파일을
`data/videomme/videos/<videoID>.mp4`에 두세요. 별도 YouTube 영상으로 대체하면
내용/시간/편집 차이가 생길 수 있습니다. CFR timestamp 가정도 확인해야 합니다.

annotation만 다시 준비하려면:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe -m event_caption_pilot.prepare_videomme --video-root data/videomme/videos --base-config configs/real.rtx3060.pinned.json --output-dir data/videomme_pilot_new
```

## 고정할 평가 규칙

- Qwen/CLIP: 기존 pinned config 모델 버전, BF16, SDPA 유지.
- 이미지 후보: 2 FPS, 224×224, 최대 8192장. 장시간 영상은 CPU 메모리와
  디코딩 비용이 큽니다. 이 실행기는 영상을 한 개씩 처리합니다.
- 프레임 선택: 전체 후보에서 균등 16장. 이번 단계에는 이벤트/캡션 선택 없음.
- primary QA: 기존 전체 선택지 텍스트 로그우도, 길이 정규화 유지.
- 4지선다: 원래 순서+시드로 고정한 다른 5순서. 모든 문항에 같은 순서 묶음.
- secondary: 옵션으로 256토큰 greedy 생성 및 검증된 최종 답 파싱.
- 자막/오디오 없음. 공식 Video-MME의 문자 답변 생성 프로토콜과 다릅니다.
- 순서 평균은 문항 안에서 계산합니다. 순서 반복은 독립 표본이 아닙니다.

영상 준비 후 생성/채점 진단까지 실행:

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.frozen_dev --config data/videomme_pilot/config.json --with-generation --output-dir outputs/videomme_dev_frozen
```

같은 설정으로 중단 후 재개:

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.frozen_dev --config data/videomme_pilot/config.json --with-generation --output-dir outputs/videomme_dev_frozen --resume
```

protocol.json에 config/manifest/관련 코드 해시를 저장합니다. 재개 시 변경이
있으면 거부하고, 문항별 checkpoint의 후보 해시도 재검사합니다. 완료된
문항은 추론하지 않으며 중간 문항은 다시 실행합니다. eval 영상은 읽거나
추론하지 않습니다. dev 영상 누락은 모델 로딩 전에 오류를 냅니다.

results.json은 길이·주제·질문유형별 결과와 문항별 순서 민감도를 제공합니다.
먼저 데이터/평가 안정성을 확인하고, 이후 별도 eval에서 동일 QA 방식으로
8개 선택 방법을 비교하세요. 현재 기본 run_pilot은 모든 영상을 메모리에
올리므로 이 규모의 8조건 평가에는 아직 별도 메모리 확장이 필요합니다.
