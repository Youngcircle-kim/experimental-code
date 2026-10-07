# 최종 답 파싱과 재집계

프로젝트 루트에서 실행합니다. 모델 추론/영상 디코딩 없이 JSON만 읽습니다.
원본은 유지하고 새 폴더에 results.json, summary.md를 저장합니다.

```powershell
.\.venv\Scripts\python.exe -m event_caption_pilot.reaggregate_answers --input outputs/order_dev/results.json --reference-results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --output-dir outputs/order_dev_reparsed_v2
.\.venv\Scripts\python.exe -m event_caption_pilot.reaggregate_answers --input outputs/order_length_256/results.json --reference-results outputs/real_rtx3060/20260917T064754_cb351b63/results.json --output-dir outputs/order_length_reparsed_v2
```

지원 입력은 order_experiments의 dev/length results.json입니다. dev 정답은
기존 manifest에서 읽고 질문/선택지/저장 정오와 대조하므로 원본을 유지하세요.
출력 폴더는 새 이름이어야 합니다.

- 마지막 비어 있지 않은 줄만 검사하며 설명 중간에서 답을 추출하지 않습니다.
- `3. o`, `Answer: 3. o`, `Final answer: 3. o`는 제시된 3번 선택지가
  o일 때만 인정합니다. 번호와 내용이 충돌하면 미판독입니다.
- 답 전체가 선택지 텍스트이거나 마지막 줄이 `Answer: 텍스트`이면 인정합니다.
- 숫자만 있는 답은 값인지 선택지 번호인지 모호하므로 미판독입니다.
- 길이 상한 도달, 중복 선택지, 알 수 없는 표현도 보수적으로 미판독입니다.
- 대소문자/바깥 공백과 마지막 줄의 굵게 표시는 정규화합니다.

generation_parsing에 상태와 마지막 줄, previous_parse에 이전 판독을
보존합니다. 정답은 파서 입력에 전달하지 않습니다. 앞으로 dev/length도
새 파서를 사용합니다. 이전 qa_comparison_probe는 기존 파서를 유지합니다.
판독률과 불일치율을 함께 보고해야 합니다. length의 4건은 잘린 순서만
고른 부분집합이며 전체 6순서 정확도가 아닙니다.
