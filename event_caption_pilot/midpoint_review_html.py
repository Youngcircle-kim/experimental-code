"""Render an offline human evidence review page without model judgments."""

# ruff: noqa: E501
# Keep embedded HTML, CSS, and JavaScript readable without Python line wrapping.

import html
import json
import re


def _escape(value):
    return html.escape(str(value), quote=True)


def _metric(value, *, scale=1, suffix=""):
    if value is None:
        return "없음"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _escape(f"{value * scale:.2f}{suffix}")
    return _escape(value)


def _time(seconds):
    seconds = float(seconds)
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{rest:05.2f}"


def _gallery(question, indices, selected=()):
    cards = []
    for index in indices:
        frame = question["frames"].get(str(index))
        if frame is None:
            continue
        label = f"후보 #{index} · {_time(frame['timestamp'])}"
        marked = index in selected
        cards.append(
            f'<figure class="frame{" chosen" if marked else ""}">'
            f'<a href="{_escape(frame["src"])}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{_escape(frame["src"])}" '
            f'alt="{_escape(label)}"></a><figcaption>{_escape(label)}'
            f"{' · D 선택' if marked else ''}</figcaption></figure>"
        )
    return '<div class="gallery">' + "".join(cards) + "</div>"


def _select(
    field, label, *, default="unknown", disabled=False, allow_na=False
):
    choices = [("unknown", "미판정"), ("yes", "예"), ("no", "아니요")]
    if allow_na:
        choices.append(("na", "해당 없음"))
    options = "".join(
        f'<option value="{value}"{" selected" if value == default else ""}>'
        f"{text}</option>"
        for value, text in choices
    )
    return (
        f'<label>{_escape(label)}<select data-field="{field}"'
        f"{' disabled' if disabled else ''}>{options}</select></label>"
    )


def _question(question, index):
    selected = question["selected_indices"]
    title = f"{index + 1}. {question['video_id']} / {question['question_id']}"
    answer = question["answer_index"]
    gold = (
        f"{answer + 1}. {question['options'][answer]}"
        if isinstance(answer, int) and 0 <= answer < len(question["options"])
        else "없음"
    )
    options = "".join(
        f"<li>{_escape(option)}</li>" for option in question["options"]
    )
    parts = [
        f'<article class="question" data-question="{index}" id="q{index}">',
        f"<h2>{_escape(title)}</h2>",
        f'<p class="question-text">{_escape(question["text"])}</p>',
        f'<ol class="options">{options}</ol>',
        f"<h3>D가 QA 모델에 제공한 전체 {len(selected)}장</h3>",
        '<p class="hint">먼저 이 입력 전체로 질문에 답할 수 있는지 판단하세요. '
        "여러 구간의 장면을 함께 보아야 답할 수 있는 문항도 있습니다.</p>",
        _gallery(question, selected, selected),
        '<div class="overall">',
        _select(
            "selected_input_sufficient",
            "D 전체 입력은 답을 판단하기에 충분한가?",
        ),
        "</div>",
        '<details class="results"><summary>정답·QA 결과 보기 (첫 판정 후 권장)</summary>',
        f"<p>정답: {_escape(gold)}</p>",
        f"<p>{_escape(question.get('comparison', ''))} · "
        f"차이: {_metric(question.get('delta_pp'), suffix='%p')} · "
        f"D 정확도: {_metric(question.get('accuracy_D'), scale=100, suffix='%')} · "
        f"비교 조건 정확도: {_metric(question.get('accuracy_control'), scale=100, suffix='%')}</p>",
        '<p class="hint">정확도는 저장된 선택지 순서별 평가의 평균입니다. '
        "근거 충분성은 사람이 별도로 판정하며 QA 점수로 자동 결정하지 않습니다.</p></details>",
        "<h3>구간별 근거 검토</h3>",
        '<p class="hint">‘중앙 한 장이 충분한가’는 <strong>이 구간에서 필요한 근거</strong>를 '
        "보여주는지에 대한 판정입니다. 이 한 장만으로 전체 질문에 답해야 한다는 뜻은 아닙니다. "
        "질문과 무관한 구간에는 ‘해당 없음’을 선택하세요. 추가 후보는 질문과 관련된 새 근거를 "
        "보여주는지 확인합니다. 0장·여러 장 배정 구간은 중앙 한 장 판정 대상이 아닙니다.</p>",
    ]
    if question["all_candidates_exported"]:
        parts.append(
            '<p class="notice">이 문항의 전체 후보 프레임이 내보내졌습니다. '
            "필요한 구간에서 ‘전체 후보 보기’를 펼칠 수 있습니다. 후보 전체를 검토해도 "
            "후보 추출 사이에 있는 원본 영상의 근거까지 확인한 것은 아닙니다.</p>"
        )
    else:
        parts.append(
            '<p class="notice">이 문항에는 선택 프레임과 일부 미리보기만 있습니다. '
            "미리보기에 근거가 없다는 이유로 해당 구간이나 후보 풀 전체에 근거가 없다고 "
            "판정할 수 없습니다. 더 확인이 필요하면 미판정으로 남기세요.</p>"
        )
    for event_index, event in enumerate(question["events"]):
        is_midpoint = event["is_single_midpoint"]
        count = event["allocation"]
        badge = (
            "0장 배정"
            if count == 0
            else "중앙 1장"
            if is_midpoint
            else f"{count}장 배정"
        )
        previews = [
            i
            for i in event["preview_indices"]
            if i not in event["selected_indices"]
        ]
        parts.extend(
            [
                f'<details class="event" data-event="{event_index}">',
                f"<summary>구간 {_escape(event['event_id'])} "
                f'<span class="badge">{_escape(badge)}</span> '
                f'<span class="range">후보 {event["start"]}–{event["stop"] - 1}'
                f" · {event['stop'] - event['start']}장</span></summary>",
                '<div class="event-body"><h4>D가 선택한 프레임</h4>',
                _gallery(question, event["selected_indices"], selected)
                if count
                else '<p class="hint">이 구간에서는 D 입력에 포함된 프레임이 없습니다.</p>',
                "<h4>추가 후보 미리보기</h4>",
                _gallery(question, previews, selected)
                if previews
                else '<p class="hint">추가 미리보기 프레임이 없습니다.</p>',
            ]
        )
        if question["all_candidates_exported"]:
            parts.extend(
                [
                    '<button type="button" class="toggle-gallery" aria-expanded="false">'
                    "전체 후보 보기</button>",
                    '<div class="full-gallery" hidden></div>',
                ]
            )
        parts.extend(
            [
                '<div class="annotation">',
                _select(
                    "midpoint_sufficient",
                    "중앙 한 장이 이 구간에서 필요한 근거를 충분히 보여주는가?",
                    default="unknown" if is_midpoint else "na",
                    disabled=not is_midpoint,
                    allow_na=True,
                ),
                _select(
                    "extra_frames_add_evidence",
                    "추가 후보에서 이 질문의 근거가 새로 보이는가?",
                ),
                "<label>확인한 근거 프레임의 후보 번호 (쉼표로 구분, 선택·추가 후보 모두 가능)"
                '<span class="hint">추가 근거가 ‘예’이면 D가 선택하지 않은 후보 번호를 최소 1개 입력하세요.</span>'
                '<input data-field="evidence_indices" type="text" inputmode="text" '
                'placeholder="예: 12, 18" autocomplete="off"></label>',
                '<label>검토 메모<textarea data-field="notes" rows="3" '
                'placeholder="필요한 근거, 중앙 장면에서 빠진 내용, 전후 변화·순서, 불확실성 등을 기록하세요.">'
                "</textarea></label></div></div></details>",
            ]
        )
    parts.append("</article>")
    return "".join(parts)


def render_html(report):
    """Return offline HTML; only explicit JSON export persists human annotations."""
    payload = (
        json.dumps(report, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    content = "".join(
        _question(question, i)
        for i, question in enumerate(report["questions"])
    )
    navigation = "".join(
        f'<a href="#q{i}">{i + 1}. {_escape(question["question_id"])}</a>'
        for i, question in enumerate(report["questions"])
    )
    replacements = {
        "__REPORT_TITLE__": _escape(report["report_id"]),
        "__SELECTION_NOTE__": _escape(report["selection_note"]),
        "__NAVIGATION__": navigation,
        "__QUESTIONS__": content,
        "__REPORT_JSON__": payload,
    }
    return re.sub(
        "|".join(replacements), lambda match: replacements[match[0]], _TEMPLATE
    )


_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>중앙 프레임 근거 검토</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Malgun Gothic",sans-serif;color:#1d2939;background:#f1f4f8}
*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:24px}
h1{font-size:28px;margin:0 0 12px}h2{font-size:21px}h3{font-size:18px;margin-top:24px}
h4{font-size:15px;margin:18px 0 10px}p{line-height:1.65}header,.question{background:white;border:1px solid #d9e2ed;border-radius:12px;padding:24px;margin-bottom:24px}
.hint,.range{color:#526171;font-size:14px}.range{margin-left:8px}.notice{background:#fff7df;border-left:4px solid #d6a32e;padding:12px 16px}
.question-text{font-size:20px;font-weight:600;white-space:pre-wrap}.options{line-height:1.7}.options li{white-space:pre-wrap}
nav{display:flex;flex-wrap:wrap;gap:8px}nav a{background:#eef2f6;border-radius:6px;padding:6px 10px;color:#264e86;text-decoration:none;font-size:14px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
.frame{margin:0;border:1px solid #d8e0ea;border-radius:6px;overflow:hidden;background:#f4f6f9}
.frame.chosen{border:2px solid #27746f}.frame img{display:block;width:100%;height:140px;object-fit:contain;background:#152030}
figcaption{padding:7px;font-size:12px;font-variant-numeric:tabular-nums}.event{border:1px solid #d7dfe8;border-radius:8px;margin:12px 0}
summary{cursor:pointer;padding:14px;line-height:1.6}.event>summary{background:#f4f7fa;font-weight:600;border-radius:8px}.event-body{padding:0 16px 16px}
.badge{font-size:12px;color:#21625f;background:#e3f3ef;padding:3px 8px;border-radius:12px;margin-left:10px;white-space:nowrap}
.annotation,.overall{background:#f0f7f5;border-radius:8px;padding:16px;margin-top:16px}.annotation{display:grid;grid-template-columns:1fr 1fr;gap:16px}
label{display:flex;flex-direction:column;gap:7px;font-size:14px;line-height:1.6}.annotation label:last-child{grid-column:1/-1}
input,select,textarea,button{font:inherit}select,input,textarea{border:1px solid #aebbc9;border-radius:5px;padding:9px;background:white;color:#1d2939;width:100%}
select:disabled{background:#e9edf1;color:#536171}.overall select{max-width:280px}textarea{resize:vertical}
button,.file-label{border:1px solid #9fadb9;background:white;border-radius:6px;padding:9px 13px;cursor:pointer;font-size:14px}
button.primary{background:#205f5a;color:white;border-color:#205f5a}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.file-label{display:inline-block}.file-label input{display:none}.results{border:1px dashed #c2cbd5;margin-top:16px;padding:0 12px}.results p{margin:8px 14px}
.toggle-gallery{margin:14px 0}.full-gallery{margin-bottom:20px}#status{min-height:24px;margin:10px 0;color:#265d44}#status.error{color:#a4262b;font-weight:600}
.report-id{font-size:12px;color:#657487;overflow-wrap:anywhere}article{scroll-margin-top:16px}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #226ccc;outline-offset:3px}
@media(max-width:700px){main{padding:12px}header,.question{padding:16px}.annotation{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.frame img{height:120px}.range{display:block;margin-left:0}}
</style></head><body><main>
<header><h1>중앙 프레임이 필요한 근거를 보여주는가?</h1>
<p>선택된 입력을 먼저 보고, 구간 안의 다른 후보와 비교해 사람이 근거 충분성을 기록하는 검토 페이지입니다.
이 페이지는 병목을 자동 판정하지 않습니다. 이미지 클릭 시 원본 크기로 열립니다.</p>
<p class="notice">__SELECTION_NOTE__</p>
<p class="hint">판정은 브라우저에 자동 저장되지 않습니다. 작업을 마치거나 페이지를 닫기 전에
‘판정 JSON 내보내기’를 누르세요. 이어서 작업하려면 같은 보고서에서 이전 JSON을 불러오세요.</p>
<div class="toolbar"><button class="primary" id="export" type="button">판정 JSON 내보내기</button>
<label class="file-label">판정 JSON 불러오기<input id="import" type="file" accept=".json,application/json"></label>
</div><p id="status" role="status" aria-live="polite"></p>
<p class="report-id">보고서 ID: __REPORT_TITLE__</p><nav aria-label="문항 이동">__NAVIGATION__</nav></header>
__QUESTIONS__
</main><script type="application/json" id="report-data">__REPORT_JSON__</script>
<script>
"use strict";
const report = JSON.parse(document.getElementById("report-data").textContent);
let dirty = false;
const status = document.getElementById("status");
const known = ["unknown", "yes", "no"];
function message(text, error=false) {status.textContent=text;status.classList.toggle("error",error);}
function field(root, name) {return root.querySelector('[data-field="'+name+'"]');}
function nodes() {return [...document.querySelectorAll("article.question")];}
function parseEvidence(raw, question, event) {
  if (!raw.trim()) return [];
  const tokens = raw.split(",").map(s=>s.trim());
  if (tokens.some(s=>!/^\\d+$/.test(s))) throw Error("근거 번호는 쉼표로 구분한 0 이상의 정수여야 합니다.");
  const indices=tokens.map(Number);
  for (const index of indices) {
    if (!Number.isSafeInteger(index) || index<event.start || index>=event.stop || !Object.prototype.hasOwnProperty.call(question.frames,String(index)))
      throw Error("근거 번호 "+index+"는 이 구간에 속하며 내보내진 프레임이어야 합니다.");
  }
  return [...new Set(indices)];
}
function requireAdditionalEvidence(indices, event, decision) {
  if (decision==="yes" && !indices.some(index=>!event.selected_indices.includes(index)))
    throw Error("추가 근거를 ‘예’로 판정했다면 D가 선택하지 않은 근거 후보 번호를 최소 1개 입력해야 합니다.");
}
function collect() {
  return {report_id:report.report_id,questions:nodes().map((node, qi)=>{
    const question=report.questions[qi];
    return {video_id:question.video_id,question_id:question.question_id,
      selected_input_sufficient:field(node,"selected_input_sufficient").value,
      events:[...node.querySelectorAll(".event")].map((eventNode, ei)=>{
        const event=question.events[ei];
        let evidence;
        try {
          evidence=parseEvidence(field(eventNode,"evidence_indices").value,question,event);
          requireAdditionalEvidence(evidence,event,field(eventNode,"extra_frames_add_evidence").value);
        }
        catch(error) {
          eventNode.open=true;field(eventNode,"evidence_indices").focus();
          throw Error(question.question_id+" / 구간 "+event.event_id+": "+error.message);
        }
        return {event_id:event.event_id,midpoint_sufficient:field(eventNode,"midpoint_sufficient").value,
          extra_frames_add_evidence:field(eventNode,"extra_frames_add_evidence").value,
          evidence_indices:evidence,notes:field(eventNode,"notes").value};
      })};
  })};
}
document.getElementById("export").addEventListener("click",()=>{
  try {
    const annotations=collect();
    const blob=new Blob([JSON.stringify(annotations,null,2)],{type:"application/json;charset=utf-8"});
    const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;
    link.download="midpoint_annotations_"+report.report_id.replace(/[^a-zA-Z0-9_-]/g,"_")+".json";
    document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    dirty=false;message("판정 JSON 다운로드를 요청했습니다. 파일이 저장되었는지 확인하세요.");
  } catch(error) {message(error.message,true);}
});
function validateImport(data) {
  if (!data || data.report_id!==report.report_id) throw Error("보고서 ID가 일치하지 않습니다. 같은 보고서의 판정 JSON을 선택하세요.");
  if (!Array.isArray(data.questions)) throw Error("문항 목록이 없는 판정 JSON입니다.");
  const updates=[],seenQuestions=new Set();
  for (const annotation of data.questions) {
    const qi=report.questions.findIndex(q=>q.video_id===annotation.video_id && q.question_id===annotation.question_id);
    if (qi<0 || seenQuestions.has(qi)) throw Error("알 수 없거나 중복된 문항입니다.");
    seenQuestions.add(qi);
    if (!known.includes(annotation.selected_input_sufficient) || !Array.isArray(annotation.events)) throw Error("문항 판정 형식이 잘못되었습니다.");
    const question=report.questions[qi],events=[],seenEvents=new Set();
    for (const record of annotation.events) {
      const ei=question.events.findIndex(e=>e.event_id===record.event_id);
      if (ei<0 || seenEvents.has(ei)) throw Error("알 수 없거나 중복된 구간입니다.");
      seenEvents.add(ei);const event=question.events[ei];
      if (![...known,"na"].includes(record.midpoint_sufficient) || (!event.is_single_midpoint && record.midpoint_sufficient!=="na") || !known.includes(record.extra_frames_add_evidence)) throw Error("구간 판정 값이 잘못되었습니다.");
      if (!Array.isArray(record.evidence_indices) || record.evidence_indices.some(i=>!Number.isSafeInteger(i)) || typeof record.notes!=="string") throw Error("근거 번호 또는 메모 형식이 잘못되었습니다.");
      let evidence;
      try {
        evidence=parseEvidence(record.evidence_indices.join(","),question,event);
        requireAdditionalEvidence(evidence,event,record.extra_frames_add_evidence);
      } catch(error) {throw Error(question.question_id+" / 구간 "+event.event_id+": "+error.message);}
      events.push({ei,record,evidence});
    }
    updates.push({qi,annotation,events});
  }
  return updates;
}
document.getElementById("import").addEventListener("change",async(event)=>{
  const file=event.target.files[0];if(!file)return;
  try {
    const updates=validateImport(JSON.parse(await file.text()));
    if (dirty && !window.confirm("현재 내보내지 않은 판정이 있습니다. JSON의 해당 문항·구간 판정으로 덮어쓸까요?")) return;
    const questionNodes=nodes();
    for (const update of updates) {
      const node=questionNodes[update.qi];field(node,"selected_input_sufficient").value=update.annotation.selected_input_sufficient;
      const events=[...node.querySelectorAll(".event")];
      for (const item of update.events) {
        const eventNode=events[item.ei];
        field(eventNode,"midpoint_sufficient").value=item.record.midpoint_sufficient;
        field(eventNode,"extra_frames_add_evidence").value=item.record.extra_frames_add_evidence;
        field(eventNode,"evidence_indices").value=item.evidence.join(", ");field(eventNode,"notes").value=item.record.notes;
      }
    }
    dirty=true;message(updates.length+"개 문항의 판정을 불러왔습니다. 추가 수정 후 JSON을 다시 내보내세요.");
  } catch(error) {message("불러오기 실패: "+error.message,true);}
  finally {event.target.value="";}
});
function timestamp(seconds) {
  const hours=Math.floor(seconds/3600),minutes=Math.floor(seconds/60)%60,rest=(seconds%60).toFixed(2);
  return String(hours).padStart(2,"0")+":"+String(minutes).padStart(2,"0")+":"+rest.padStart(5,"0");
}
for (const [qi,node] of nodes().entries()) {
  const question=report.questions[qi];
  for (const [ei,eventNode] of [...node.querySelectorAll(".event")].entries()) {
    const button=eventNode.querySelector(".toggle-gallery");if(!button)continue;
    button.addEventListener("click",()=>{
      const container=eventNode.querySelector(".full-gallery"),event=question.events[ei];
      if (!container.dataset.loaded) {
        const gallery=document.createElement("div");gallery.className="gallery";
        for(let index=event.start;index<event.stop;index++) {
          const frame=question.frames[String(index)];if(!frame)continue;
          const figure=document.createElement("figure"),chosen=question.selected_indices.includes(index);
          figure.className="frame"+(chosen?" chosen":"");
          const link=document.createElement("a");link.href=frame.src;link.target="_blank";link.rel="noopener";
          const img=document.createElement("img");img.loading="lazy";img.src=frame.src;
          img.alt="후보 #"+index+" · "+timestamp(frame.timestamp);link.append(img);
          const caption=document.createElement("figcaption");caption.textContent=img.alt+(chosen?" · D 선택":"");
          figure.append(link,caption);gallery.append(figure);
        }
        container.append(gallery);container.dataset.loaded="true";
      }
      container.hidden=!container.hidden;button.textContent=container.hidden?"전체 후보 보기":"전체 후보 접기";
      button.setAttribute("aria-expanded",String(!container.hidden));
    });
  }
}
document.addEventListener("input",event=>{if(event.target.matches("[data-field]")){dirty=true;message("판정이 변경되었습니다. JSON 내보내기로 저장하세요.");}});
document.addEventListener("change",event=>{if(event.target.matches("select[data-field]")){dirty=true;message("판정이 변경되었습니다. JSON 내보내기로 저장하세요.");}});
window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
</script></body></html>
"""
