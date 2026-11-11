"""
llm.py —— "把结构化结果写成说明文字"的后端。

TemplateBackend：确定性模板，不依赖任何外部服务（CI、评测、LLM失败时的回退）
OllamaBackend  ：本地LLM(通过Ollama的HTTP接口)

注意：LLM只收到 facts(数字和枚举值) 和固定的指令，看不到用户原话。
"""
import json
import os
import urllib.request
"""
SYSTEM_PROMPT = (
    "You are an assistant for mine control-room staff. Write 2 to 4 plain English sentences that explain "
    "the structured FACTS you are given. Rules: use only numbers that appear in FACTS; do not add numbers, "
    "dates, times, equipment names or recommendations that are not in FACTS; do not give safety-critical "
    "instructions; if the risk level is ALERT, say that a human inspection is advised; mention that "
    "forecasts and risk scores are decision support, not certainties."
)
"""
REWRITE_PROMPT = (
    "Rewrite the REFERENCE TEXT below in plain, natural English for mine control-room staff. "
    "Rules: keep every number, every risk-level word (ALERT, ELEVATED, NORMAL) and every caveat or negation "
    "exactly as in the reference; do not add any information, advice, reassurance, explanation or numbers; "
    "never say anything is safe or acceptable; never remove the word 'not' from a sentence. "
    "Output only the rewritten text, at most 3 sentences."
)



class TemplateBackend:
    name = "template"
    is_llm = False

    def explain(self, intent: str, facts: dict) -> str:
        if intent == "geohazard":
            lvl, thr = facts["risk_level"], facts["alert_threshold"]
            head = (f"Shift {facts['shift_id']}: rockburst risk level is {lvl} "
                    f"(risk score {facts['risk_score']:.4f}, alert threshold {thr:.4f}). ")
            tail = {"ALERT": "The score is at or above the alert threshold, so a human inspection is advised. ",
                    "ELEVATED": "The score is above half of the alert threshold, so keep watching this area. ",
                    "NORMAL": "The score is in the normal range. "}[lvl]
            return head + tail + "The score is for ranking shifts, not a calibrated probability of a rockburst."
        if intent == "flotation":
            s = (f"Hour {facts['hour_index']}: predicted silica in the concentrate is "
                 f"{facts['predicted_silica_pct']:.2f}%, compared with the last lab value of "
                 f"{facts['last_lab_silica_pct']:.2f}% (expected change {facts['expected_change_pct_points']:+.2f} points). ")
            if facts["method"] == "baseline_fallback":
                return s + "No validated model is deployed, so this simply repeats the last lab value."
            return s + (f"The model is only about {facts['model_gain_vs_baseline_pct']:.1f}% better than repeating "
                        f"the last lab value, so treat it as a small correction.")
        if intent == "monitoring":
            f, g = facts["flotation"], facts["geohazard"]
            return (f"Flotation model: retraining is {'recommended' if f['retrain_recommended'] else 'not recommended'}; "
                    f"input drift {'is' if f['investigate_drift'] else 'is not'} flagged for investigation "
                    f"({f['windows']} replay windows). "
                    f"Geohazard model: threshold recalibration is {'recommended' if g['recalibrate_recommended'] else 'not recommended'}; "
                    f"the mean alert rate in the replay was {g['mean_alert_rate_pct']:.1f}%, compared with the planned alert budget.")
        if intent == "draft_order":
            if not facts["created"]:
                return (f"No draft was created for shift {facts['shift_id']}: the risk level is NORMAL "
                        f"(score {facts['risk_score']:.4f}); inspection orders are only drafted for ELEVATED or ALERT shifts.")
            return (f"Draft inspection order number {facts['order_no']} was created for shift {facts['shift_id']} "
                    f"(risk level {facts['risk_level']}, score {facts['risk_score']:.4f}). "
                    f"It has not been sent: a human must review and approve it.")
        raise ValueError(f"unknown intent {intent}")

"""
class OllamaBackend:
    name = "ollama"
    is_llm = True

    def __init__(self, model=None, url=None, timeout=90):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")) + "/api/chat"
        self.timeout = timeout

    def explain(self, intent: str, facts: dict) -> str:
        body = {"model": self.model, "stream": False,
                "options": {"temperature": 0, "num_predict": 220},     # 温度0：尽量稳定可复现
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": "FACTS:\n" + json.dumps(facts, indent=1)}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()
"""

class OllamaBackend:
    """本地LLM后端。收到的输入是"确定性模板文本"，任务是改写得更自然(而不是自己组织内容)。
    输出是否被采用，由 guardrails.check_llm_output 决定。"""
    name = "ollama"
    is_llm = True

    def __init__(self, model=None, url=None, timeout=90):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")) + "/api/chat"
        self.timeout = timeout

    def explain(self, intent: str, facts: dict) -> str:
        reference = TemplateBackend().explain(intent, facts)        # 标准答案文本
        body = {"model": self.model, "stream": False,
                "options": {"temperature": 0, "num_predict": 220},  # 温度0：尽量稳定可复现
                "messages": [{"role": "system", "content": REWRITE_PROMPT},
                             {"role": "user", "content": "REFERENCE TEXT:\n" + reference}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()