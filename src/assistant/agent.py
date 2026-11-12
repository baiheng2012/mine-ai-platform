"""
agent.py —— 受控的工具调用助手（规则路由 + RBAC + 审计 + 护栏）

一次请求的流程：
  清洗输入 -> [注入检查] -> 规则路由(意图+参数) -> [RBAC检查] -> 调用工具(服务API)
  -> 把facts交给后端写说明 -> [数字校验，不通过则回退模板] -> 返回；每一步都写审计日志。
关键安全属性：
  1) 角色由调用方传入，不从消息文字里解析
  2) 用户原话和未授权数据都不会进入LLM
  3) 工具只返回读取结果；工单只生成草稿
"""
import re
import uuid

from assistant.audit import AuditLog
#from assistant.guardrails import looks_like_injection, numbers_supported, sanitize
from assistant.guardrails import check_llm_output, looks_like_injection, sanitize
from assistant.llm import TemplateBackend
from assistant.rbac import is_allowed
from assistant.tools import (Ctx, DataSource, ToolError, draft_inspection_order, get_flotation_forecast,
                             get_geohazard_risk, get_monitoring_status)

INTENT_TOOL = {"geohazard": "get_geohazard_risk", "flotation": "get_flotation_forecast",
               "monitoring": "get_monitoring_status", "draft_order": "draft_inspection_order"}
NEEDS = {"geohazard": "shift", "flotation": "hour", "draft_order": "shift", "monitoring": None}
EXAMPLES = {"shift": "e.g. 'rockburst risk for shift 2100'", "hour": "e.g. 'silica forecast for hour 3500'"}

HELP = ("I can answer questions about rockburst risk for a shift, silica forecasts for an hour, "
        "model monitoring status, and draft inspection orders.")


def route(msg: str):
    """规则路由：按优先级匹配关键词(中英文)，返回意图和参数。不使用LLM，行为可预测、可测试。"""
    m = msg.lower()
    if re.search(r"draft|work order|inspection order|工单|起草", m):
        intent = "draft_order"
    elif re.search(r"monitor|drift|retrain|漂移|监控|重训练", m):
        intent = "monitoring"
    elif re.search(r"silica|flotation|concentrate|forecast|浮选|二氧化硅|精矿", m):
        intent = "flotation"
    elif re.search(r"risk|rockburst|seismic|bump|inspect|岩爆|风险|微震|预警", m):
        intent = "geohazard"
    else:
        return None, {}
    ids = {}
    if (s := re.search(r"(?:shift|班次)\s*#?\s*(\d+)", m)):
        ids["shift"] = int(s.group(1))
    if (h := re.search(r"(?:hour|小时)\s*#?\s*(\d+)", m)):
        ids["hour"] = int(h.group(1))
    return intent, ids


class Assistant:
    def __init__(self, api_post, backend=None, audit=None, data=None, orders_path="orders/drafts.jsonl"):
        self.backend = backend or TemplateBackend()
        self.audit = audit or AuditLog()
        self.ctx = Ctx(api_post=api_post, data=data or DataSource(), orders_path=orders_path)

    #def _result(self, rid, status, answer, **kw):
    #    return {"request_id": rid, "status": status, "answer": answer, "intent": None,
    #            "tool": None, "backend_used": None, "facts": None, **kw}
    def _result(self, rid, status, answer, **kw):
        return {"request_id": rid, "status": status, "answer": answer, "intent": None, "tool": None,
                "backend_used": None, "facts": None, "reject_reason": None, "candidate": None, **kw}
    
    def handle(self, user_id: str, role: str, message: str) -> dict:
        rid = uuid.uuid4().hex[:8]
        msg = sanitize(message)
        log = lambda event, **kw: self.audit.write(event=event, rid=rid, user=user_id, role=role, **kw)
        log("request", message=msg)

        if looks_like_injection(msg):
            log("blocked_input")
            return self._result(rid, "blocked", "This request was blocked by an input safety check. "
                                "Please rephrase it as a plain question.")
        intent, ids = route(msg)
        if intent is None:
            log("refused", reason="out_of_scope")
            return self._result(rid, "refused", HELP)

        tool = INTENT_TOOL[intent]
        if not is_allowed(role, tool):                       # 默认拒绝；工具根本不会被执行
            log("denied", tool=tool)
            return self._result(rid, "denied", f"Your role ({role}) is not permitted to use this function. "
                                "Please ask a user with the engineer role.", intent=intent, tool=tool)
        need = NEEDS[intent]
        if need and need not in ids:
            log("needs_info", tool=tool, missing=need)
            return self._result(rid, "needs_info", f"Please give a {need} number, {EXAMPLES[need]}.",
                                intent=intent, tool=tool)
        try:
            if tool == "get_geohazard_risk":
                res = get_geohazard_risk(self.ctx, ids["shift"])
            elif tool == "get_flotation_forecast":
                res = get_flotation_forecast(self.ctx, ids["hour"])
            elif tool == "get_monitoring_status":
                res = get_monitoring_status(self.ctx)
            else:
                res = draft_inspection_order(self.ctx, ids["shift"], msg, created_by=user_id)
        except ToolError as e:                               # 可预期错误：信息可展示给用户
            log("tool_error", tool=tool, error=str(e))
            return self._result(rid, "error", f"Could not complete the request: {e}.", intent=intent, tool=tool)
        except Exception as e:                               # 非预期错误：不向用户泄露细节
            log("tool_failure", tool=tool, error=type(e).__name__)
            return self._result(rid, "error", "The prediction service is not available right now.",
                                intent=intent, tool=tool)
        log("tool_call", tool=tool, args=ids, meta=res["meta"])
        """
        facts, template = res["facts"], TemplateBackend()
        used, text = template.name, None
        if self.backend.is_llm:
            try:
                cand = self.backend.explain(intent, facts)   # LLM只看到facts，看不到用户原话
                ok, bad = numbers_supported(cand, facts)
                if ok:
                    used, text = self.backend.name, cand
                else:                                        # 出现了无依据的数字：丢弃，改用模板
                    log("llm_output_rejected", unsupported_number=bad)
            except Exception as e:
                log("llm_unavailable", error=type(e).__name__)
        if text is None:
            text = template.explain(intent, facts)
        """
        """
        facts, template = res["facts"], TemplateBackend()
        reference = template.explain(intent, facts)         # 确定性的"标准答案文本"，永远可用
        used, text = template.name, None
        if self.backend.is_llm:
            try:
                cand = self.backend.explain(intent, facts)   # LLM只看到模板文字，看不到用户原话
                ok, reason = check_llm_output(cand, reference, facts)
                if ok:
                    used, text = self.backend.name, cand
                else:                                        # 校验不通过：丢弃，改用模板，并记录原因
                    log("llm_output_rejected", reason=reason)
            except Exception as e:
                log("llm_unavailable", error=type(e).__name__)
        if text is None:
            text = reference

        log("response", backend=used, status="ok")
        return self._result(rid, "ok", text, intent=intent, tool=tool, backend_used=used, facts=facts)
    """

        facts, template = res["facts"], TemplateBackend()
        reference = template.explain(intent, facts)         # 确定性的"标准答案文本"，永远可用
        used, text, reject, candidate = template.name, None, None, None
        if self.backend.is_llm:
            try:
                candidate = self.backend.explain(intent, facts)   # LLM只看到模板文字，看不到用户原话
                ok, reason = check_llm_output(candidate, reference, facts)
                if ok:
                    used, text = self.backend.name, candidate
                else:                                        # 校验不通过：丢弃，改用模板，并记录原因
                    reject = reason
                    log("llm_output_rejected", reason=reason)
            except Exception as e:
                reject = f"llm_unavailable: {type(e).__name__}"
                log("llm_unavailable", error=type(e).__name__)
        if text is None:
            text = reference
        log("response", backend=used, status="ok")
        return self._result(rid, "ok", text, intent=intent, tool=tool, backend_used=used, facts=facts,
                            reject_reason=reject, candidate=candidate if reject else None)
        