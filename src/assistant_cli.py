"""
assistant_cli.py —— 预警助手命令行入口

示例（先启动预测服务：uv run uvicorn app:app --app-dir src --port 8000）：
  uv run python src/assistant_cli.py --role operator "What is the rockburst risk for shift 2100?"
  uv run python src/assistant_cli.py --role engineer --backend ollama     # 交互模式
  uv run python src/assistant_cli.py --verify-audit                       # 校验审计日志哈希链
"""
import argparse

from assistant.agent import Assistant
from assistant.api import http_post
from assistant.audit import AuditLog, verify_chain
from assistant.llm import OllamaBackend, TemplateBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("message", nargs="*", help="question; omit for interactive mode")
    p.add_argument("--role", default="operator", choices=["operator", "engineer"])
    p.add_argument("--user", default="demo_user")
    p.add_argument("--backend", default="template", choices=["template", "ollama"])
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--audit", default="logs/audit.jsonl")
    p.add_argument("--verify-audit", action="store_true")
    a = p.parse_args()

    if a.verify_audit:
        ok, bad = verify_chain(a.audit)
        print("Audit chain OK" if ok else f"Audit chain BROKEN at line {bad}")
        return

    backend = OllamaBackend() if a.backend == "ollama" else TemplateBackend()
    bot = Assistant(http_post(a.api), backend, AuditLog(a.audit))
    print(f"[role={a.role}, backend={backend.name}]")

    def ask(text):
        r = bot.handle(a.user, a.role, text)
        print(f"\n{r['answer']}\n  (status={r['status']}, tool={r['tool']}, text by={r['backend_used']})\n")

    if a.message:
        ask(" ".join(a.message))
    else:
        while (q := input("> ").strip()) not in ("", "exit", "quit"):
            ask(q)


if __name__ == "__main__":
    main()
