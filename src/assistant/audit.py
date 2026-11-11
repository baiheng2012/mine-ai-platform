"""
audit.py —— 防篡改的审计日志(JSON Lines + 哈希链)

每条记录包含上一条记录的哈希(prev_hash)和自身哈希(hash)。
任何人修改、删除或插入历史记录，校验时都会在该位置之后断链。
注意：这是"可检测篡改"，不是"防止篡改"；真实系统还需要把日志送到只追加的外部存储。
简化：假设单进程写入，没有处理并发。
"""
import hashlib
import json
import os
import time

GENESIS = "0" * 64


def _digest(rec: dict) -> str:
    # sort_keys保证同一内容得到同一字符串，哈希才稳定
    return hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, path="logs/audit.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.prev = self._last_hash()

    def _last_hash(self):
        """续写已有日志：读取最后一条的哈希作为链头。"""
        if not os.path.exists(self.path):
            return GENESIS
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else GENESIS

    def write(self, **fields):
        rec = {"ts": round(time.time(), 3), **fields, "prev_hash": self.prev}
        rec["hash"] = _digest(rec)                       # 哈希覆盖除hash外的所有字段
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.prev = rec["hash"]
        return rec


def verify_chain(path):
    """校验整条哈希链。返回 (是否完好, 第一个出错的行号或None)。"""
    prev = GENESIS
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev_hash"] != prev or _digest(rec) != h:
                return False, i
            prev = h
    return True, None
