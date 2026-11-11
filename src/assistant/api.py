"""api.py —— 调用线上预测服务(FastAPI)的最小HTTP客户端(只用标准库)"""
import json
import urllib.request


def http_post(base_url="http://localhost:8000", timeout=10):
    def post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(base_url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    return post
