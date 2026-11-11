"""guardrails.py —— 输入清洗、注入模式检查、LLM输出的数字校验"""
import re

MAX_INPUT_LEN = 500
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")          # 控制字符(保留换行和制表符)
_INJECTION = re.compile("|".join([
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"(reveal|show|print).{0,20}system prompt",
    r"you are now",
    r"developer mode",
    r"忽略.{0,6}(之前|以上|前面).{0,6}(指令|说明|规则)",
    r"系统提示",
]), re.IGNORECASE)
# 这只是第一道粗筛，不能当作完整的注入防御；真正的防线是"用户文本根本不进LLM"和RBAC


def sanitize(text) -> str:
    """去掉控制字符、首尾空白，并限制长度。"""
    return _CTRL.sub("", str(text)).strip()[:MAX_INPUT_LEN]


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str):
    """提取文本中的所有数字。先去掉千分位逗号(48,020 -> 48020)，避免被拆成两个数。"""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return [float(t) for t in _NUM.findall(text)]


def flatten_numbers(obj):
    """递归收集结构化结果里的所有数值(布尔值不算)。"""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in flatten_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in flatten_numbers(v)]
    return []


def numbers_supported(text: str, facts: dict):
    """文本里的每个数字，都必须能在facts里找到(允许四舍五入，或换算成百分数)。
    返回 (是否通过, 第一个找不到依据的数字)。"""
    allowed = flatten_numbers(facts)
    for x in numbers_in(text):
        if not any(abs(x - v) <= 0.006 or abs(x - v * 100) <= 0.06 for v in allowed):
            return False, x
    return True, None
# ---------------- 语义护栏：检查LLM改写是否"额外加了东西" ----------------
# 以确定性模板文本(reference)为标准答案，LLM改写后的文本必须满足：
#   1) 风险等级标签、行动/建议类词、安抚类词：模板里没有的不允许出现
#   2) 模板里的数字必须全部保留
#   3) 模板里的否定和限定语(如 "not recommended")必须保留
# 说明：这是基于词表的启发式检查，不能保证语义完全正确，只是把已观察到的失败模式拦下来。
_ACTION = re.compile(r"\b(inspect\w*|evacuat\w*|shut\s?down|stop\w*|halt\w*|urgent\w*|immediate\w*|"
                     r"advis\w*|recommend\w*|should|must|need to|required?)\b", re.IGNORECASE)
_REASSURE = re.compile(r"\b(safe\w*|acceptable|fine|all clear|no (?:risk|danger|concern|issue)s?|"
                       r"nothing to worry|within (?:acceptable|normal|safe))\b", re.IGNORECASE)
_LEVEL_UPPER = re.compile(r"\b(ALERT|ELEVATED|NORMAL)\b")
KEEP_PHRASES = ["not recommended", "not been sent", "not a calibrated probability",
                "small correction", "not flagged"]


def _levels(text: str) -> set:
    """文本里出现的风险等级标签；小写的 elevated 也算(它不会出现在别的语境里)。"""
    found = set(_LEVEL_UPPER.findall(text))
    if re.search(r"\belevated\b", text):
        found.add("ELEVATED")
    return found


#def _stems(pattern, text: str) -> set:
#    """匹配到的词取前5个字母作为词干，使 advise/advised、inspect/inspection 视为同一个词。"""
#    return {m.group(0).lower()[:5] for m in pattern.finditer(text)}

def _stems(pattern, text: str, n=5) -> set:
    """匹配到的词取前n个字母作为词干，使 advise/advised、inspect/inspection 视为同一个词；
    n=None 表示保留完整匹配文本(用于多词短语，如 'within acceptable'，日志里更容易看懂)。"""
    return {m.group(0).lower()[:n] for m in pattern.finditer(text)}

def numbers_missing(output: str, reference: str) -> list:
    """模板里有、但改写文本里丢失的数字(允许四舍五入，或换算成百分数)。"""
    got = numbers_in(output)
    return [x for x in numbers_in(reference)
            if not any(abs(x - g) <= 0.006 or abs(x * 100 - g) <= 0.06 for g in got)]


def claims_check(output: str, reference: str):
    """返回 (是否通过, 问题列表)。"""
    problems = []
    if (extra := _levels(output) - _levels(reference)):
        problems.append(f"unsupported risk level label {sorted(extra)}")
    if (extra := _stems(_ACTION, output) - _stems(_ACTION, reference)):
        problems.append(f"unsupported action or advice wording {sorted(extra)}")
    #if (extra := _stems(_REASSURE, output) - _stems(_REASSURE, reference)):
    if (extra := _stems(_REASSURE, output, None) - _stems(_REASSURE, reference, None)):
        problems.append(f"unsupported reassurance wording {sorted(extra)}")
    
    
    
    for phrase in KEEP_PHRASES:
        if phrase in reference.lower() and phrase not in output.lower():
            problems.append(f"dropped caveat '{phrase}'")
    return (not problems), problems


def check_llm_output(output: str, reference: str, facts: dict):
    """对LLM改写结果做全部检查。返回 (是否采用, 原因字符串)。"""
    problems = []
    if not output.strip():
        problems.append("empty output")
    if len(output) > 2 * len(reference):
        problems.append("output much longer than the reference")
    ok, bad = numbers_supported(output, facts)
    if not ok:
        problems.append(f"unsupported number {bad}")
    if (missing := numbers_missing(output, reference)):
        problems.append(f"dropped numbers {missing}")
    problems += claims_check(output, reference)[1]
    return (not problems), "; ".join(problems)
