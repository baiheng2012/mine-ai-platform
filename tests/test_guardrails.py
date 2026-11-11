"""语义护栏测试：模板文本自己必须通过；真实出现过的错误输出必须被拒绝；忠实的改写必须被接受。"""
import pytest

from assistant.guardrails import check_llm_output
from assistant.llm import TemplateBackend

T = TemplateBackend()

GEO = {"shift_id": 2100, "risk_score": 0.0536, "risk_level": "ELEVATED", "alert_threshold": 0.0654,
       "indicators": {"genergy": 5310.0, "gpuls": 287.0, "nbumps": 0.0}}
# 注意：下面浮选facts里的数值是根据真实LLM输出反推的，用于回归测试，不一定与当时完全一致
FLOT = {"hour_index": 3500, "predicted_silica_pct": 1.63, "last_lab_silica_pct": 1.33,
        "expected_change_pct_points": 0.3, "method": "model", "model_gain_vs_baseline_pct": 4.6}
MON = {"flotation": {"retrain_recommended": False, "investigate_drift": True, "windows": 9},
       "geohazard": {"recalibrate_recommended": True, "investigate_drift": True, "mean_alert_rate_pct": 3.1}}
DRAFT_YES = {"shift_id": 2100, "created": True, "order_no": 1, "risk_level": "ALERT", "risk_score": 0.12}
DRAFT_NO = {"shift_id": 2100, "created": False, "risk_level": "NORMAL", "risk_score": 0.01}


@pytest.mark.parametrize("intent,facts", [("geohazard", GEO), ("flotation", FLOT), ("monitoring", MON),
                                          ("draft_order", DRAFT_YES), ("draft_order", DRAFT_NO)])
def test_template_text_passes_all_checks(intent, facts):
    ref = T.explain(intent, facts)
    assert check_llm_output(ref, ref, facts) == (True, "")


def test_real_failure_hallucinated_alert_in_flotation_is_rejected():
    ref = T.explain("flotation", FLOT)
    bad = ("The predicted silica percentage increased by 1.63 percentage points from the last lab test. "
           "The model predicts an expected increase of 0.3 percentage points, which is a 4.6% gain compared "
           "to the baseline. A human inspection is advised as the risk level is ALERT.")
    ok, why = check_llm_output(bad, ref, FLOT)
    assert not ok and "ALERT" in why and "inspe" in why


def test_real_failure_contradictory_reassurance_is_rejected():
    ref = T.explain("geohazard", GEO)
    bad = ("The energy level is at 5310, the pulse rate is 287, and there are no bumps. The risk score is 0.0536, "
           "which is below the alert threshold of 0.0654. This suggests the system is operating within acceptable "
           "parameters, but a human inspection is advised given the elevated risk level.")
    ok, why = check_llm_output(bad, ref, GEO)
    assert not ok and "inspe" in why and "accep" in why


def test_faithful_rewrites_are_accepted():
    good_geo = ("Shift 2100 has an ELEVATED rockburst risk level: the risk score is 0.0536 against an alert threshold "
                "of 0.0654. Keep watching this area. The score ranks shifts and is not a calibrated probability "
                "of a rockburst.")
    assert check_llm_output(good_geo, T.explain("geohazard", GEO), GEO)[0]
    good_flot = ("Hour 3500: the model predicts 1.63% silica in the concentrate, up 0.30 points from the last lab "
                 "value of 1.33%. The model is only about 4.6% better than repeating the last lab value, so treat "
                 "it as a small correction.")
    assert check_llm_output(good_flot, T.explain("flotation", FLOT), FLOT)[0]


def test_dropped_negation_is_rejected():
    ref = T.explain("monitoring", MON)
    bad = ref.replace("not recommended", "recommended")          # 把"不建议"改成"建议"，含义反转
    ok, why = check_llm_output(bad, ref, MON)
    assert not ok and "not recommended" in why
