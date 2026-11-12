# Model Card: Mine AI Platform (portfolio project)

**Status.** Demonstration project on public datasets. Nothing here has been validated for operational use
and none of it should be used for safety decisions. Numbers come from single runs on small public data sets.

## 1. Geohazard risk ranking (rockburst)

| | |
|---|---|
| Intended use | Rank shifts by relative seismic risk so that a human can decide where to look first. |
| Out of scope | Automatic shutdowns, safety-critical decisions, any mine other than the one in the data. |
| Data | UCI Seismic-bumps (2,584 records, Polish coal mine, about 6.6% positive). Row order assumed to be time order; this is not verified. |
| Model | Logistic regression on standardised features, Platt-calibrated, alert = top 10% of scores (alert budget, an assumption). |
| Evaluation | Time-ordered split; 26 positive events in the 30% hold-out. |

| Hold-out result | Model | Reference |
|---|---|---|
| PR-AUC | 0.105 | 0.034 (random) |
| ROC-AUC | 0.531 | 0.5 |
| Top-10% lift | 0.19 | 0.10 (random) |
| Alerts raised / events caught | 22 alerts, 3 of 26 events | |
| Cost (miss = 10, false alarm = 1) | 249 | 260 (never alert) |

**Key findings.** A random split inflated the random-forest PR-AUC from 0.07 to 0.23 (leakage between neighbouring
records), so the time split is used throughout. More complex models and lag features did not beat the simple model.
The signal is weak and based on 26 events: treat it as a ranking aid, not a detector.
**Monitoring replay.** 7 windows of 100 shifts: 5 flagged DRIFT, 2 WARN; the mean alert rate was 3.1% against a 10% budget,
which triggers a "recalibrate threshold" recommendation.
**Assumptions to confirm with operators.** Cost ratio 10:1, alert budget 10%, risk-level bands.
**Known flaw in the risk bands.** In the 776-shift test segment, 22 shifts are ALERT, 754 are ELEVATED (score of at least half the alert threshold) and none is NORMAL, so the middle band is not informative; it should be redefined (for example by score quantiles).

## 2. Flotation silica forecast (lab-assisted correction)

| | |
|---|---|
| Intended use | Anticipate the next hourly lab value of silica in the concentrate, given the last lab value and process data. |
| Out of scope | Replacing the lab, closed-loop control, "what to change" advice (the model learns correlations, not causes). |
| Data | Kaggle "Quality Prediction in a Mining Process" (CC0), aggregated to 4,091 hourly rows (Mar to Sep 2017). |
| Model | Random forest predicting the change from the last hour's lab value (104 features: sensor means/std, lags, last lab value). |
| Deployment rule | Promotion gate: must beat the persistence baseline (repeat last lab value) by at least 3% RMSE in cross-validation and on hold-out; otherwise the service falls back to persistence and says so. |

| Result | Model | Persistence |
|---|---|---|
| Hold-out RMSE (819 h) | 0.680 | 0.713 |
| RMSE improvement: CV / hold-out | 4.4% / 4.6% | |
| Block-bootstrap 90% interval of improvement | 2.9% to 6.2% | |
| Hold-out MAE | about 4.6% worse | |

**Key findings.** Process data alone has almost no skill (best sensor-only model hold-out R-squared 0.07 versus 0.61 for
repeating the last lab value). The correction helps mainly around changes in the lab value (RMSE about 4.9% better on those hours)
and adds a little noise when nothing changes.
**Data quality.** Two air-flow sensors are frozen for long stretches (99% of March hours repeat the previous value);
about a third of the reference-period hours have feed grades stuck at one constant; feed grades are updated roughly every ten hours.
These artifacts inflate measured drift and probably explain unstable linear models in early folds. They were not removed.
**Monitoring replay.** All 9 weekly windows show input drift, but the model still beat persistence in 8 of 9 (mean improvement about 5%),
so drift triggers investigation, not automatic retraining.

## 3. Neural variant (edge benchmark only)

A small MLP with the same inputs was exported to ONNX (max difference to PyTorch 2.4e-7) and benchmarked on CPU, CUDA and TensorRT FP16.
It improved RMSE by only 2.1% over persistence, below the 3% gate, so **it is not deployed**; the random forest remains the served model.
For models of this size, ONNX Runtime on CPU is fastest at batch size 1 (about 15 microseconds); GPUs pay off for larger models or large batches.

## 4. Alert assistant (rule-routed tools plus local LLM)

| | |
|---|---|
| Design | Rule-based router picks the tool; role-based access control; read-only tools; inspection orders are drafts that need human approval; hash-chained audit log. |
| LLM role | Rewrites a deterministic template text. It never sees the user's message or unauthorised data, and its output is used only if it keeps all numbers and caveats and adds no numbers, risk labels, advice or reassurance. Otherwise the template is shown. |
| Behaviour tests | 19 scripted cases (access control, injection, out-of-scope, errors): 19/19 pass. |
| LLM rewrite acceptance | 21 of 41 requests (51%) with model qwen2.5:3b, all of them verbatim copies of the template (no genuine rewrites); all 20 flotation rewrites were rejected for a stray appended "not" (negation count changed), 2 of them also dropped a number. |

**Observed failure before the semantic guard.** The LLM produced text with correct numbers but wrong meaning
(invented "risk level is ALERT" for a forecast without a risk level; "acceptable parameters" next to "inspection is advised").
These two outputs are kept as regression tests.

## 5. Limitations that apply to everything above

Small public data sets, single plant, short time range, few positive events, no controlled experiments, no external validation,
thresholds and costs are assumptions, and the "agent" is a controlled tool-calling assistant, not an autonomous agent.
