# Governance and security notes

Scope: what this demonstration implements, and what a production deployment would still need.

## Controls implemented

| Area | Control | Where |
|---|---|---|
| Model release | Promotion gate: a model must beat a baseline by a set margin in cross-validation and on hold-out, otherwise no deployable model is produced and CI fails. The service falls back to the baseline and labels the response. | `src/train_flotation_final.py`, `src/app.py` |
| Monitoring | Population Stability Index per feature, alert-rate and baseline-gap checks, consecutive-window rules. Drift triggers investigation; retraining is triggered by performance, not by drift alone (avoids alert fatigue). | `src/drift.py`, `src/monitor_*.py` |
| Reproducibility | Locked dependencies, CI that retrains from committed public data, a script that reproduces CI in a clean clone, training and serving share one feature function (tested for skew). | `uv.lock`, `scripts/ci_*.sh`, `tests/` |
| Input validation | Typed request schemas, range checks, consecutive-hour checks, finite-number checks; invalid input returns 422. | `src/app.py` |
| Access control | Role-based, default deny; the role comes from the caller, never from message text; unauthorised tools are never executed. | `src/assistant/rbac.py` |
| Prompt-injection resistance | User text never reaches the LLM; tool outputs are numbers and enumerated values only; free text is stored as data; input pattern screen as a coarse first filter. | `src/assistant/agent.py`, `guardrails.py` |
| Output integrity | LLM output must keep all numbers and caveats and add no numbers, labels, advice or reassurance; otherwise a deterministic template is shown. | `src/assistant/guardrails.py` |
| Human in the loop | Inspection orders are drafts with status "pending human approval"; none are drafted for NORMAL risk. | `src/assistant/tools.py` |
| Auditability | Every request, denial, tool call and LLM rejection is written to a hash-chained log; tampering is detectable. | `src/assistant/audit.py` |
| Data governance | Only public datasets with documented licences; no personal data; derived data documented. | `data/README.md` |
| Container hygiene | Non-root user, no training libraries in the serving image, health checks and resource limits in the Kubernetes manifest. | `Dockerfile`, `k8s/` |

## Known gaps (would be required in production)

- Real authentication and authorisation (SSO, per-user roles); here the role is a command-line argument.
- Audit log in append-only external storage with retention rules; here it is a local file and tamper-evident only.
- A real approval workflow and integration for inspection orders.
- Model registry with approvals and rollback (MLflow Registry or Azure ML), instead of files and image tags.
- Privacy and security review, red-team testing of the assistant, and incident response procedures.
- Validation on the operator's own data, with operators confirming thresholds, costs and alert budgets.
- Monitoring on live data with sensor-health checks (frozen or stuck values were found in the public data).
- Secrets management, network policies and image scanning for the cluster deployment.
