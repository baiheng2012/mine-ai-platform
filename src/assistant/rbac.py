"""rbac.py —— 基于角色的访问控制(RBAC)：角色 -> 允许使用的工具。默认拒绝。"""

ROLE_TOOLS = {
    # 操作员：只能查看预警和预测
    "operator": {"get_geohazard_risk", "get_flotation_forecast"},
    # 工程师：另外可以查看模型监控状态、起草检查工单(草稿，需人工批准)
    "engineer": {"get_geohazard_risk", "get_flotation_forecast",
                 "get_monitoring_status", "draft_inspection_order"},
}


def is_allowed(role: str, tool: str) -> bool:
    """未知角色返回空集合 -> 没有任何权限(默认拒绝)。"""
    return tool in ROLE_TOOLS.get(role, set())
