from agent.gate import decide, load_costs


def test_positive_evsi_beats_robust_action_stop():
    costs = load_costs()
    result = decide(0.52, (0.51, 0.55), 50.0, costs, [], 0)
    assert result.gather is True
    assert result.action in {"customer_validation", "step_up_auth", "analyst_info"}


def test_gate_action_is_recorded_as_a_decision():
    result = decide(0.98, (0.97, 0.99), 1000.0, load_costs(), [], 0)
    assert result.gather is False
    assert result.action


def test_ambiguous_case_requests_step_up_before_decision():
    result = decide(0.38, (0.275, 0.457), 131.30, load_costs(), [], 0)
    assert result.gather is True
    assert result.action == "step_up_auth"
    assert result.evsi_best > 0
