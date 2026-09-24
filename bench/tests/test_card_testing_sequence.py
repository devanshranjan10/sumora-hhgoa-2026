from agent.runner import GraphAgentRunner


def test_card_testing_requires_probe_to_large_transition():
    runner = GraphAgentRunner()
    scores = {
        "card_testing": {"small_auths": 3, "probe_to_large": 0, "score": 0.82},
        "new_device_cnp": {"new_device": False, "cnp": True},
    }
    assert runner._classify(scores) == ("none", "")

    scores["card_testing"]["probe_to_large"] = 1
    assert runner._classify(scores) == ("card_testing", "")
