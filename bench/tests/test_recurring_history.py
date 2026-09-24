"""A recurring device must be on this card before the flagged transaction."""

from agent.tools import InvestigationTools
from bench.dataset import DatasetIndex


def test_recurring_device_uses_prior_same_card_history() -> None:
    idx = DatasetIndex()
    card = "C00001-K1"
    flagged = "T9000005"
    prior = [f"T900000{i}" for i in range(1, 5)]
    future = "T9000006"
    idx.card_txns[card] = [*prior, flagged, future]
    idx.txn_dt.update({txn: i for i, txn in enumerate(idx.card_txns[card])})
    idx.txn_device.update({txn: "same device" for txn in [*prior, flagged, future]})
    idx.card_txns["C00002-K1"] = ["T9000007"]
    idx.txn_dt["T9000007"] = 0
    idx.txn_device["T9000007"] = "other card device"

    result = InvestigationTools(idx).q_recurring_entities(card, flagged)
    assert result.data["recurring"] == ["same device"]

    idx.txn_device[prior[-1]] = "one-off"
    assert InvestigationTools(idx).q_recurring_entities(card, flagged).data["recurring"] == []
