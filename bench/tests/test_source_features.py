from pathlib import Path

import pytest

from agent.source_features import SourceFeatureStore


def test_source_feature_store_reads_transaction_and_identity_by_id(tmp_path: Path) -> None:
    (tmp_path / "transactions.csv").write_text(
        "TransactionID,TransactionAmt,ProductCD\n"
        "1000001,12.5,alpha\n"
        '1000002,20.0,"beta,extra"\n'
    )
    (tmp_path / "identity.csv").write_text(
        "TransactionID,DeviceType,id_01\n1000002,mobile,3.5\n"
    )
    store = SourceFeatureStore(tmp_path)

    assert store.get("T1000001")["ProductCD"] == "alpha"
    assert store.get("1000002")["ProductCD"] == "beta,extra"
    assert store.get("T1000002")["DeviceType"] == "mobile"
    assert store.get("T9000001") is None


def test_source_feature_store_rejects_preprocessed_rows(tmp_path: Path) -> None:
    (tmp_path / "transactions.csv").write_text(
        "TransactionID,card_key\n1000001,C00001-K1\n"
    )
    (tmp_path / "identity.csv").write_text("TransactionID\n")

    with pytest.raises(ValueError, match="requires raw"):
        SourceFeatureStore(tmp_path)
