import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.features import basic_feature_set
from fraud_risk_platform.features.transaction import build_transaction_features


def test_builds_basic_features_without_customer_history():
    transactions = pd.DataFrame(
        {
            "amount": [20.0, 40.0, 60.0, 80.0],
            "tx_datetime": pd.to_datetime(
                [
                    "2018-04-01 00:00:00",
                    "2018-04-01 06:00:00",
                    "2018-04-01 12:00:00",
                    "2018-04-01 18:00:00",
                ]
            ),
        },
        index=[10, 20, 30, 40],
    )

    features = build_transaction_features(transactions, basic_feature_set())

    assert features.columns.tolist() == [
        "amount",
        "hour_sin",
        "hour_cos",
    ]
    assert features.index.equals(transactions.index)

    np.testing.assert_array_equal(
        features["amount"],
        [20.0, 40.0, 60.0, 80.0],
    )
    np.testing.assert_allclose(
        features["hour_sin"],
        [0.0, 1.0, 0.0, -1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features["hour_cos"],
        [1.0, 0.0, -1.0, 0.0],
        atol=1e-12,
    )
