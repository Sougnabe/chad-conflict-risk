"""Ethics guards: no protected characteristic and no outcome column may become a feature."""

import pytest

from chadrisk.experiment import check_feature_sets


@pytest.mark.parametrize("column", ["south_ethnic_share", "religion_majority", "o_actor_fulani",
                                    "main_language", "clan_count", "Tribe"])
def test_protected_characteristics_are_refused(column):
    with pytest.raises(ValueError, match="protected characteristic"):
        check_feature_sets({"B": ["h_fh_lag1", column]})


def test_outcome_columns_are_refused():
    with pytest.raises(ValueError, match="outcome columns"):
        check_feature_sets({"A": ["h_fh_lag1", "y_fh"]})


def test_ordinary_features_are_accepted():
    # words that merely contain a protected term ("factor", "trace") are not refused
    check_feature_sets({"OB": ["o_fh_12m", "h_month_sin", "south_ndvi_crop_anom",
                               "risk_factor", "trace_rain"]})
