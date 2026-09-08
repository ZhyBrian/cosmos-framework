import copy

import pytest

from examples.umift.history_selection import choose_candidate


def _reports():
    return [{"iteration": step, "history_frames": 5, "protocol_sha256": "frozen",
             "session_equal_aggregate": {"overall": {"lpips": value}}}
            for step, value in zip((500, 1000, 1500, 2000, 2500, 3000), (.3, .2, .1, .1, .2, .3))]


def test_selection_uses_metric_and_earlier_exact_tie():
    assert choose_candidate(_reports(), "frozen", 5)["iteration"] == 1500


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "protocol", "history"])
def test_selection_refuses_incomplete_or_incomparable_candidates(change):
    reports = copy.deepcopy(_reports())
    if change == "missing":
        reports.pop()
    elif change == "duplicate":
        reports[0]["iteration"] = 1000
    elif change == "nan":
        reports[0]["session_equal_aggregate"]["overall"]["lpips"] = float("nan")
    elif change == "protocol":
        reports[0]["protocol_sha256"] = "changed"
    else:
        reports[0]["history_frames"] = 17
    with pytest.raises(ValueError):
        choose_candidate(reports, "frozen", 5)
