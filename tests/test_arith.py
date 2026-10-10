from memory import arith


def test_days_between_is_computed_in_code():
    assert arith.compute({"op": "days_between", "a": "2026-09-10", "b": "2026-09-16"}) == 6
    assert arith.compute({"op": "days_between", "a": "2026-09-16", "b": "2026-09-10"}) == 6
    assert arith.apply("That was {{result}} days later.", {"op": "days_between", "a": "2026-01-01", "b": "2026-03-01"}) == "That was 59 days later."


def test_unusable_specs_are_ignored_safely():
    assert arith.compute({"op": "days_between", "a": "soon", "b": "2026-09-16"}) is None
    assert arith.compute(None) is None
    assert arith.apply("x {{result}} y", {"op": "nope"}) == "x y"


def test_count_dedupes_items():
    assert arith.compute({"op": "count", "items": ["A", "a", "b", ""]}) == 2
