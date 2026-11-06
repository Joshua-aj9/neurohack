"""Marker streams with the same name must not cross-talk (G0 regression, Fri 6 Nov)."""

from __future__ import annotations

import time

from neurohack.markers import MarkerInlet, MarkerOutlet, decode


def test_inlet_by_source_id_ignores_same_named_stream():
    other = MarkerOutlet("nh_test_markers", source_id="nh_test_other")
    mine = MarkerOutlet("nh_test_markers", source_id="nh_test_mine")
    assert other.source_id != mine.source_id
    inlet = MarkerInlet(source_id=mine.source_id, timeout=5.0)
    inlet.inlet.open_stream(timeout=5.0)
    time.sleep(0.3)
    other.push("sync", who="other")
    mine.push("sync", who="mine")
    got = []
    t_end = time.time() + 3.0
    while time.time() < t_end and not got:
        got += inlet.read()
        time.sleep(0.05)
    assert got and all(decode(v)["who"] == "mine" for _, v in got)


def test_default_source_ids_are_unique_per_process_and_name():
    a, b = MarkerOutlet("nh_a"), MarkerOutlet("nh_b")
    assert a.source_id != b.source_id and "pid" in a.source_id
