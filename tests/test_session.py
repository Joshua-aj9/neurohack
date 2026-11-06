"""Session folder + electrode-condition recording."""

from __future__ import annotations

import argparse
import json

import pytest

from neurohack.session import Session, add_electrodes_arg, resolve_electrodes


def test_resolve_electrodes():
    assert resolve_electrodes("dry", live=True) == "dry"
    assert resolve_electrodes(None, live=False) == "n/a"
    assert resolve_electrodes(None, live=True, required=False) == "unknown"
    with pytest.raises(SystemExit):
        resolve_electrodes(None, live=True)


def test_arg_choices():
    ap = argparse.ArgumentParser()
    add_electrodes_arg(ap)
    assert ap.parse_args(["--electrodes", "saline"]).electrodes == "saline"
    with pytest.raises(SystemExit):
        ap.parse_args(["--electrodes", "wet"])


def test_session_records_electrodes(tmp_path):
    s = Session("t", root=tmp_path, label="LIVE", electrodes="dry")
    assert s.dir.name.endswith("_t_dry")
    s.save_results({"x": 1})
    s.close()
    res = json.loads((s.dir / "results.json").read_text(encoding="utf-8"))
    assert res["electrodes"] == "dry" and res["label"] == "LIVE"
    first = json.loads((s.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["electrodes"] == "dry"
    assert (s.dir / "config" / "decoder.yaml").exists()
