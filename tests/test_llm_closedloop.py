"""LLM parsing/fallback (no server needed) and closed-loop controller logic."""

from __future__ import annotations

import pytest

from neurohack.closedloop import Controller
from neurohack.llm import LLMClient, clean, fallback_suggestions, parse_suggestions


@pytest.mark.parametrize(
    "text",
    [
        '{"suggestions": ["am fine", "need help", "am tired"]}',
        '```json\n{"suggestions": ["am fine", "need help", "am tired"]}\n```',
        'Sure! ["am fine", "need help", "am tired"]',
        '{"items": ["am fine", "need help", "am tired"]}',
    ],
)
def test_parse_variants(text):
    assert parse_suggestions(text) == ["am fine", "need help", "am tired"]


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_suggestions("I think you should say hello")


def test_clean_strips_echo_and_dupes():
    assert clean(["I am fine", "am fine", "  ", "tired"], "I") == ["am fine", "tired"]


def test_fallback_extends_partial(cfgs):
    fb = cfgs["llm"]["fallback"]
    out = fallback_suggestions("I need", 3, fb)
    assert len(out) == 3 and out[0] in {"help", "water", "the toilet"}
    assert len(fallback_suggestions("", 4, fb)) == 4
    assert len(fallback_suggestions("zzz", 3, fb)) == 3
    assert fallback_suggestions("I am fine", 3, fb) == fb["continuations"][:3]


def test_client_falls_back_when_server_down(cfgs):
    cfg = dict(cfgs["llm"], base_url="http://127.0.0.1:9/v1", timeout_s=0.5)
    s = LLMClient(cfg).suggest("How are you feeling?", "I am", 3)
    assert s.source == "fallback" and len(s.items) == 3 and s.error


def test_controller_hysteresis_and_refractory():
    c = Controller(threshold_on=1.0, threshold_off=0.5, refractory_s=2.0)
    seq = [
        (0.0, 0.2),
        (0.1, 1.2),
        (0.2, 1.3),
        (0.3, 0.8),
        (0.4, 1.2),  # no re-arm above 0.5
        (0.5, 0.4),
        (0.6, 1.2),  # re-armed but refractory (t-0.1 < 2)
        (2.0, 0.4),
        (2.2, 1.5),
    ]  # re-armed and past refractory
    fired = [t for t, v in seq if c.update(t, v)]
    assert fired == [0.1, 2.2]


def test_controller_rejects_bad_hysteresis():
    with pytest.raises(ValueError):
        Controller(0.5, 1.0, 1.0)
