from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "speller"))

from logic import (  # noqa: E402
    CANCEL,
    MORE,
    SEND,
    SPACE,
    UNDO,
    Speller,
    label_for,
    sentence_metrics,
    split4,
)

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def find(sp: Speller, item: str) -> list[int]:
    """Sequence of target indices that selects `item` from the current state (letter view)."""
    path = []
    for _ in range(5):
        for i, t in enumerate(sp.targets()):
            if item in t:
                path.append(i)
                if len(t) == 1:
                    return path
                sp.select(i)
                break
    raise AssertionError(item)


def test_split4_sizes():
    assert [len(g) for g in split4(list(range(29)))] == [8, 7, 7, 7]
    assert [len(g) for g in split4(list(range(2)))] == [1, 1]


def test_baseline_spell_word_and_send():
    sp = Speller("baseline", LETTERS)
    n_sel = 0
    for ch in ["h", "i", SPACE]:
        path = find(sp, ch)
        sp.select(path[-1])
        n_sel += len(path)
    assert sp.text == "hi "
    assert n_sel <= 9  # <= 3 selections per character
    sp.select(find(sp, SEND)[-1])
    assert sp.sent_text == "hi"


def test_cancel_returns_to_top():
    sp = Speller("baseline", LETTERS)
    sp.select(0)
    sp.select(0)  # narrowed to 2 items -> 2 groups + cancel
    tg = sp.targets()
    assert [CANCEL] in tg
    sp.select(tg.index([CANCEL]))
    assert sp.group is None and len(sp.targets()[0]) > 1


def test_llm_mode_flow_and_undo():
    sp = Speller("llm", LETTERS)
    sp.set_suggestions(["I am", "Yes", "No"])
    assert sp.targets()[3] == [MORE]
    assert sp.select(0) == "suggestion" and sp.text == "I am "
    sp.set_suggestions(["fine", "tired", "hungry"])
    sp.select(1)
    assert sp.text == "I am tired "
    sp.select(3)  # more
    assert sp.view == "letters"
    sp.select(find(sp, UNDO)[-1])
    assert sp.text == "I am " and sp.view == "suggest"


def test_suggestion_completes_partial_word():
    sp = Speller("llm", LETTERS)
    sp.chunks = ["I need ", "w"]
    assert sp.suggestion_chunk("water please") == "ater please "
    assert sp.suggestion_chunk("help") == " help "


def test_labels():
    lab = {
        "space": "space",
        "undo": "undo",
        "send": "send",
        "back": "back",
        "more": "more >",
        "cancel": "cancel",
    }
    assert label_for(list("abcdefgh"), lab) == "A-H"
    assert label_for(["y", "z", SPACE], lab) == "Y Z space"
    assert label_for([UNDO], lab) == "undo"


def test_metrics():
    m = sentence_metrics("i am fine", 0.0, 60.0, n_commits=10, n_undo=1, n_abstain=2, n_timeout=0)
    assert m["wpm"] == pytest.approx(3.0)
    assert m["p_est"] == pytest.approx(0.9)
    assert m["itr_bits_min"] > 0
