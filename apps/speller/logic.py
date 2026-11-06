"""Pure speller state machine (no pygame): views, letter-group narrowing, undo, metrics.

LLM mode:      suggest view = [s1, s2, s3, MORE];  MORE -> letter view (has BACK).
Baseline mode: letter view only.
Letter view:   item list split into 4 contiguous groups; selecting a group narrows; a single item acts.
               At narrowed levels with < 4 groups, a CANCEL target returns to the top level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurohack.eval.itr import wolpaw_bits

SPACE, UNDO, SEND, BACK, MORE, CANCEL, EMPTY = (
    "<space>",
    "<undo>",
    "<send>",
    "<back>",
    "<more>",
    "<cancel>",
    "",
)
N_TARGETS = 4


def split4(items: list[str]) -> list[list[str]]:
    """Split into up to 4 contiguous, near-equal groups (larger groups first)."""
    n = len(items)
    k = min(N_TARGETS, n)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, i = [], 0
    for s in sizes:
        out.append(items[i : i + s])
        i += s
    return out


@dataclass
class Speller:
    mode: str  # "llm" | "baseline"
    letters: str
    return_after_letter: bool = True
    view: str = ""
    chunks: list[str] = field(default_factory=list)
    group: list[str] | None = None
    suggestions: list[str] = field(default_factory=list)
    sent_text: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("llm", "baseline"):
            raise ValueError(self.mode)
        self.view = "suggest" if self.mode == "llm" else "letters"

    # ---- state -------------------------------------------------------------------------
    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def items(self) -> list[str]:
        base = list(self.letters) + [SPACE, UNDO, SEND]
        return base + [BACK] if self.mode == "llm" else base

    def targets(self) -> list[list[str]]:
        """What each of the 4 targets holds: a list of items (len 1 = direct action)."""
        if self.view == "suggest":
            sug = (self.suggestions + [EMPTY] * 3)[:3]
            return [[s] for s in sug] + [[MORE]]
        groups = split4(self.group if self.group is not None else self.items())
        if self.group is not None and len(groups) < N_TARGETS:
            groups.append([CANCEL])
        while len(groups) < N_TARGETS:
            groups.append([EMPTY])
        return groups

    def needs_suggestions(self) -> bool:
        return self.mode == "llm" and self.view == "suggest"

    def set_suggestions(self, items: list[str]) -> None:
        self.suggestions = list(items)[:3]

    # ---- actions -----------------------------------------------------------------------
    def select(self, idx: int) -> str:
        """Apply a selection of target idx. Returns the action name (for logging)."""
        tgt = self.targets()[idx]
        if len(tgt) > 1:
            self.group = tgt
            return "narrow"
        item = tgt[0]
        self.group = None
        if item == EMPTY:
            return "noop"
        if item == MORE:
            self.view = "letters"
            return "more"
        if item in (BACK, CANCEL):
            if item == BACK:
                self.view = "suggest"
            return item.strip("<>")
        if item == UNDO:
            if self.chunks:
                self.chunks.pop()
            self._after_edit()
            return "undo"
        if item == SEND:
            self.sent_text = self.text.strip()
            return "send"
        if item == SPACE:
            self.chunks.append(" ")
            self._after_edit()
            return "space"
        if self.view == "suggest":
            self.chunks.append(self.suggestion_chunk(item))
            return "suggestion"
        self.chunks.append(item)
        self._after_edit()
        return "letter"

    def _after_edit(self) -> None:
        if self.mode == "llm" and self.return_after_letter:
            self.view = "suggest"

    def suggestion_chunk(self, s: str) -> str:
        """Text to append for suggestion s: completes an unfinished word, else appends a new word."""
        t = self.text
        if t and not t.endswith(" "):
            partial = t.split(" ")[-1]
            if s.lower().startswith(partial.lower()):
                return s[len(partial) :] + " "
            return " " + s + " "
        return s + " "

    def reset(self) -> None:
        self.chunks.clear()
        self.group = None
        self.sent_text = None
        self.view = "suggest" if self.mode == "llm" else "letters"


def label_for(target: list[str], labels: dict) -> str:
    names = {
        SPACE: labels["space"],
        UNDO: labels["undo"],
        SEND: labels["send"],
        BACK: labels["back"],
        MORE: labels["more"],
        CANCEL: labels["cancel"],
        EMPTY: "",
    }
    if len(target) == 1:
        return names.get(target[0], target[0])
    shown = [names.get(x, x.upper()) for x in target]
    if len(shown) <= 3:
        return " ".join(shown)
    return (
        f"{shown[0]}-{shown[-1]}"
        if all(len(x) == 1 for x in target)
        else f"{shown[0]} .. {shown[-1]}"
    )


def sentence_metrics(
    final_text: str,
    t_start: float,
    t_end: float,
    n_commits: int,
    n_undo: int,
    n_abstain: int,
    n_timeout: int,
    n_targets: int = N_TARGETS,
) -> dict:
    """Words/min and Wolpaw ITR for one sentence.

    P (selection accuracy) is estimated as 1 - n_undo / n_commits (each undo implies one wrong selection);
    time includes every gap, abstention, timeout and LLM wait between the first trial and 'send'.
    """
    minutes = max(1e-9, (t_end - t_start) / 60.0)
    words = len(final_text.split())
    p = max(0.0, 1.0 - n_undo / n_commits) if n_commits else 0.0
    bits = wolpaw_bits(n_targets, p)
    return {
        "text": final_text,
        "words": words,
        "minutes": minutes,
        "wpm": words / minutes,
        "n_commits": n_commits,
        "n_undo": n_undo,
        "n_abstain": n_abstain,
        "n_timeout": n_timeout,
        "p_est": p,
        "selections_per_min": n_commits / minutes,
        "itr_bits_min": bits * n_commits / minutes,
    }
