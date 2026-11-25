"""Text rules: one table per dataset, applied to the task texts (``tasks.parquet``, at build)
and to the subtask texts (the lookup, at curate, before it is written).

- ``galaxea``: 3 mixed-language strings mapped (one also without
  its final period: the trained tasks hold its mapped text); otherwise date / digit suffixes
  stripped, spacing fixed, first letter capitalized, final period added. The same rule on
  tasks and subtasks (13,706 → 13,693 distinct tasks).
- ``agibot_world_2026``: tasks: 3 exact strings (4 episodes); subtasks: 5 past-tense verbs to
  present (2,080 episodes). Together they reproduce the trained texts.
- RoboMIND: tasks only (the subtask lookups are published as
  trained): phrasing overrides, typo fixes, hand/article restructuring, final period.
  ``robomind_annotation`` cleans the per-episode annotations the one-word AgileX tasks take.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


def _identity(text: str) -> str:
    return text


@dataclass(frozen=True)
class TextRules:
    task: Callable[[str], str] = _identity
    subtask: Callable[[str], str] = _identity


# ------------------------------------------------------------------ galaxea

GALAXEA_OVERRIDES = {
    "俯视到窗帘前": "Move forward and look down to the front of the curtain.",
    "俯视到窗帘前.": "Move forward and look down to the front of the curtain.",
    "俯视到窗帘前 Move forward and look down to the front of the curtain.": (
        "Move forward and look down to the front of the curtain."
    ),
    "Arrange the black cup on the洗漱 box neatly with your left hand.": (
        "Arrange the black cup on the toiletry box neatly with your left hand."
    ),
}
_DATE_SUFFIX = re.compile(r"_?20[0-9]{6}$")
_DIGIT_SUFFIX = re.compile(r"[0-9]{6,}$")


def galaxea_text(text: str) -> str:
    """One Galaxea task or subtask text, cleaned."""
    override = GALAXEA_OVERRIDES.get(text.strip())
    if override:
        return override
    out = _DIGIT_SUFFIX.sub("", _DATE_SUFFIX.sub("", text).strip()).strip()
    if not out:
        return out
    out = out.replace("，", ",")
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",(?!\s)", ", ", out)
    out = out[0].upper() + out[1:]
    if not out.rstrip().endswith((".", "!", "?")):
        out = out.rstrip() + "."
    return out


# ------------------------------------------------------------------ agibot

AGIBOT_TASKS = {
    "Handing the flyer to the customer": "Hand the flyer to the customer.",
    "Handing out flyers to customers": "Hand out flyers to customers.",
    "Put the King Beans and green beans from the shelf into the shopping cart.": (
        "Put the King beans and green beans from the shelf into the shopping cart."
    ),
}
AGIBOT_VERBS = {
    "picked": "picks",
    "placed": "places",
    "threw": "throws",
    "passed": "passes",
    "pushed": "pushes",
}
_AGIBOT_VERB = re.compile(r"(?<!\bthe )\b(" + "|".join(AGIBOT_VERBS) + r")\b(?!-)")
"""A verb, not an adjective: not after "the" ("the picked peas"), not hyphenated ("picked-up")."""


def agibot_task(text: str) -> str:
    return AGIBOT_TASKS.get(text, text)


def agibot_subtask(text: str) -> str:
    return _AGIBOT_VERB.sub(lambda m: AGIBOT_VERBS[m.group(1)], text)


# ------------------------------------------------------------------ robomind

ROBOMIND_TYPO_FIXES: list[tuple[str, str]] = [
    (r"\bmochine\b", "machine"),
    (r"\borignal\b", "original"),
    (r"\bwifruite\b", "kiwifruit"),
    (r"\bleft had\b", "left hand"),
    (r"\bfurit\b", "fruit"),
    (r"\bthorw\b", "throw"),
    (r"\bobejct\b", "object"),
    (r"^otate\b", "rotate"),
    (r"^putthe\b", "put the"),
    (r"^pike up\b", "pick up"),
    (r"\bthe break\b(?=.*(?:shelf|machine|plate|mochine))", "the bread"),
    (r"\b[Uu]ncap open\b", "open"),
]


ROBOMIND_OVERRIDES: dict[str, str] = {
    "bread on the table": "Place the bread on the table.",
    "twist knob start the bread machine": "Twist the knob to start the bread machine.",
    "piled the yellow cube up on the purple cube": "Pile the yellow cube on the purple cube.",
    "flip marco cup": "Flip the mug.",
    "mobile marco cup": "Move the mug.",
    "side opening drawer": "Open the drawer from the side.",
    "side pull drawer": "Pull the drawer from the side.",
    "strawberries on a ceramic plate": "Place the strawberries on the ceramic plate.",
    "apples placed on a ceramic plate": "Place the apples on the ceramic plate.",
    "bananas placed on a ceramic plate": "Place the bananas on the ceramic plate.",
    "bread is placed on a ceramic plate": "Place the bread on the ceramic plate.",
    "chili peppers are placed on a ceramic plate": "Place the chili peppers on the ceramic plate.",
    "chili peppers are placed on the right side of a plastic tray": "Place the chili peppers on the right side of the plastic tray.",
    "cucumber placed on a ceramic plate": "Place the cucumber on the ceramic plate.",
    "red apple placed in the center of the desktop": "Place the red apple in the center of the desktop.",
    "yellow bananas placed on a plastic tray": "Place the yellow bananas on the plastic tray.",
    "yellow square placed on ceramic plate": "Place the yellow square on the ceramic plate.",
    "put the bread and put it into the plate": "Put the bread into the plate.",
    "slide close drawer close the drawer": "Slide close the drawer.",
    "pick cup pour cup place": "Pick up the cup, pour, and place the cup.",
    "pick pipe place plate twice": "Pick up the pipe and place it on the plate twice.",
    "push cup pick cup insert cup": "Push the cup, pick up the cup, and insert the cup.",
    "pour water with both arm": "Pour water with both arms.",
    "pull across the basket": "Pull the basket across the table.",
    "push away from the basket": "Push the basket away.",
    "put in the toy": "Place the toy into the container.",
    "place in trash": "Place the trash in the trash can.",
    "place trash": "Place the trash in the trash can.",
}


GERUND_MAP = {
    "placing": "place",
    "picking": "pick",
    "putting": "put",
    "transferring": "transfer",
    "opening": "open",
    "closing": "close",
    "making": "make",
    "retrieving": "retrieve",
    "taking": "take",
    "stacking": "stack",
    "moving": "move",
    "baking": "bake",
}


def robomind_task(s: str) -> str:
    """One RoboMIND task text (``RoboMIND_v1_2_instr.csv`` instruction or folder name), cleaned."""
    out = s.strip()
    if not out:
        return out

    override = ROBOMIND_OVERRIDES.get(out.lower())
    if override:
        return override

    for pat, repl in ROBOMIND_TYPO_FIXES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    def _restructure_hand(m: re.Match[str]) -> str:
        hand = m.group(1).lower()
        rest = m.group(2)
        rest = re.sub(r"^(\w+?)s\b", lambda v: v.group(1), rest, count=1)
        return f"With the {hand} hand, {rest}"

    out = re.sub(
        r"^(left|right)\s+hand\s+(.+)$", _restructure_hand, out, count=1, flags=re.IGNORECASE
    )

    hand_match = re.match(r"^(With the (?:left|right) hand, )(.*)", out)
    if hand_match:
        prefix, clause = hand_match.group(1), hand_match.group(2)
        clause = re.sub(
            r"^(pick up|put down|press down|press|pick|put|place|open|close|push|pull|"
            r"take|move|lift|drop|insert|remove|grab|turn|rotate|fetch|stack|stand|clean|wipe)\s+"
            r"(?!the\b|a\b|an\b|it\b|them\b|up\b|down\b|in\b|out\b|back\b|away\b|off\b|on\b"
            r"|open\b|close\b|across\b)",
            lambda m: m.group(1) + " the ",
            clause,
            count=1,
            flags=re.IGNORECASE,
        )
        out = prefix + clause

    out = re.sub(
        r"^the\s+(.+?)\s+in\s+the\s+(left|right)\s+hand\s+is\s+(\w+)\b(.*)$",
        lambda m: f"With the {m.group(2)} hand, {m.group(3)} the {m.group(1)}{m.group(4)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"^[Uu]pright\s+(.+)$", r"Stand \1 upright", out)
    out = re.sub(
        r"^(put|place)\s+(.+?)\s+in\s+(left|right)\s+hand\s+on\s+",
        lambda m: f"With the {m.group(3)} hand, {m.group(1)} {m.group(2)} on ",
        out,
        flags=re.IGNORECASE,
    )

    def _insert_article(m: re.Match[str]) -> str:
        rest = out[m.end() :]
        if re.match(r"\w+\s+\d", rest):
            return m.group(0)
        return m.group(1) + " the "

    out = re.sub(
        r"^(Pick up|Put down|Press down|Turn off|Turn on|Slide open|Slide close|"
        r"Rotate open|Rotate close|Cap close|Uncap open|Flap open|Flap close|"
        r"Press|Pick|Put|Place|Open|Close|Push|Pull|"
        r"Take|Move|Lift|Drop|Insert|Remove|Grab|Turn|Rotate|Fetch|Stack|Stand|Clean|Wipe)\s+"
        r"(?!the\b|a\b|an\b|it\b|them\b|up\b|down\b|in\b|out\b|back\b|away\b|off\b|on\b"
        r"|open\b|close\b|across\b|and\b|or\b|both\b|forward\b|backward\b|left\b|right\b)",
        _insert_article,
        out,
        count=1,
        flags=re.IGNORECASE,
    )

    out = re.sub(
        r"\b(from|on|in|into|onto|across|under|behind|near)\s+"
        r"(?!the\b|a\b|an\b|it\b|top\b|left\b|right\b|both\b|that\b|front\b"
        r"|this\b|these\b|those\b|each\b|every\b|some\b|any\b|no\b|\d|\w+ing\b"
        r"|from\b|on\b|in\b|into\b|onto\b|across\b|under\b|behind\b|near\b"
        r"|above\b|below\b|between\b|inside\b|outside\b|over\b"
        r"|position\b|place\b|order\b)",
        lambda m: m.group(1) + " the ",
        out,
        flags=re.IGNORECASE,
    )

    out = out.replace("，", ",")
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r",(?!\s)", ", ", out)
    out = out[0].upper() + out[1:]
    if not out.rstrip().endswith((".", "!", "?")):
        out = out.rstrip() + "."
    return out


def robomind_annotation(s: str) -> str:
    """One per-episode RoboMIND annotation (``task_summary``): gerund → imperative, articles,
    final period."""
    out = s.strip()
    if not out:
        return out
    out = out.replace("，", ",")
    words = out.split()
    if words and words[0].lower() in GERUND_MAP:
        words[0] = GERUND_MAP[words[0].lower()]
    out = " ".join(words)
    out = re.sub(r"\b[Aa]n?\s+(?=[a-z])", "the ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = out[0].upper() + out[1:]
    if not out.endswith((".", "!", "?")):
        out = out.rstrip() + "."
    return out


ROBOMIND = (
    "franka_1rgb",
    "franka_3rgb",
    "ur_1rgb",
    "franka_fr3_dual",
    "agilex_3rgb",
    "tienkung_gello_1rgb",
    "tienkung_prod1_gello_1rgb",
    "tienkung_xsens_1rgb",
)
"""The RoboMIND embodiments of the registry (dataset ``robomind_<embodiment>``)."""

RULES = {
    "galaxea": TextRules(galaxea_text, galaxea_text),
    "agibot_world_2026": TextRules(agibot_task, agibot_subtask),
    **{f"robomind_{e}": TextRules(robomind_task) for e in ROBOMIND},
}


def text_rules(name: str) -> TextRules:
    """The text rules of registry dataset ``name`` (identity when it has none)."""
    return RULES.get(name, TextRules())
