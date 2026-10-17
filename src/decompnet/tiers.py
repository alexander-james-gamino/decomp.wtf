"""Model tiers: which functions suit which agent, and how long it may try before moving on.

A helper's model decides what it can take on. A small, cheap model (Claude Haiku, a local model on your own GPU)
matches small functions on the first tries but burns compile after compile on hard ones; a large model is worth its
price on exactly those. So each helper gets a tier from its model, and the coordinator hands out functions by tier:

- local:  tiny functions, one at a time, a short leash (a model on your own machine: slow, and easily confused)
- small:  small functions in batches, the shortest leash of the cloud models (Claude Haiku and its peers)
- medium: what small models can't do: bigger functions, ones they gave up on (Claude Sonnet and its peers)
- large:  the frontier: big functions, near misses with real C differences, what everyone else released
          (Claude Opus and up)

A tier above another that is working right now takes what that one can't: a large model doesn't spend Opus prices
on functions a running Haiku helper would match anyway. With nobody below it, it works like before.

Whatever only differs in register allocation or instruction order goes to the permuter first, from every tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .scoring import DiffResult


@dataclass(frozen=True)
class Tier:
    name: str
    max_insns: int | None  # the biggest function it's handed automatically (None: any size)
    max_abandons: int | None  # functions given up on more often than this go to stronger models
    batch: int  # small functions claimed together (claim_function count is capped to this)
    stall_bonus: int  # compiles added to (or taken from) the size-based "move on" nudge
    hard_stop: bool  # refuse more compiles once twice the nudge has passed without a better score
    per_session: int  # claims per helper session
    strategy: str  # claim order when the agent doesn't ask for one
    guidance: str  # added to the helper's session prompt


TIERS: dict[str, Tier] = {
    "local": Tier(
        "local", max_insns=40, max_abandons=0, batch=1, stall_bonus=-1, hard_stop=True, per_session=1,
        strategy="easiest",
        guidance=(
            "You run on the user's own hardware. You get the smallest functions, one at a time.\n"
            "- Read the target assembly line by line before writing anything. Check which registers hold the\n"
            "  arguments ($a0 is the first, $a1 the second, ...) and keep unused leading arguments.\n"
            "- Declare every local at the top of the function. No inline asm.\n"
            "- Three compiles without a better score: release_function with a short note. Don't start over."),
    ),
    "small": Tier(
        "small", max_insns=80, max_abandons=1, batch=4, stall_bonus=0, hard_stop=True, per_session=2,
        strategy="easiest",
        guidance=(
            "You're a fast model on small functions; bigger and harder ones go to stronger models, so you never\n"
            "need to fight one.\n"
            "- Start from the best attempt or the draft in the brief; fix the first differing instruction first.\n"
            "- When compile_and_diff says to move on, release_function right away with what you tried.\n"
            "- When only registers or instruction order differ, that's the permuter's job: release_function with\n"
            "  leave_for=\"permuter\"."),
    ),
    "medium": Tier(
        "medium", max_insns=250, max_abandons=3, batch=3, stall_bonus=0, hard_stop=False, per_session=2,
        strategy="finish",
        guidance=(
            "You get what small models can't do: bigger functions and ones they gave up on. Read their notes first\n"
            "and don't repeat an approach they documented as failing. Get control flow and calls right before\n"
            "types and expressions; registers last. Types and structs you work out go in context.h (edit_context)\n"
            "so the small models after you benefit."),
    ),
    "large": Tier(
        "large", max_insns=None, max_abandons=None, batch=2, stall_bonus=2, hard_stop=False, per_session=2,
        strategy="frontier",
        guidance=(
            "You get the hardest work: big functions, near misses that still differ in real code, and what other\n"
            "agents released. Read every note on a function before you start, and work out why the others\n"
            "failed. If the cause is shared (a wrong type or prototype in context.h, a pattern the compiler\n"
            "needs), fix it with edit_context or leave a project note (add_note with no function): one fix can\n"
            "unblock many functions. If a function can't match because of a tool problem rather than the C,\n"
            "release_function with leave_for=\"human\" and say what's wrong."),
    ),
}
ORDER = ["local", "small", "medium", "large"]  # weakest first
STRONG_FAMILIES = ("opus", "fable", "mythos")
LOCAL_PROVIDERS = {"ollama", "lmstudio", "llamacpp", "llama.cpp", "local", "vllm", "mlx"}
# an Ollama-style model tag ("qwen3.8:27b", "qwen3-coder:30b-a3b-q4_K_M"): a model somebody runs themselves
_OLLAMA_TAG = re.compile(r"^[\w.\-]+:[\w.\-]+$")


def get(name: str | None) -> Tier | None:
    return TIERS.get((name or "").strip().lower())


def detect(kind: str, options: dict | None, is_local=None) -> str | None:
    """The tier of a helper from how it runs: its `tier` option when set ("any" switches routing off), else its
    model. None: unknown (the agent you talk to, a default model): no routing, everything as before."""
    options = options or {}
    explicit = str(options.get("tier") or "").strip().lower()
    if explicit in TIERS:
        return explicit
    if explicit in ("any", "none", "off"):
        return None
    model = str(options.get("model") or "").strip().lower()
    if kind == "api":
        base = str(options.get("base_url") or "")
        if is_local is None:
            from .app.agents import is_local as _is_local  # (agents imports engine, which imports this module)
            is_local = _is_local
        try:
            if base and is_local(base, resolve=False):
                return "local"
        except Exception:
            pass
    name = model.rsplit("/", 1)[-1]  # "anthropic/claude-haiku-4-5" -> "claude-haiku-4-5"
    if "haiku" in name:
        return "small"
    if any(f in name for f in STRONG_FAMILIES):
        return "large"
    if "sonnet" in name:
        return "medium"
    if kind == "opencode" and model:
        provider = model.split("/", 1)[0] if "/" in model else ""
        if provider in LOCAL_PROVIDERS or _OLLAMA_TAG.match(name):
            return "local"
    if kind == "api" and model and _OLLAMA_TAG.match(name):
        return "local"
    return None


def stronger(a: str | None, b: str | None) -> bool:
    """Whether tier `a` is above tier `b`."""
    return a in ORDER and b in ORDER and ORDER.index(a) > ORDER.index(b)


def diff_kind(d: DiffResult, verdict: bool | None = None) -> str | None:
    """What a non-matching attempt's diff says about the work left:
    regalloc  - the same instructions, only registers, order or stack offsets differ: the permuter's job
    data      - every instruction reads the same, only bytes or data differ (a jump table, strings, constants,
                or a tool problem): no rewrite of the code helps
    structure - missing, extra or different instructions: C work
    None for a match."""
    if d.bytes_match and verdict is not False:
        return None
    marks = {m for m, _, _ in d.rows if m != " "}
    if not marks:
        return "data"
    if marks <= {"r", "o", "s"}:
        return "regalloc"
    return "structure"
