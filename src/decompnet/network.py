"""The coordinator: who works on what, what counts as a match, and what everyone has learned.

Every public method here is one agent-facing action. They return plain text because the
consumer is an LLM reading a tool result; structured data lives in `status_json()` for
the website.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import policy, scoring
from .compiler import compile_source, header_asm_problems
from .elfobj import FunctionObj, Reloc, extract_function
from .project import compiler_config
from .store import Store

CLAIM_TTL = 30 * 60
# A function an agent just released goes to the back of the queue for this long, for every agent: the next one
# starts on something fresh instead of passing the same function back and forth. Asked for by name, it comes at once.
RELEASE_COOLDOWN = 15 * 60
# A function an agent released isn't handed back to that agent for this long (not forever: helpers keep one identity
# for days, and once they had released everything left, they got "nothing left to claim" while 175 functions were
# open). By then the shared header and the notes have usually moved on; asked for by name, it comes at once.
RELEASE_RETRY = 12 * 3600
MACHINE_AGENTS = ("autopilot", "permuter")  # decomp.wtf's own workers: their claims never mean "given up on"
# Twins first (see app/twins.py): a function whose code the game has more than once (boost: how many more times)
# and isn't too big or given up on, best value per instruction first; its twins (boost -1) last, as they match for
# free once it does.
_TWINS = ("(f.boost > 0 AND f.n_insns <= 400 AND f.abandons < 2) DESC, "
          "CASE WHEN f.boost > 0 THEN (f.boost + 1) * 64 / (f.n_insns + 16) ELSE 0 END DESC, f.boost < 0, ")
STRATEGIES = {
    "easiest": _TWINS + "f.abandons > 0, f.n_insns ASC",
    # what nobody gave up on, small first; among functions of about the same size, the files closest to done (a
    # file is linked into the game once all of it matches). Leftovers agents released again and again come last,
    # or a crowd of agents spends its time passing the hardest functions around.
    "finish": _TWINS + "f.abandons ASC, (f.n_insns + 31) / 32 ASC, (SELECT COUNT(*) FROM functions g WHERE "
                       "g.project_id=f.project_id AND g.unit=f.unit AND g.status='open') ASC, f.n_insns ASC",
    # near-misses, the ones given up on least first (else every agent resumes the same stuck one)
    "closest": "f.best_score IS NULL, f.abandons ASC, CAST(f.best_score AS REAL) / (f.n_insns * 100) ASC",
    "hardest": "f.n_insns DESC",
    "random": "RANDOM()",
}
# The default order isn't a fixed one: a weighted random pick (Network._weight), so concurrent helpers don't all start
# on the same smallest functions, and functions that soaked up visit after visit without progress sink to the back.
# "order-only" is the same pick from the permuter's near misses only, for a helper started on those on purpose.
WEIGHTED = ("shuffle", "order-only")
# Permuter runs on a function per round, and rounds (the second with longer runs: tools/permute/dp_worker.py), before
# a near miss that only differs in registers/order goes back to the agents (handoff cleared).
PERMUTER_TURNS = 3
PERMUTER_ROUNDS = 2
PERMUTER_RUNS = PERMUTER_TURNS * PERMUTER_ROUNDS
# The permuter's near misses while it has runs left: general agents don't get them (they'd fiddle with code the C can
# barely change), a helper started with strategy="order-only" gets only these.
POOL = ("(f.permuter_turns < %d AND (COALESCE(f.handoff, '') = 'permuter' "
        "OR (COALESCE(f.best_kind, '') = 'regalloc' AND COALESCE(f.best_score, 0) > 0)))" % PERMUTER_RUNS)
# Compiles without beating its best after which an agent must release a function: this many times the nudge
# (stall_note), which grows with the function's size.
HARD_STOP = 3
BIG_INSNS = 300  # from this size on, compile_and_diff adds a section-by-section report (scoring.section_report)
# Agents that claimed, released or matched something this recently count as working (for the division of labour).
ACTIVE_WINDOW = 20 * 60
LEAVE_FOR = ("permuter", "human")
# A brief must fit in one tool result: the biggest functions' assembly alone ran to 76,000 characters, more than an
# agent can receive, so they were released untried. Long assembly comes in pages (get_function page=N), and the
# parts that can grow without bound are capped (the whole best attempt: get_function page=-1).
ASM_PAGE_LINES = 450
BRIEF_SOURCE_MAX = 12000
BRIEF_NOTE_MAX = 700
# A note that starts with this is the function's status digest: the latest one heads its brief, so an agent sees
# what's been tried and what to try next without reading dozens of notes.
DIGEST = "DIGEST:"


class NetworkError(Exception):
    """An error meant to be shown to the agent as-is."""


@dataclass
class Agent:
    id: int
    name: str
    owner: str | None
    trusted: bool = False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _ago(ts: float | None) -> str:
    if not ts:
        return "never"
    s = int(time.time() - ts)
    return f"{s}s ago" if s < 120 else f"{s // 60}m ago" if s < 7200 else f"{s // 3600}h ago"


class Network:
    def __init__(self, db_path: str | Path, claim_ttl: int = CLAIM_TTL):
        self.store = Store(db_path)
        self.claim_ttl = claim_ttl
        self._rng = random.Random()  # the claim queue's weighted pick (tests seed it)
        self._pool = ThreadPoolExecutor(max_workers=8)

    # ------------------------------------------------------------------ agents

    def register_agent(self, name: str, owner: str | None = None, *, trusted: bool = True, origin: str = "cli",
                       terms_accepted: bool = False) -> str:
        """Mint an agent token. Operator-created (CLI) agents are trusted; self-registered (web) agents
        start at reputation level 'new' and earn the rest through verified matches."""
        name = name.strip()
        if not name or len(name) > 40 or not all(c.isalnum() or c in "-_." for c in name):
            raise NetworkError("agent name must be 1-40 chars of [A-Za-z0-9-_.]")
        owner = (owner or "").strip()[:60] or None
        token = "dn_" + secrets.token_urlsafe(24)
        now = time.time()
        with self.store.tx() as db:
            if db.execute("SELECT 1 FROM agents WHERE lower(name)=lower(?)", (name,)).fetchone():
                raise NetworkError(f"agent name {name!r} is taken")
            db.execute(
                "INSERT INTO agents(name,owner,token_hash,created_at,trusted,origin,terms_accepted_at) VALUES(?,?,?,?,?,?,?)",
                (name, owner, _hash_token(token), now, int(trusted), origin, now if terms_accepted else None),
            )
        return token

    def authenticate(self, token: str | None) -> Agent:
        if not token:
            raise NetworkError("missing API token: add `--header \"Authorization: Bearer dn_...\"` to your MCP config")
        row = self.store.one("SELECT id,name,owner,trusted FROM agents WHERE token_hash=?", _hash_token(token))
        if row is None:
            raise NetworkError("invalid API token")
        with self.store.tx() as db:
            db.execute("UPDATE agents SET last_seen=? WHERE id=?", (time.time(), row["id"]))
        return Agent(row["id"], row["name"], row["owner"], bool(row["trusted"]))

    def _require_budget(self, agent: Agent) -> None:
        ok, why = policy.compile_budget(self.store, agent.id)
        if not ok:
            raise NetworkError(why)

    def _require_level(self, agent: Agent, what: str) -> None:
        rep = policy.reputation(self.store, agent.id)
        if not rep["level"].can_edit_context:
            raise NetworkError(f"{what} needs reputation level 'member' (one verified match). You are '{rep['level'].name}': "
                               f"match a function first; add_note is open to everyone.")

    # ---------------------------------------------------------------- helpers

    def _project(self, slug: str | None):
        if slug:
            row = self.store.one("SELECT * FROM projects WHERE slug=?", slug)
            if row is None:
                raise NetworkError(f"no project {slug!r}; call list_projects")
            return row
        rows = self.store.all("SELECT * FROM projects")
        if len(rows) == 1:
            return rows[0]
        raise NetworkError("several projects exist; pass `project` (see list_projects)")

    def _function(self, name: str, project: str | None):
        if project is None and "/" in name and "@" not in name.split("/", 1)[0]:  # project/name (fork names hold units)
            project, name = name.split("/", 1)
        if project is None:
            rows = self.store.all("SELECT * FROM functions WHERE name=?", name)
            if len(rows) > 1:
                raise NetworkError(f"{name!r} exists in several projects; pass `project`")
            row = rows[0] if rows else None
        else:
            row = self.store.one(
                "SELECT f.* FROM functions f JOIN projects p ON p.id=f.project_id WHERE p.slug=? AND f.name=?",
                project, name,
            )
        if row is None:
            raise NetworkError(f"no function {name!r}" + (f" in {project!r}" if project else ""))
        return row

    def _context(self, project_id: int, unit: str | None = ""):
        """Latest header for a translation unit, falling back to the project-wide one. Real codebases have
        one header set per source file; small projects just have the project-wide header ('' unit)."""
        unit = unit or ""
        return self.store.one(
            "SELECT * FROM context_versions WHERE project_id=? AND unit IN (?, '') "
            "ORDER BY unit=? DESC, version DESC LIMIT 1", project_id, unit, unit)

    def _unit_contexts(self, project_id: int) -> set[str]:
        return {r["unit"] for r in self.store.all(
            "SELECT DISTINCT unit FROM context_versions WHERE project_id=? AND unit != ''", project_id)}

    def _effective_unit(self, fn, units: set[str]) -> str:
        return fn["unit"] if fn["unit"] in units else ""

    @staticmethod
    def _hash_only(fn) -> bool:
        return fn["target"] is None

    def _target_obj(self, fn, proj) -> FunctionObj:
        return FunctionObj(
            name=fn["name"], arch=proj["arch"], big_endian=bool(proj["big_endian"]), code=fn["target"],
            relocs=[Reloc.from_json(r) for r in json.loads(fn["relocs"])],
        )

    def _expire_claims(self, db) -> None:
        now = time.time()
        own = {r["id"] for r in db.execute("SELECT id FROM agents WHERE name IN (?, ?) OR name LIKE 'permuter-%'",
                                            MACHINE_AGENTS)}  # more permuter workers: permuter-2, ...
        for row in db.execute("SELECT c.function_id, c.agent_id, c.claimed_at, f.project_id, f.name FROM claims c "
                              "JOIN functions f ON f.id=c.function_id WHERE c.expires_at < ?", (now,)).fetchall():
            if row["agent_id"] in own:  # decomp.wtf's own tries (cut short by a restart): nobody gave up on it
                continue
            db.execute("UPDATE functions SET abandons=abandons+1 WHERE id=?", (row["function_id"],))
            self._count_visit(db, row["function_id"], row["claimed_at"])
            self._event(db, row["project_id"], row["agent_id"], "expired", row["name"])
        db.execute("DELETE FROM claims WHERE expires_at < ?", (now,))

    @staticmethod
    def _count_visit(db, function_id: int, claimed_at: float) -> None:
        """A claim on a function ended (released, or its lease ran out). If its best score didn't improve during the
        claim, that's one more fruitless visit: the claim queue makes such functions less likely (_weight)."""
        db.execute("UPDATE functions SET stale_visits=stale_visits+1 WHERE id=? AND status='open' "
                   "AND (best_at IS NULL OR best_at < ?)", (function_id, claimed_at))

    def _event(self, db, project_id, agent_id, kind, detail) -> None:
        db.execute("INSERT INTO events(project_id,agent_id,kind,detail,created_at) VALUES(?,?,?,?,?)",
                   (project_id, agent_id, kind, detail, time.time()))

    def _agent_name(self, agent_id) -> str:
        row = self.store.one("SELECT name FROM agents WHERE id=?", agent_id) if agent_id else None
        return row["name"] if row else "?"

    def _compile(self, proj, header: str, source: str):
        root = Path(proj["root"])
        return compile_source(compiler_config(root, json.loads(proj["config"])), header, source)

    def _evaluate(self, fn, proj, obj: bytes) -> tuple[FunctionObj | None, bool | None, list[str]]:
        """(candidate, verdict, problems) for a compiled object. verdict None means: decide by comparing
        fingerprints. Projects whose targets come from a ROM link the candidate instead (see app.engine)."""
        return extract_function(obj, fn["name"]), None, []

    def _asm_text(self, fn, proj) -> str:
        return fn["asm"]

    def _extra_brief(self, fn, proj) -> list[str]:
        return []

    # ---------------------------------------------------------------- projects

    def list_projects(self) -> str:
        rows = self.store.all(
            """SELECT p.slug, p.name, p.arch, p.description, p.config, COUNT(f.id) AS total,
                      SUM(f.status='matched') AS matched, SUM(f.size) AS bytes,
                      SUM(CASE WHEN f.status='matched' THEN f.size ELSE 0 END) AS matched_bytes
               FROM projects p LEFT JOIN functions f ON f.project_id=p.id GROUP BY p.id ORDER BY p.slug"""
        )
        if not rows:
            return "No projects yet."
        out = []
        for r in rows:
            pct = 100 * (r["matched_bytes"] or 0) / (r["bytes"] or 1)
            legal = json.loads(r["config"]).get("legal", {})
            out.append(f"- {r['slug']}: {r['name']} [{r['arch']}] — {r['matched'] or 0}/{r['total']} functions "
                       f"matched ({pct:.1f}% of code); legal basis: {legal.get('basis')}, contributions licensed "
                       f"{legal.get('license')}\n  {r['description']}".rstrip())
        return "\n".join(out)

    def get_context(self, project: str | None, function: str | None = None) -> str:
        if function:
            fn = self._function(function, project)
            proj = self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"])
            ctx = self._context(proj["id"], fn["unit"])
        else:
            proj = self._project(project)
            if self._unit_contexts(proj["id"]):
                raise NetworkError(f"{proj['slug']} has one header per source file: pass `function` to get the "
                                   f"header that function compiles against")
            ctx = self._context(proj["id"])
        return (f"// context.h for {proj['slug']} — version {ctx['version']} "
                f"(prepended to every compile as #include \"context.h\")\n{ctx['header']}")

    # ------------------------------------------------------------------ claims

    def claim(self, agent: Agent, project: str | None = None, function: str | None = None,
              strategy: str = "easiest") -> str:
        fn = self.claim_row(agent, project, function, strategy)
        if fn is None:
            proj = self._project(project)
            return f"Nothing left to claim in {proj['slug']} — every open function is matched or claimed. 🎉"
        return self._brief(fn, agent, header="CLAIMED")

    def claim_row(self, agent: Agent, project: str | None = None, function: str | None = None,
                  strategy: str = "shuffle"):
        """Claim (or renew) a function and return its row, or None when nothing is left."""
        if strategy not in STRATEGIES and strategy not in WEIGHTED:
            raise NetworkError(f"strategy must be one of {', '.join([*WEIGHTED, *STRATEGIES])}")
        proj = self._project(project) if (project or function is None) else None
        now = time.time()
        with self.store.tx() as db:
            self._expire_claims(db)
            held = db.execute("SELECT f.* FROM claims c JOIN functions f ON f.id=c.function_id WHERE c.agent_id=?",
                              (agent.id,)).fetchone()
            if held is not None:
                if function and held["id"] != self._function(function, proj["slug"] if proj else None)["id"]:
                    raise NetworkError(f"you already hold {held['name']}; match it or release_function it first")
                fn = held
                db.execute("UPDATE claims SET expires_at=? WHERE function_id=?", (now + self.claim_ttl, fn["id"]))
            elif function:
                fn = self._function(function, proj["slug"] if proj else None)
                if fn["status"] == "matched":
                    raise NetworkError(f"{fn['name']} is already matched; use get_function to read its source")
                if fn["status"] == "asm":
                    raise NetworkError(f"{fn['name']} is hand-written assembly: no C compiles to it, so it's left out")
                other = db.execute("SELECT agent_id, expires_at FROM claims WHERE function_id=?", (fn["id"],)).fetchone()
                if other is not None:
                    raise NetworkError(f"{fn['name']} is claimed by {self._agent_name(other['agent_id'])} "
                                       f"for another {int(other['expires_at'] - now) // 60} min")
                if fn["handoff"] == "human":  # asked for by name: somebody is looking at it now
                    db.execute("UPDATE functions SET handoff=NULL, handoff_note=?, handoff_at=? WHERE id=?",
                               (f"taken back by {agent.name}", now, fn["id"]))
                db.execute("INSERT INTO claims VALUES(?,?,?,?)", (fn["id"], agent.id, now, now + self.claim_ttl))
                self._event(db, fn["project_id"], agent.id, "claim", fn["name"])
            else:
                fn = self._next_for(db, agent, proj, strategy, now)
                if fn is None:
                    return None
                db.execute("INSERT INTO claims VALUES(?,?,?,?)", (fn["id"], agent.id, now, now + self.claim_ttl))
                self._event(db, fn["project_id"], agent.id, "claim", fn["name"])
        return fn

    # ------------------------------------------------------------- who gets what

    def agent_limit(self, agent: Agent | int) -> int | None:
        """The largest function (instructions) this agent is handed automatically; None: any size."""
        row = self.store.one("SELECT max_insns FROM agents WHERE id=?", agent if isinstance(agent, int) else agent.id)
        return row["max_insns"] if row and row["max_insns"] else None

    def set_agent_limit(self, agent_id: int, max_insns: int | None) -> None:
        with self.store.tx() as db:
            db.execute("UPDATE agents SET max_insns=? WHERE id=?", (int(max_insns) if max_insns else None, agent_id))

    def default_strategy(self, agent: Agent, project: str | None) -> str:
        """The claim order an agent gets when it doesn't ask for one."""
        return "shuffle"

    def _eligible(self, db, agent: Agent, proj, now: float, own_releases: bool = False,
                  pool: bool | None = False) -> tuple[str, list]:
        """(WHERE clause, its parameters) for what this agent may be handed automatically: open, unclaimed, not left
        for a human, within its size limit. own_releases: also what it released lately (RELEASE_RETRY). pool: False
        leaves out the permuter's near misses (POOL), True takes only those, None both."""
        where = ("f.project_id=? AND f.status='open' AND f.id NOT IN (SELECT function_id FROM claims) "
                 "AND COALESCE(f.handoff, '') != 'human'")
        params: list = [proj["id"]]
        if not own_releases:
            where += (" AND f.name NOT IN (SELECT detail FROM events WHERE project_id=? AND agent_id=? "
                      "AND kind='release' AND created_at > ?)")
            params += [proj["id"], agent.id, now - RELEASE_RETRY]
        if pool is not None:
            where += f" AND {'' if pool else 'NOT '}{POOL}"
        if (limit := self.agent_limit(agent)) is not None:
            where += " AND f.n_insns <= ?"
            params.append(limit)
        return where, params

    def _next_for(self, db, agent: Agent, proj, strategy: str, now: float):
        """The function an automatic claim hands out. First from what nobody's on, not left for a human, within the
        agent's size limit and not the permuter's near misses; if that's all taken, what this agent released lately;
        then the permuter's near misses. "shuffle" (the default) picks at random, weighted (_weight); "order-only"
        picks from the permuter's near misses only; the other strategies are fixed orders."""
        stages = ([(False, True), (True, True)] if strategy == "order-only"
                  else [(False, False), (True, False), (True, None)])
        for own, pool in stages:
            where, params = self._eligible(db, agent, proj, now, own, pool)
            if strategy in WEIGHTED:
                rows = db.execute(f"SELECT f.* FROM functions f WHERE {where}", params).fetchall()
                if rows:
                    return self._pick(db, proj, rows, now, closest=strategy == "order-only")
                continue
            row = db.execute(
                f"""SELECT f.* FROM functions f WHERE {where}
                    ORDER BY f.name IN (SELECT detail FROM events WHERE project_id=? AND kind='release'
                                        AND created_at > ?), {STRATEGIES[strategy]} LIMIT 1""",
                (*params, proj["id"], now - RELEASE_COOLDOWN)).fetchone()
            if row is not None:
                return row
        return None

    @staticmethod
    def _weight(fn, now: float, cooling: set, closest: bool = False) -> float:
        """How likely a function is to be handed out next.
        - Every claim on it that ended without beating its best score makes it less likely: 1 / (1 + visits/4)^2,
          so 4 such visits give a quarter, 12 a sixteenth, 40 about a hundredth. Stuck functions sink but never
          vanish: they still get a fresh look now and then. (On slus-00844, 51 functions had soaked up 42% of all
          fruitless visits.)
        - Improved in the last day: twice as likely.
        - Code the game has more than once (twins): more likely; the twins themselves, less (they match for free).
        - Released by anyone in the last minutes (RELEASE_COOLDOWN): much less likely, so the next agent starts on
          something fresh.
        - closest (the permuter's near misses): closer to done, more likely."""
        w = 1.0 / (1.0 + (fn["stale_visits"] or 0) / 4.0) ** 2
        if fn["best_at"] and now - fn["best_at"] < 86400:
            w *= 2.0
        boost = fn["boost"] or 0
        if boost > 0 and (fn["n_insns"] or 0) <= 400:
            w *= 1.0 + min(boost, 3)
        elif boost < 0:
            w *= 0.25
        if fn["name"] in cooling:
            w *= 0.05
        if closest and fn["best_score"] and fn["n_insns"]:
            w *= 1.0 / (1.0 + fn["best_score"] / fn["n_insns"])
        return w

    def _pick(self, db, proj, rows, now: float, closest: bool = False):
        cooling = {r["detail"] for r in db.execute(
            "SELECT DISTINCT detail FROM events WHERE project_id=? AND kind='release' AND created_at > ?",
            (proj["id"], now - RELEASE_COOLDOWN))}
        return self._rng.choices(rows, weights=[self._weight(r, now, cooling, closest) for r in rows], k=1)[0]

    def left_for(self, agent: Agent, project: str, include_claimed: bool = True) -> int:
        """How many open functions this agent could still be handed: within its size limit, its own recent releases
        and the permuter's near misses included. include_claimed: count ones others hold now too (they may come
        back)."""
        proj = self._project(project)
        with self.store.lock:
            where, params = self._eligible(self.store.conn, agent, proj, time.time(), True, None)
            if include_claimed:
                where = where.replace("AND f.id NOT IN (SELECT function_id FROM claims) ", "")
            return self.store.conn.execute(f"SELECT COUNT(*) AS n FROM functions f WHERE {where}", params).fetchone()["n"]

    def fits(self, agent: Agent, fn) -> bool:
        """Whether a function may be handed to the agent automatically from the user's priority list: not left for a
        human or the permuter, and within the agent's size limit."""
        if fn["handoff"]:
            return False
        limit = self.agent_limit(agent)
        return limit is None or (fn["n_insns"] or 0) <= limit

    # ------------------------------------------------------------- the permuter's queue

    def permuter_work(self, fn) -> bool:
        """Whether a function is the permuter's kind of work: a best attempt to start from whose diff is registers/
        order only, or that is within 3% of matching (score <= 3 per instruction), and permuter runs left."""
        if not fn["best_source"] or not (fn["best_score"] or 0) > 0 or fn["permuter_turns"] >= PERMUTER_RUNS:
            return False
        return (fn["best_kind"] or "") == "regalloc" or fn["best_score"] <= 3 * (fn["n_insns"] or 0)

    def permuter_queue(self, proj, limit: int = 20, max_score: int | None = None, near: float = 0.06,
                       max_insns: int = 400) -> list:
        """What the permuter should run on next, best bets first: functions agents left for it, then near misses
        whose diff is registers/order only (the server sees that on every compile), then the closest near misses.
        Fresh ones (fewer permuter runs) before ones it already tried; never data-only differences (no rewrite of
        the code fixes those) or what was left for a human."""
        sql = ("SELECT * FROM functions WHERE project_id=? AND status='open' AND best_source IS NOT NULL "
               "AND best_score > 0 AND permuter_turns < ? AND COALESCE(best_kind, '') != 'data' "
               "AND COALESCE(handoff, '') != 'human' AND id NOT IN (SELECT function_id FROM claims) "
               "AND (COALESCE(handoff, '') = 'permuter' OR (n_insns <= ? AND (best_kind = 'regalloc' "
               "     OR best_score <= n_insns * 100 * ?)))")
        params: list = [proj["id"], PERMUTER_RUNS, max_insns, near]
        if max_score:
            sql += " AND best_score <= ?"
            params.append(int(max_score))
        sql += (" ORDER BY COALESCE(handoff, '') = 'permuter' DESC, COALESCE(best_kind, '') = 'regalloc' DESC, "
                "permuter_turns ASC, CAST(best_score AS REAL) / (n_insns * 100) ASC, n_insns ASC LIMIT ?")
        params.append(int(limit))
        return self.store.all(sql, *params)

    def permuter_done(self, agent: Agent, function_id: int, notes: str = "") -> None:
        """A permuter run on a function ended (its claim, if any, goes; no "given up" is counted). After
        PERMUTER_RUNS runs without a match (two rounds) it goes back to the agents."""
        now = time.time()
        with self.store.tx() as db:
            db.execute("DELETE FROM claims WHERE function_id=? AND agent_id=?", (function_id, agent.id))
            fn = db.execute("SELECT * FROM functions WHERE id=?", (function_id,)).fetchone()
            if fn is None or fn["status"] == "matched":
                return
            turns = fn["permuter_turns"] + 1
            db.execute("UPDATE functions SET permuter_turns=? WHERE id=?", (turns, function_id))
            if turns >= PERMUTER_RUNS and fn["handoff"] == "permuter":
                db.execute("UPDATE functions SET handoff=NULL, handoff_note=?, handoff_at=? WHERE id=?",
                           (f"permuter: {turns} runs, best score {fn['best_score']}; back to the agents", now, function_id))
            if notes.strip():
                db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,?)",
                           (fn["project_id"], function_id, agent.id, notes.strip(), now))
            self._event(db, fn["project_id"], agent.id, "permuted", fn["name"])

    def set_handoff(self, project: str | None, function: str, to: str | None, note: str = "") -> str:
        """Leave a function for the permuter or a human (no automatic claims), or give it back (to=None)."""
        fn = self._function(function, project)
        to = (to or "").strip().lower() or None
        if to is not None and to not in LEAVE_FOR:
            raise NetworkError(f"leave it for one of: {', '.join(LEAVE_FOR)}")
        with self.store.tx() as db:
            db.execute("UPDATE functions SET handoff=?, handoff_note=?, handoff_at=? WHERE id=?",
                       (to, note.strip() or None, time.time(), fn["id"]))
        return f"{fn['name']}: " + (f"left for the {to}" if to else "back in the queue for agents")

    def release(self, agent: Agent, function: str, notes: str, project: str | None = None, leave_for: str = "") -> str:
        """Give a function up. leave_for: "permuter" (only registers/order differ: no more agents on it until the
        permuter has had its runs) or "human" (it can't match for a reason C can't fix, e.g. a tool problem).
        Without it, a function whose best attempt differs only in registers/order, or whose note says so, goes to
        the permuter anyway."""
        fn = self._function(function, project)
        leave_for = (leave_for or "").strip().lower()
        if leave_for and leave_for not in LEAVE_FOR:
            raise NetworkError(f"leave_for must be one of: {', '.join(LEAVE_FOR)} (or empty)")
        why = notes.strip()
        refused = ""
        if not leave_for and self.permuter_work(fn) and (
                (fn["best_kind"] or "") == "regalloc"
                or re.search(r"(?<!not )(?<!no )\bregalloc(?:ation)?[- ]only\b", notes, re.I)):
            leave_for, why = "permuter", "auto: the best attempt differs only in registers/order. " + why
        if leave_for == "permuter" and not self.permuter_work(fn):
            # the permuter only reorders code and reshuffles registers: a function whose best attempt still differs
            # in real instructions would only sit in its queue, and agents would never get it back
            leave_for = ""
            refused = (" Not left for the permuter: its best attempt still differs in real instructions (or the "
                       "permuter already had its runs), so it stays with the agents.")
        now = time.time()
        with self.store.tx() as db:
            row = db.execute("SELECT agent_id FROM claims WHERE function_id=?", (fn["id"],)).fetchone()
            if row is None or row["agent_id"] != agent.id:
                raise NetworkError(f"you don't hold a claim on {fn['name']}")
            claimed = db.execute("SELECT claimed_at FROM claims WHERE function_id=?", (fn["id"],)).fetchone()
            db.execute("DELETE FROM claims WHERE function_id=?", (fn["id"],))
            db.execute("UPDATE functions SET abandons=abandons+1 WHERE id=?", (fn["id"],))
            self._count_visit(db, fn["id"], claimed["claimed_at"] if claimed else now)
            if leave_for:
                db.execute("UPDATE functions SET handoff=?, handoff_note=?, handoff_at=? WHERE id=?",
                           (leave_for, f"{agent.name}: {why}"[:600], now, fn["id"]))
            if notes.strip():
                db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,?)",
                           (fn["project_id"], fn["id"], agent.id, notes.strip(), now))
            self._event(db, fn["project_id"], agent.id, "release", fn["name"])
        left = {"permuter": " Left for the permuter: agents get it back only if the permuter can't match it.",
                "human": " Left for a human: no agent gets it automatically until someone looks at it."}
        return (f"Released {fn['name']}. Your best attempt and notes are saved so the next agent starts "
                f"where you left off.{left.get(leave_for, '')}{refused} Call claim_function for a new one.")

    def get_function(self, function: str, project: str | None = None, agent: Agent | None = None,
                     page: int = 0) -> str:
        """The function's brief (page 0), one page of its target assembly (page 1, 2, ...), or its whole best attempt
        or machine draft (page -1)."""
        fn = self._function(function, project)
        proj = self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"])
        if not page or self._hash_only(fn) or json.loads(proj["config"] or "{}").get("kind") == "fork":
            return self._brief(fn, agent, header="VIEW")
        if page < 0:
            src = (fn["best_source"] or fn["seed"] or "").strip()
            if not src:
                return f"{fn['name']} has no attempt or machine draft yet."
            what = f"best attempt, score {fn['best_score']}" if fn["best_source"] else "machine draft"
            return f"## {fn['name']}: {what}\n```c\n{src}\n```"
        asm = scoring.compact_asm(self._asm_text(fn, proj)).splitlines()
        pages = max(1, -(-len(asm) // ASM_PAGE_LINES))
        if page > pages:
            raise NetworkError(f"{fn['name']}'s assembly has {pages} page{'s' if pages != 1 else ''}")
        first = (page - 1) * ASM_PAGE_LINES
        chunk = asm[first:first + ASM_PAGE_LINES]
        return (f"## {fn['name']}: target assembly, page {page} of {pages} (lines {first + 1}-{first + len(chunk)} "
                f"of {len(asm)})\n```asm\n" + "\n".join(chunk) + "\n```")

    def _clip(self, text: str, limit: int, rest: str) -> str:
        text = text.strip()
        return text if len(text) <= limit else text[:limit].rstrip() + f"\n... ({len(text) - limit:,} more characters: {rest})"

    def _brief(self, fn, agent: Agent | None, header: str) -> str:
        proj = self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"])
        ctx = self._context(proj["id"], fn["unit"])
        claim = self.store.one("SELECT * FROM claims WHERE function_id=?", fn["id"])
        lines = [f"## {header}: {proj['slug']}/{fn['name']}",
                 f"size {fn['size']} bytes, {fn['n_insns']} instructions, arch {proj['arch']}, "
                 f"context.h v{ctx['version']} ({len(ctx['header'].splitlines())} lines)"]
        if claim:
            who = "you" if agent and claim["agent_id"] == agent.id else self._agent_name(claim["agent_id"])
            lines.append(f"claimed by {who}, lease expires in {int(claim['expires_at'] - time.time()) // 60} min "
                         f"(each compile_and_diff renews it)")
        digest = self.store.one(
            "SELECT n.text, n.created_at, a.name FROM notes n LEFT JOIN agents a ON a.id=n.agent_id "
            "WHERE n.function_id=? AND n.text LIKE ? ORDER BY n.id DESC LIMIT 1", fn["id"], DIGEST + "%")
        if digest and fn["status"] != "matched":
            lines += ["", f"### Status digest ({digest['name'] or 'review'}, {_ago(digest['created_at'])}): what's been "
                          "tried and what to try next. Keep it current: add_note with this function and a text that "
                          f"starts with {DIGEST}",
                      self._clip(digest["text"][len(DIGEST):], 2 * BRIEF_NOTE_MAX, "the rest is in search_notes")]
        if fn["status"] == "matched":
            lines += [f"MATCHED by {self._agent_name(fn['matched_agent_id'])} {_ago(fn['matched_at'])}:",
                      "```c", fn["matched_source"].strip(), "```"]
        if self._hash_only(fn):
            lines += ["", "### Target assembly", "This is a hash-only project: the coordinator keeps a one-way fingerprint of "
                      "each function, never its code. Disassemble your own lawful copy with "
                      f"`decompnet local asm <your-object.o> {fn['name']}` and diff locally with `decompnet local diff`."]
        else:
            asm = scoring.compact_asm(self._asm_text(fn, proj)).splitlines()
            pages = max(1, -(-len(asm) // ASM_PAGE_LINES))
            lines += ["", "### Target assembly" + (f" (page 1 of {pages})" if pages > 1 else ""), "```asm",
                      *asm[:ASM_PAGE_LINES], "```"]
            if pages > 1:
                lines.append(f"(the assembly continues: {len(asm) - ASM_PAGE_LINES} more lines. Read them with "
                             f"get_function(function=\"{fn['name']}\", page=2) up to page={pages}, one page at a time "
                             "as you write that part)")
        lines += self._extra_brief(fn, proj)
        if fn["status"] != "matched":
            if fn["best_source"]:
                lines += ["", f"### Best attempt so far: score {fn['best_score']} by "
                              f"{self._agent_name(fn['best_agent_id'])} ({fn['attempts']} attempts total) — "
                              f"start from this", "```c",
                          self._clip(fn["best_source"], BRIEF_SOURCE_MAX, "get_function with page=-1 shows all of it"),
                          "```"]
            elif fn["seed"]:
                lines += ["", "### Machine-generated starting point (m2c: expect wrong types and names; "
                              "the context header helps)", "```c",
                          self._clip(fn["seed"], BRIEF_SOURCE_MAX, "get_function with page=-1 shows all of it"), "```"]
        notes = self.store.all(
            "SELECT n.text, n.created_at, a.name FROM notes n LEFT JOIN agents a ON a.id=n.agent_id "
            "WHERE n.function_id=? AND n.text NOT LIKE ? ORDER BY n.id DESC LIMIT 8", fn["id"], DIGEST + "%")
        if notes:
            lines += ["", "### Notes from previous agents on this function (latest first)"]
            lines += [f"- ({n['name']}, {_ago(n['created_at'])}) "
                      + self._clip(" ".join(n["text"].split()), BRIEF_NOTE_MAX, "search_notes has the full note")
                      for n in notes]
        pnotes = self.store.all(
            "SELECT n.text, a.name FROM notes n LEFT JOIN agents a ON a.id=n.agent_id "
            "WHERE n.project_id=? AND n.function_id IS NULL ORDER BY n.id DESC LIMIT 5", proj["id"])
        if pnotes:
            lines += ["", "### Project-wide notes (latest)"] + [
                f"- ({n['name']}) " + self._clip(" ".join(n["text"].split()), 400, "search_notes has the full note")
                for n in pnotes]
        if fn["unit"] and fn["unit"] in self._unit_contexts(proj["id"]):
            lines.insert(2, f"source unit {fn['unit']}: call get_context with function=\"{fn['name']}\" for its header")
        if header == "CLAIMED":
            lines += ["", "Next: read context.h with get_context, write C for this function, and iterate with "
                          "compile_and_diff until score is 0. Submit only the function body (plus any helper "
                          "declarations); context.h is included automatically."]
        return "\n".join(lines)

    # --------------------------------------------------------- compile & diff

    def compile_and_diff(self, agent: Agent, function: str, source: str, project: str | None = None) -> str:
        return self.attempt(agent, function, source, project)["text"]

    def attempt(self, agent: Agent, function: str, source: str, project: str | None = None) -> dict:
        """Compile, score and record one attempt. Returns {"text", "score", "matched"} (score None: didn't compile)."""
        self._require_budget(agent)
        fn = self._function(function, project)
        proj = self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"])
        now = time.time()

        # Claim bookkeeping first: you can't score (or get credit) on someone else's function.
        with self.store.tx() as db:
            self._expire_claims(db)
            claim = db.execute("SELECT agent_id FROM claims WHERE function_id=?", (fn["id"],)).fetchone()
            scratch = fn["status"] == "matched"
            if not scratch:
                if claim is None:
                    held = db.execute("SELECT 1 FROM claims WHERE agent_id=?", (agent.id,)).fetchone()
                    if held:
                        raise NetworkError("you hold another claim; release it before working on a different function")
                    db.execute("INSERT INTO claims VALUES(?,?,?,?)", (fn["id"], agent.id, now, now + self.claim_ttl))
                    self._event(db, fn["project_id"], agent.id, "claim", fn["name"])
                elif claim["agent_id"] != agent.id:
                    raise NetworkError(f"{fn['name']} is claimed by {self._agent_name(claim['agent_id'])}")
                else:
                    db.execute("UPDATE claims SET expires_at=? WHERE function_id=?", (now + self.claim_ttl, fn["id"]))

        if not scratch:
            since, after = self._stall(fn["id"], agent.id)
            if since >= HARD_STOP * after:
                raise NetworkError(
                    f"{since} compiles of {fn['name']} without beating your best: stop here. release_function with "
                    "notes (what you tried, what's left; leave_for=\"permuter\" if only registers or order differ), "
                    "then claim_function for the next one.")

        ctx = self._context(proj["id"], fn["unit"])
        result = self._compile(proj, ctx["header"], source)

        if not result.ok:
            self._record(fn, agent, None, source, ctx["version"])
            return {"text": f"❌ COMPILE ERROR (context v{ctx['version']})\n{result.log.strip()}", "score": None, "matched": False}

        cand, verdict, problems = self._evaluate(fn, proj, result.obj)
        if cand is None:
            self._record(fn, agent, None, source, ctx["version"])
            return {"text": f"❌ Compiled, but the object has no function named `{fn['name']}`. "
                            f"Define it with exactly that name (non-static).", "score": None, "matched": False}
        if self._hash_only(fn):
            matched_now = cand.fingerprint() == fn["fingerprint"] and not scratch
            self._record(fn, agent, 0 if matched_now else None, source, ctx["version"], matched=matched_now, compiled=True)
            if matched_now:
                return {"text": f"✅ MATCH! {fn['name']} matches the target fingerprint. Credited to {agent.name}; "
                                f"your claim is released.", "score": 0, "matched": True}
            return {"text": "❌ No match. This is a hash-only project, so the coordinator can only say match / no match. "
                            f"For a scored diff, run `decompnet local diff <your-object.o> {fn['name']} <your.c>` "
                            "against your own lawful copy.", "score": None, "matched": False}
        d = scoring.diff(self._target_obj(fn, proj), cand)
        if verdict is not None:
            d = scoring.DiffResult(0 if verdict else max(1, d.score), d.max_score, d.rows, verdict)
        matched_now = d.bytes_match and not scratch
        self._record(fn, agent, d.score, source, ctx["version"], matched=matched_now)
        kind = scoring.diff_kind(d, verdict)
        if kind and not scratch:  # what the best attempt's diff says is left (the permuter's queue reads this)
            with self.store.tx() as db:
                db.execute("UPDATE functions SET best_kind=? WHERE id=? AND best_score=? AND best_source=?",
                           (kind, fn["id"], d.score, source))

        if scratch:
            head = f"(scratch — {fn['name']} was already matched) score {d.score}"
        elif d.bytes_match:
            return {"text": f"✅ MATCH! {fn['name']} is byte-identical to the target ({fn['n_insns']} instructions). "
                            f"Credited to {agent.name}; your claim is released. Call claim_function for the next one.",
                    "score": 0, "matched": True}
        else:
            best = self.store.one("SELECT best_score FROM functions WHERE id=?", fn["id"])["best_score"]
            head = (f"score {d.score} ({scoring.percent_text(d.percent)} match) — best so far {best}. "
                    f"Target {len([r for r in d.rows if r[1]])} insns, yours {len([r for r in d.rows if r[2]])}.")
            if d.rows and all(m == " " for m, _, _ in d.rows):
                head += ("\nNOT A MATCH YET: every instruction reads the same, but the bytes differ"
                         + (" because of the linking notes at the end: fix those." if problems else
                            ": an encoding the listing doesn't show (branch form, an immediate, a relocation)."))
                head += ("\nA switch's case values come only from its jump table: read_data the jtbl_ address and "
                         "label each case by where its entry points. If your data is right and it still differs, it "
                         "may be a tool problem: release_function with leave_for=\"human\" and say what you checked.")
            elif kind == "regalloc":
                head += ("\nOnly registers, instruction order or stack offsets differ: the permuter's job. Unless you "
                         "see a concrete fix (a pin from context.h on the register that differs often is one), "
                         "release_function with leave_for=\"permuter\" and claim the next one.")
        warn = f"\ncompiler output:\n{result.log.strip()}" if result.log.strip() else ""
        if problems:
            warn += "\nwhen linking your code at the function's address:\n" + "\n".join(f"- {p}" for p in problems)
        sections = ""
        if (fn["n_insns"] or 0) >= BIG_INSNS and fn["vram"] and d.rows:
            labels = [int(m, 16) for m in re.findall(r"^\.L([0-9A-Fa-f]{8}):", fn["asm"] or "", re.M)]
            sections = "\n" + scoring.section_report(d.rows, fn["vram"], labels)
        return {"text": f"{head}{sections}\n{d.render()}{warn}{self.stall_note(fn['id'], agent.id)}", "score": d.score, "matched": False}

    def stall_note(self, function_id: int, agent_id: int, after: int | None = None) -> str:
        """A nudge once an agent's last `after` compiles of a function didn't beat its best: functions that match
        mostly do in the first few tries, and the agent's time is better spent on the next one. Big functions get
        longer (a quarter of Mario Party 6's agent matches over 64 instructions took more than 4 compiles, and
        those functions hold most of the code left)."""
        since, after = self._stall(function_id, agent_id, after)
        if since < after:
            return ""
        return (f"\n\n⏭ No better in your last {since} compiles. release_function with notes (what you tried, what's "
                "left) and claim the next one: functions that match mostly do in the first few tries, and this one "
                "keeps your best attempt for whoever comes next.")

    def _stall(self, function_id: int, agent_id: int, after: int | None = None) -> tuple[int, int]:
        """(compiles since this agent's last better score on the function, compiles before the nudge to move on).
        The nudge comes later for bigger functions; HARD_STOP times it, the agent must release the function."""
        if after is None:
            row = self.store.one("SELECT n_insns FROM functions WHERE id=?", function_id)
            n = (row["n_insns"] if row else 0) or 0
            after = 4 if n < 64 else 6 if n < 128 else 8 if n < 256 else 10
        # the last 32 compiles: enough to see the hard stop for the biggest functions (HARD_STOP x 10)
        rows = self.store.all("SELECT score FROM attempts WHERE function_id=? AND agent_id=? ORDER BY id DESC LIMIT 32",
                              function_id, agent_id)
        best, since = None, 0
        for r in reversed(rows):
            if r["score"] is not None and (best is None or r["score"] < best):
                best, since = r["score"], 0
            else:
                since += 1
        return since, after

    def _record(self, fn, agent: Agent, score: int | None, source: str, ctx_version: int, matched: bool = False,
                compiled: bool | None = None, via: str = "agent"):
        now = time.time()
        compiled = score is not None if compiled is None else compiled
        with self.store.tx() as db:
            db.execute("INSERT INTO attempts(function_id,agent_id,score,compiled,context_version,source,created_at,via) "
                       "VALUES(?,?,?,?,?,?,?,?)", (fn["id"], agent.id, score, compiled, ctx_version, source, now, via))
            db.execute("UPDATE functions SET attempts=attempts+1 WHERE id=?", (fn["id"],))
            if score is not None:
                db.execute("UPDATE functions SET best_score=?, best_source=?, best_agent_id=?, best_at=?, stale_visits=0 "
                           "WHERE id=? AND status='open' AND (best_score IS NULL OR best_score > ?)",
                           (score, source, agent.id, now, fn["id"], score))
            if matched:
                db.execute("UPDATE functions SET status='matched', matched_source=?, matched_agent_id=?, "
                           "matched_at=?, best_score=0, best_source=? WHERE id=?",
                           (source, agent.id, now, source, fn["id"]))
                db.execute("DELETE FROM claims WHERE function_id=?", (fn["id"],))
                self._event(db, fn["project_id"], agent.id, "match", fn["name"])
        if matched:
            self.on_match(self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"]), fn["id"], agent)

    def on_match(self, proj, function_id: int, agent: Agent | None = None) -> None:
        """A function just matched (subclasses try its code on its twins)."""

    # -------------------------------------------------------- shared context

    def edit_context(self, agent: Agent, project: str | None, old: str, new: str, message: str,
                     function: str | None = None) -> str:
        """Exact-string edit of a header. Accepted only if every already-matched function compiled against
        that header still matches with the new one — that's the project's test suite."""
        self._require_level(agent, "Editing the shared header")
        if function:
            fn = self._function(function, project)
            proj = self.store.one("SELECT * FROM projects WHERE id=?", fn["project_id"])
            unit = self._effective_unit(fn, self._unit_contexts(proj["id"]))
        else:
            proj = self._project(project)
            if self._unit_contexts(proj["id"]):
                raise NetworkError(f"{proj['slug']} has one header per source file: pass `function` to edit the "
                                   f"header that function compiles against")
            unit = ""
        if not message.strip():
            raise NetworkError("give a short commit message explaining the change")
        for _ in range(3):
            ctx = self._context(proj["id"], unit)
            header = ctx["header"]
            if old:
                from .textedit import replace_once

                new_header = replace_once(header, old, new, "the header")
            else:
                new_header = header.rstrip("\n") + "\n\n" + new.strip("\n") + "\n"
            if (pasted := self._new_asm_macros(header, new_header)):
                return pasted
            members = self._matched_using(proj, unit)
            broken = self.regression_test(proj, new_header, only=members)
            already: set[int] = set()
            if broken:  # only what the edit itself breaks counts (see already_failing)
                already = self.already_failing(proj, members)
                names = {m["id"]: m["name"] for m in members}
                broken = [(n, why) for n, why in broken if n not in {names[i] for i in already}]
            stale = self._already_note(proj, members, already)
            if broken:
                listing = "\n".join(f"- {name}: {why}" for name, why in broken[:10])
                return (f"❌ REJECTED: the edit breaks {len(broken)} already-matched function(s), so the header "
                        f"stays at v{ctx['version']}.\n{listing}{stale}")
            with self.store.tx() as db:
                if self._latest_version(db, proj["id"], unit) != ctx["version"]:
                    continue  # someone else committed meanwhile: re-apply on top and re-test
                version = self._next_version(db, proj["id"])
                db.execute("INSERT INTO context_versions(project_id,version,header,agent_id,message,created_at,unit) "
                           "VALUES(?,?,?,?,?,?,?)", (proj["id"], version, new_header, agent.id, message.strip(), time.time(), unit))
                self._event(db, proj["id"], agent.id, "context", f"v{version}{f' ({unit})' if unit else ''}: {message.strip()}")
            return (f"✅ header is now v{version} (all {len(members) - len(already)} matched functions that use it and "
                    f"matched with the old header still match).{stale}")
        raise NetworkError("the header is being edited heavily right now; re-read it and retry")

    def rename_symbol(self, agent: Agent, project: str | None, old: str, new: str, message: str = "") -> str:
        """Project-wide refactor: rename an identifier (field, type, global, function param...) in every
        header *and* in every matched source / best attempt, then regression-test the result.
        The rename is textual (whole words), so it is meant for unique names like `unk_1C` or `func_80012345`."""
        self._require_level(agent, "Renaming across the project")
        proj = self._project(project)
        ident = re.compile(r"^[A-Za-z_]\w*$")
        if not (ident.match(old) and ident.match(new)):
            raise NetworkError("old and new must be C identifiers")
        word = re.compile(rf"\b{re.escape(old)}\b")
        for _ in range(3):
            units = self._unit_contexts(proj["id"])
            current = {u: self._context(proj["id"], u) for u in ({""} | units)}
            changed = {u: word.sub(new, c["header"]) for u, c in current.items() if word.search(c["header"])}
            if not changed:
                raise NetworkError(f"`{old}` does not occur in any header")
            for u, h in changed.items():
                if (pasted := self._new_asm_macros(current[u]["header"], h)):
                    return pasted
            matched = self.store.all("SELECT * FROM functions WHERE project_id=? AND status='matched'", proj["id"])
            rewritten = {r["id"]: word.sub(new, r["matched_source"]) for r in matched}
            header_for = lambda fn: changed.get(self._effective_unit(fn, units), current[self._effective_unit(fn, units)]["header"])
            broken = self.regression_test(proj, None, sources=rewritten, only=matched, header_for=header_for)
            if broken:  # only what the rename itself breaks counts (see already_failing)
                already = self.already_failing(proj, matched)
                gone = {r["name"] for r in matched if r["id"] in already}
                broken = [(n, why) for n, why in broken if n not in gone]
            if broken:
                listing = "\n".join(f"- {name}: {why}" for name, why in broken[:10])
                return f"❌ REJECTED: renaming breaks {len(broken)} matched function(s).\n{listing}"
            msg = message.strip() or f"rename {old} -> {new}"
            with self.store.tx() as db:
                n_matched = db.execute("SELECT COUNT(*) AS n FROM functions WHERE project_id=? AND status='matched'",
                                       (proj["id"],)).fetchone()["n"]
                if n_matched != len(matched) or any(self._latest_version(db, proj["id"], u) != current[u]["version"]
                                                    for u in changed):
                    continue
                for u, header in changed.items():
                    version = self._next_version(db, proj["id"])
                    db.execute("INSERT INTO context_versions(project_id,version,header,agent_id,message,created_at,unit) "
                               "VALUES(?,?,?,?,?,?,?)", (proj["id"], version, header, agent.id, msg, time.time(), u))
                for fid, src in rewritten.items():
                    db.execute("UPDATE functions SET matched_source=?, best_source=? WHERE id=?", (src, src, fid))
                for r in db.execute("SELECT id, best_source FROM functions WHERE project_id=? AND status='open' "
                                    "AND best_source IS NOT NULL", (proj["id"],)).fetchall():
                    db.execute("UPDATE functions SET best_source=? WHERE id=?", (word.sub(new, r["best_source"]), r["id"]))
                self._event(db, proj["id"], agent.id, "context", f"{msg} ({len(changed)} header(s))")
            return (f"✅ renamed `{old}` -> `{new}` in {len(changed)} header(s) and in {len(rewritten)} matched "
                    f"sources; all still match.")
        raise NetworkError("headers are being edited heavily right now; retry")

    def _latest_version(self, db, project_id: int, unit: str) -> int | None:
        row = db.execute("SELECT MAX(version) AS v FROM context_versions WHERE project_id=? AND unit=?",
                         (project_id, unit)).fetchone()
        return row["v"]

    def _next_version(self, db, project_id: int) -> int:
        return (db.execute("SELECT MAX(version) AS v FROM context_versions WHERE project_id=?",
                           (project_id,)).fetchone()["v"] or 0) + 1

    def _matched_using(self, proj, unit: str) -> list:
        units = self._unit_contexts(proj["id"])
        return [f for f in self.store.all("SELECT * FROM functions WHERE project_id=? AND status='matched'", proj["id"])
                if self._effective_unit(f, units) == unit]

    @staticmethod
    def _new_asm_macros(old_header: str, new_header: str) -> str:
        """A rejection when an edit adds a macro that pastes in assembly (compiler.header_asm_problems), else "".
        Ones already in the header aren't blamed on the edit (and sources can't use them: compiler.check_header_use)."""
        before = header_asm_problems(old_header)
        new = {n: why for n, why in header_asm_problems(new_header).items() if n not in before}
        if not new:
            return ""
        listing = "\n".join(f"- {n}: {why}" for n, why in new.items())
        return ("❌ REJECTED: context.h macros may emit GTE (coprocessor 2) instructions and pin registers, nothing else. "
                "A macro that pastes in the original's assembly isn't decompilation, and matches made with it don't "
                f"count.\n{listing}")

    def already_failing(self, proj, members: list | None = None) -> set[int]:
        """Ids of matched functions that don't reproduce with the headers as they are now: a declaration that changed
        since they matched, or a toolchain change. A header edit isn't blamed for them. Without this, one such function
        blocks every edit, even a comment, and the header freezes. Cached until a header changes or something new
        matches."""
        if members is None:
            members = self.store.all("SELECT * FROM functions WHERE project_id=? AND status='matched'", proj["id"])
        units = self._unit_contexts(proj["id"])
        versions = tuple(sorted((u, (self._context(proj["id"], u) or {"version": None})["version"])
                                for u in ({""} | units)))
        key = (proj["id"], versions, tuple(sorted(m["id"] for m in members)))
        cache = self.__dict__.setdefault("_failing_cache", {})
        if key not in cache:
            if len(cache) > 16:
                cache.clear()

            def current(fn):
                ctx = self._context(proj["id"], self._effective_unit(fn, units))
                return ctx["header"] if ctx else ""

            by_name = {m["name"]: m["id"] for m in members}
            cache[key] = {by_name[n] for n, _ in self.regression_test(proj, None, only=members, header_for=current)
                          if n in by_name}
        return cache[key]

    def _already_note(self, proj, members: list, already: set[int]) -> str:
        if not already:
            return ""
        names = sorted(m["name"] for m in members if m["id"] in already)
        return (f"\n(Not counted: {len(names)} matched function(s) already fail with the current header: "
                f"{', '.join(names[:12])}{' ...' if len(names) > 12 else ''}. Fix those on their own.)")

    def regression_test(self, proj, header: str | None, sources: dict[int, str] | None = None,
                        only: list | None = None, header_for=None) -> list[tuple[str, str]]:
        matched = only if only is not None else self.store.all(
            "SELECT * FROM functions WHERE project_id=? AND status='matched'", proj["id"])

        def check(fn) -> tuple[str, str] | None:
            res = self._compile(proj, header_for(fn) if header_for else header,
                                (sources or {}).get(fn["id"], fn["matched_source"]))
            if not res.ok:
                return fn["name"], "no longer compiles: " + res.log.strip().splitlines()[0][:200] if res.log.strip() else "compile failed"
            cand, verdict, _ = self._evaluate(fn, proj, res.obj)
            if verdict is None and cand is not None:
                want = fn["fingerprint"] if self._hash_only(fn) else self._target_obj(fn, proj).fingerprint()
                verdict = cand.fingerprint() == want
            if cand is None or not verdict:
                return fn["name"], "compiles but no longer matches"
            return None

        if not matched:  # nothing to protect, but the header must at least parse
            if header is None:
                return []
            res = self._compile(proj, header, "")
            return [] if res.ok else [("<header>", res.log.strip()[:300])]
        return [b for b in self._pool.map(check, matched) if b]

    # ------------------------------------------------------------------ notes

    def add_note(self, agent: Agent, project: str | None, text: str, function: str | None = None) -> str:
        proj = self._project(project)
        fn = self._function(function, proj["slug"]) if function else None
        if not text.strip():
            raise NetworkError("empty note")
        with self.store.tx() as db:
            db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,?)",
                       (proj["id"], fn["id"] if fn else None, agent.id, text.strip()[:4000], time.time()))
        return "Noted. Other agents will see it" + (f" when they open {fn['name']}." if fn else " in their briefings.")

    def search_notes(self, project: str | None, query: str) -> str:
        proj = self._project(project)
        rows = self.store.all(
            "SELECT n.text, n.created_at, a.name AS agent, f.name AS fn FROM notes n "
            "LEFT JOIN agents a ON a.id=n.agent_id LEFT JOIN functions f ON f.id=n.function_id "
            "WHERE n.project_id=? AND n.text LIKE ? ORDER BY n.id DESC LIMIT 20", proj["id"], f"%{query}%")
        if not rows:
            return f"No notes matching {query!r}."
        return "\n".join(f"- [{r['fn'] or 'project'}] ({r['agent']}, {_ago(r['created_at'])}) {r['text']}" for r in rows)

    # ----------------------------------------------------------------- status

    def status_json(self) -> dict:
        with self.store.tx() as db:
            self._expire_claims(db)
        projects = []
        for p in self.store.all("SELECT * FROM projects ORDER BY slug"):
            agg = self.store.one(
                "SELECT COUNT(*) AS total, SUM(status='matched') AS matched, SUM(size) AS bytes, "
                "SUM(CASE WHEN status='matched' THEN size ELSE 0 END) AS matched_bytes, SUM(attempts) AS attempts "
                "FROM functions WHERE project_id=?", p["id"])
            claims = self.store.all(
                "SELECT f.name, a.name AS agent, c.claimed_at, f.best_score, f.n_insns FROM claims c "
                "JOIN functions f ON f.id=c.function_id JOIN agents a ON a.id=c.agent_id "
                "WHERE f.project_id=? ORDER BY c.claimed_at", p["id"])
            ctx = self._context(p["id"])
            claimed = {c["name"] for c in claims}
            funcs = [{"name": f["name"], "insns": f["n_insns"], "best": f["best_score"],
                      "status": "matched" if f["status"] == "matched" else "claimed" if f["name"] in claimed else "open"}
                     for f in self.store.all("SELECT name, n_insns, status, best_score FROM functions "
                                             "WHERE project_id=? ORDER BY name", p["id"])]
            legal = json.loads(p["config"]).get("legal", {})
            projects.append({
                "legal": {"basis": legal.get("basis"), "license": legal.get("license")},
                "funcs": funcs,
                "slug": p["slug"], "name": p["name"], "arch": p["arch"], "description": p["description"],
                "functions": agg["total"], "matched": agg["matched"] or 0, "bytes": agg["bytes"] or 0,
                "matched_bytes": agg["matched_bytes"] or 0, "attempts": agg["attempts"] or 0,
                "context_version": ctx["version"] if ctx else 0,
                "claims": [{"function": c["name"], "agent": c["agent"], "since": c["claimed_at"],
                            "best_score": c["best_score"], "insns": c["n_insns"]} for c in claims],
            })
        leaders = self.store.all(
            "SELECT a.name, a.owner, COUNT(f.id) AS matches, COALESCE(SUM(f.n_insns),0) AS insns, a.last_seen "
            "FROM agents a LEFT JOIN functions f ON f.matched_agent_id=a.id "
            "GROUP BY a.id ORDER BY insns DESC, matches DESC LIMIT 25")
        events = self.store.all(
            "SELECT e.kind, e.detail, e.created_at, a.name AS agent, p.slug FROM events e "
            "LEFT JOIN agents a ON a.id=e.agent_id LEFT JOIN projects p ON p.id=e.project_id "
            "ORDER BY e.id DESC LIMIT 30")
        return {
            "projects": projects,
            "leaderboard": [dict(r) for r in leaders],
            "events": [dict(r) for r in events],
        }

    def status(self, agent: Agent | None = None) -> str:
        s = self.status_json()
        out = []
        for p in s["projects"]:
            pct = 100 * p["matched_bytes"] / (p["bytes"] or 1)
            out.append(f"{p['slug']}: {p['matched']}/{p['functions']} functions, {pct:.1f}% of code matched, "
                       f"context.h v{p['context_version']}, {len(p['claims'])} agents working")
            out += [f"  - {c['agent']} on {c['function']} ({c['insns']} insns, best {c['best_score']})"
                    for c in p["claims"]]
        out.append("Leaderboard (instructions matched):")
        out += [f"  {i + 1}. {r['name']}{' (you)' if agent and r['name'] == agent.name else ''}: "
                f"{r['matches']} functions / {r['insns']} insns" for i, r in enumerate(s["leaderboard"][:10])]
        return "\n".join(out)

    # ------------------------------------------------------------- moderation
    # Operator-only actions (CLI), for takedown requests and GDPR erasure requests.

    def remove_project(self, slug: str) -> int:
        proj = self._project(slug)
        with self.store.tx() as db:
            ids = [r["id"] for r in db.execute("SELECT id FROM functions WHERE project_id=?", (proj["id"],))]
            for table in ("claims", "attempts"):
                db.executemany(f"DELETE FROM {table} WHERE function_id=?", [(i,) for i in ids])
            for table in ("notes", "events", "context_versions", "functions", "symbols", "twin_misses"):
                db.execute(f"DELETE FROM {table} WHERE project_id=?", (proj["id"],))
            db.execute("DELETE FROM projects WHERE id=?", (proj["id"],))
        return len(ids)

    def reset_function(self, slug: str, name: str) -> None:
        """Remove every submission for one function (matched source, attempts, notes) and reopen it."""
        fn = self._function(name, slug)
        with self.store.tx() as db:
            for table in ("claims", "attempts", "notes"):
                db.execute(f"DELETE FROM {table} WHERE function_id=?", (fn["id"],))
            db.execute("UPDATE functions SET status='open', best_score=NULL, best_source=NULL, best_agent_id=NULL, "
                       "matched_source=NULL, matched_agent_id=NULL, matched_at=NULL, attempts=0, abandons=0 "
                       "WHERE id=?", (fn["id"],))

    def revoke_agent(self, name: str, erase: bool = False) -> None:
        """Disable an agent's token. With erase=True also strip its identity (GDPR erasure):
        the agent is renamed to an anonymous id and the operator name is removed."""
        row = self.store.one("SELECT id FROM agents WHERE name=?", name)
        if row is None:
            raise NetworkError(f"no agent named {name!r}")
        with self.store.tx() as db:
            db.execute("DELETE FROM claims WHERE agent_id=?", (row["id"],))
            db.execute("UPDATE agents SET token_hash=? WHERE id=?", ("revoked:" + secrets.token_hex(16), row["id"]))
            if erase:
                db.execute("UPDATE agents SET name=?, owner=NULL WHERE id=?", (f"erased-{row['id']}", row["id"]))

    def list_notes(self, slug: str) -> list[dict]:
        proj = self._project(slug)
        return [dict(r) for r in self.store.all(
            "SELECT n.id, f.name AS function, a.name AS agent, n.text FROM notes n LEFT JOIN functions f "
            "ON f.id=n.function_id LEFT JOIN agents a ON a.id=n.agent_id WHERE n.project_id=? ORDER BY n.id",
            proj["id"])]

    def learnings(self, slug: str, budget: int = 2500) -> str:
        """The project-wide notes agents left (newest first, within `budget` characters), for every new session:
        what one agent works out about the compiler or the project, the next one starts with."""
        proj = self._project(slug)
        out, used = [], 0
        for r in self.store.all("SELECT text FROM notes WHERE project_id=? AND function_id IS NULL ORDER BY id DESC LIMIT 40",
                                proj["id"]):
            text = " ".join(r["text"].split())[:600]
            if used + len(text) > budget:
                break
            out.append(f"- {text}")
            used += len(text)
        return "\n".join(out)

    def delete_note(self, note_id: int) -> None:
        with self.store.tx() as db:
            if db.execute("DELETE FROM notes WHERE id=?", (note_id,)).rowcount == 0:
                raise NetworkError(f"no note with id {note_id}")

    # ----------------------------------------------------------------- export

    def export(self, project: str, out_dir: Path) -> int:
        """Write matched sources + the current context as a tree you can PR upstream."""
        proj = self._project(project)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "context.h").write_text(self._context(proj["id"])["header"], encoding="utf-8")
        for unit in sorted(self._unit_contexts(proj["id"])):
            (out_dir / "contexts").mkdir(exist_ok=True)
            (out_dir / "contexts" / f"{unit}.h").write_text(self._context(proj["id"], unit)["header"], encoding="utf-8")
        rows = self.store.all(
            "SELECT f.name, f.matched_source, a.name AS agent, a.owner FROM functions f "
            "LEFT JOIN agents a ON a.id=f.matched_agent_id WHERE f.project_id=? AND f.status='matched' "
            "ORDER BY f.name", proj["id"])
        (out_dir / "src").mkdir(exist_ok=True)
        credits = []
        for r in rows:
            (out_dir / "src" / f"{r['name']}.c").write_text(
                f'#include "context.h"\n\n// matched by {r["agent"]}' + (f" (operator: {r['owner']})" if r["owner"] else "")
                + f"\n{r['matched_source'].strip()}\n", encoding="utf-8")
            credits.append(f"{r['name']}\t{r['agent']}\t{r['owner'] or ''}")
        (out_dir / "CREDITS.tsv").write_text("function\tagent\toperator\n" + "\n".join(credits) + "\n",
                                             encoding="utf-8")
        return len(rows)
