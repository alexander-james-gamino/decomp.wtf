"""The permuter, in the background: the decomp community's way through the last few percent.

A function at 95% usually differs in register allocation or instruction order, which comes down to how the C
is written, not what it does: the order of statements and declarations, a temporary, a cast, `x += 1` for
`x++`, `register`. decomp-permuter tries thousands of such rewrites and keeps what scores better; agents doing
it by hand spend turns (and tokens) on it, and stall. Here it runs on its own, on spare CPU, on the near misses
agents left (or autopilot's drafts reached): it holds a function for a minute at a time, so an agent that
wants it gets it next, records every improvement as the function's best attempt (agents start from it), and
a match as a match. Rewrites keep the meaning of the code (see permuter.py); what matches is still judged
byte for byte, like everything else.

A project can bring its own worker instead (mode "decomp-permuter"): the project config's `permuter.cmd`, or
<project root>/tools/permute/dp_worker.py run by $DECOMP_PERMUTER_PYTHON (default
~/Decomp/tools/decomp-permuter/.venv/bin/python). It runs only from the Permuter button, with the options shown
there, claims each function through this app, submits through the same judge, prints `PROGRESS <json>` lines
(events start/more/function/result/done) and exits when nothing is left; Stop sends it SIGTERM.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import scoring
from ..network import NetworkError
from ..permuter import mutate

log = logging.getLogger("decomp.wtf")
NEAR = 0.06  # a near miss: at most this much of the function differs (best_score <= 6% of n_insns * 100)
SECONDS = 60  # per function per turn
MAX_TURNS = 4  # turns per function before it's left to agents
EXTERNAL = "decomp-permuter"


@dataclass
class State:
    state: str = "running"
    tried: int = 0  # functions permuted
    variants: int = 0
    improved: int = 0
    matched: int = 0
    current: str = ""
    started_at: float = field(default_factory=time.time)
    stop: threading.Event = field(default_factory=threading.Event)
    # an external worker (mode "decomp-permuter")
    mode: str = "built-in"
    total: int = 0
    current_best: int | None = None
    matched_names: list = field(default_factory=list)
    options: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    error: str = ""
    proc: subprocess.Popen | None = None


class Permuter:
    def __init__(self, net):
        self.net = net
        self.jobs: dict[str, State] = {}
        self._turns: dict[int, int] = {}  # function id -> turns spent on it

    def external_cmd(self, slug: str) -> list[str] | None:
        """The project's own permuter worker, if it has one (see the module docstring)."""
        try:
            proj = self.net.project_row(slug)
        except NetworkError:
            return None
        if self.net.fork_slug(slug) or not proj["root"]:
            return None
        cmd = (self.net.cfg(proj).get("permuter") or {}).get("cmd")
        if cmd:
            return [str(c) for c in cmd]
        script = Path(proj["root"]) / "tools/permute/dp_worker.py"
        py = Path(os.environ.get("DECOMP_PERMUTER_PYTHON")
                  or Path.home() / "Decomp/tools/decomp-permuter/.venv/bin/python")
        return [str(py), "-I", str(script)] if script.is_file() and py.is_file() else None

    def status(self, slug: str) -> dict | None:
        st = self.jobs.get(slug)
        if st is None:
            return {"state": None, "mode": EXTERNAL, "available": True} if self.external_cmd(slug) else None
        out = {k: v for k, v in st.__dict__.items() if k not in ("stop", "proc")}
        out["available"] = st.mode == EXTERNAL or self.external_cmd(slug) is not None
        return out

    def start(self, slug: str, options: dict | None = None, manual: bool = False) -> State | None:
        """Run on the project's near misses until there are none (looking again every few minutes while the
        app runs: agents make new ones). A project's own worker runs only when asked for (manual)."""
        if (st := self.jobs.get(slug)) is not None and st.state == "running":
            return st
        if slug in self.net.links:  # a room member: the host's copy permutes, and its matches come back here
            return None
        if (cmd := self.external_cmd(slug)) is not None:
            return self._start_external(slug, cmd, options or {}) if manual else None
        st = self.jobs[slug] = State()
        threading.Thread(target=self._loop, args=(slug, st), daemon=True, name=f"permuter-{slug}").start()
        return st

    def stop(self, slug: str) -> None:
        if (st := self.jobs.get(slug)) is not None:
            st.stop.set()
            if st.proc is not None and st.proc.poll() is None:
                try:
                    os.killpg(st.proc.pid, signal.SIGTERM)  # the worker releases its claim and stops its permuter
                except ProcessLookupError:
                    pass

    # ------------------------------------------------------------------ an external worker (decomp-permuter)

    def _start_external(self, slug: str, cmd: list[str], options: dict) -> State:
        minutes = min(max(float(options.get("minutes") or 10), 0.5), 240)
        max_score = min(max(int(options.get("max_score") or 150), 1), 1_000_000)
        threads = min(max(int(options.get("threads") or 0), 0), 128)
        names = ",".join(re.findall(r"[A-Za-z_]\w*", str(options.get("functions") or "")))
        db = self.net.store.conn.execute("PRAGMA database_list").fetchone()[2]  # the worker uses this app's data
        args = ["--progress", "--db", db, "--minutes", f"{minutes:g}", "--max-score", str(max_score)]
        args += ["--threads", str(threads)] if threads else []
        args += ["--functions", names] if names else []
        st = State(mode=EXTERNAL, options={"minutes": minutes, "threads": threads, "max_score": max_score,
                                           "functions": names})
        st.proc = subprocess.Popen(cmd + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, start_new_session=True)
        self.jobs[slug] = st
        threading.Thread(target=self._read_external, args=(st,), daemon=True, name=f"permuter-{slug}").start()
        return st

    def _read_external(self, st: State) -> None:
        for line in st.proc.stdout:
            line = line.rstrip("\n")
            if not line.startswith("PROGRESS "):
                if line.strip():
                    st.log = (st.log + [line])[-30:]
                continue
            try:
                ev = json.loads(line[9:])
            except ValueError:
                continue
            kind = ev.get("event")
            if kind in ("start", "more"):
                st.total = ev.get("total", st.total)
            elif kind == "function":
                st.current, st.current_best = ev.get("name") or "", ev.get("best")
            elif kind == "result":
                st.tried = ev.get("done", st.tried + 1)
                st.current = ""
                if ev.get("matched"):
                    st.matched += 1
                    st.matched_names = (st.matched_names + [ev.get("name")])[-50:]
        st.proc.wait()
        st.current = ""
        if st.stop.is_set():
            st.state = "stopped"
        elif st.proc.returncode:
            st.state = "failed"
            st.error = st.log[-1] if st.log else f"the worker exited with {st.proc.returncode}"
        else:
            st.state = "done"

    def candidates(self, proj, limit: int = 20) -> list:
        return [r for r in self.net.store.all(
            "SELECT * FROM functions WHERE project_id=? AND status='open' AND best_source IS NOT NULL "
            "AND best_score > 0 AND best_score <= n_insns * 100 * ? AND n_insns <= 400 "
            "AND id NOT IN (SELECT function_id FROM claims) ORDER BY CAST(best_score AS REAL) / (n_insns * 100) LIMIT ?",
            proj["id"], NEAR, limit * 4) if self._turns.get(r["id"], 0) < MAX_TURNS][:limit]

    def _loop(self, slug: str, st: State) -> None:
        agent, _ = self.net.local_agent("permuter")
        idle = 0
        try:
            while not st.stop.is_set():
                try:
                    proj = self.net.project_row(slug)
                except NetworkError:
                    return  # the project is gone
                todo = self.candidates(proj)
                if not todo:
                    idle += 1
                    if idle > 20:  # nothing for an hour: done until agents start again
                        return
                    st.stop.wait(180)
                    continue
                idle = 0
                for fn in todo:
                    if st.stop.is_set():
                        break
                    self._turns[fn["id"]] = self._turns.get(fn["id"], 0) + 1
                    try:
                        self._permute(agent, proj, fn, st)
                    except Exception as e:  # one odd function must not stop the rest
                        log.info("permuter: %s: %s", fn["name"], e)
        finally:
            st.state = "stopped" if st.stop.is_set() else "done"
            st.current = ""

    def _hold(self, agent, fn) -> bool:
        now = time.time()
        with self.net.store.tx() as db:
            self.net._expire_claims(db)
            if db.execute("SELECT 1 FROM claims WHERE function_id=?", (fn["id"],)).fetchone():
                return False
            db.execute("INSERT INTO claims VALUES(?,?,?,?)", (fn["id"], agent.id, now, now + SECONDS + 120))
        return True

    def _let_go(self, agent, fn) -> None:
        with self.net.store.tx() as db:
            db.execute("DELETE FROM claims WHERE function_id=? AND agent_id=?", (fn["id"], agent.id))

    def _permute(self, agent, proj, fn, st: State) -> None:
        if not self._hold(agent, fn):
            return
        st.current, st.tried = fn["name"], st.tried + 1
        fork = self.net.fork_slug(proj["slug"])
        rng = random.Random(f"{fn['id']}-{self._turns.get(fn['id'])}")
        try:
            best_src = cur_src = fn["best_source"]
            best = cur = fn["best_score"]
            deadline = time.monotonic() + SECONDS
            while time.monotonic() < deadline and best > 0 and not st.stop.is_set():
                cand = mutate(cur_src if rng.random() < 0.85 else best_src, rng)
                if cand is None:
                    continue
                st.variants += 1
                score = self._score(agent, proj, fn, cand, best, fork)
                if score is None:
                    continue
                if score < best:
                    best, best_src = score, cand
                    st.improved += 1
                    if score == 0:
                        st.matched += 1
                        log.info("permuter: %s matched", fn["name"])
                        break
                if score <= cur or rng.random() < 0.02:  # mostly downhill, now and then sideways off a plateau
                    cur, cur_src = score, cand
        finally:
            self._let_go(agent, fn)

    def _score(self, agent, proj, fn, source: str, best: int, fork: str | None) -> int | None:
        """A variant's score; recorded (as the function's best attempt, or its match) only when it beats `best`."""
        if fork:
            res = self.net.forks.attempt(agent, fork, fn["name"], source, keep=False, quiet=best)
            return res.get("score")
        ctx = self.net._context(proj["id"], fn["unit"])
        result = self.net._compile(proj, ctx["header"], source)
        if not result.ok:
            return None
        cand, verdict, _ = self.net._evaluate(fn, proj, result.obj)
        if cand is None:
            return None
        d = scoring.diff(self.net._target_obj(fn, proj), cand)
        if verdict is not None:
            d = scoring.DiffResult(0 if verdict else max(1, d.score), d.max_score, d.rows, verdict)
        if d.score < best:
            self.net._record(fn, agent, d.score, source, ctx["version"], matched=d.bytes_match, via="permuter")
        return 0 if d.bytes_match else max(1, d.score)
