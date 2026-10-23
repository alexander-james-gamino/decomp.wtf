"""Which function an agent gets next, now that model tiers are gone: a weighted random pick, away from functions many
agents visited without progress; never what was left for a human; the permuter's near misses only when nothing else is
left (or for a helper started on them); an optional size limit per helper. Plus the permuter's two rounds, the stall
stop for everyone, the overlay-aware reader and the exported symbol list."""

import json
import random
import time

import pytest

from decompnet import network
from decompnet.app.engine import RomNetwork
from decompnet.network import NetworkError
from decompnet.rom import split
from decompnet.scoring import DiffResult, diff_kind


def _project(net, root, rows, cfg=None):
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(root), json.dumps({"kind": "rom", **(cfg or {})}))).lastrowid
        for r in rows:
            cols = {"project_id": pid, "status": "open", "unit": "a", **r}
            cols.setdefault("size", cols["n_insns"] * 4)
            db.execute(f"INSERT INTO functions({','.join(cols)}) VALUES({','.join('?' * len(cols))})", tuple(cols.values()))
    return pid


def _agent(net, name, limit=None):
    agent, _ = net.local_agent(name)
    net.set_agent_limit(agent.id, limit)
    return agent


def _held(net, agent):
    return net.store.one("SELECT f.name FROM claims c JOIN functions f ON f.id=c.function_id WHERE c.agent_id=?",
                         agent.id)["name"]


# ---------------------------------------------------------------------------------------- the weighted pick

def test_functions_visited_again_and_again_without_progress_sink_but_never_vanish():
    now = time.time()
    row = lambda **k: {"name": "f", "stale_visits": 0, "best_at": None, "boost": 0, "n_insns": 40,  # noqa: E731
                       "best_score": None, **k}
    w = network.Network._weight
    assert w(row(), now, set()) == 1.0
    assert w(row(stale_visits=4), now, set()) == pytest.approx(0.25)
    assert w(row(stale_visits=12), now, set()) == pytest.approx(1 / 16)
    assert 0 < w(row(stale_visits=40), now, set()) < 0.01  # sinks, but stays possible
    assert w(row(best_at=now - 3600), now, set()) == 2.0  # improved within the day
    assert w(row(boost=2), now, set()) == 3.0 and w(row(boost=-1), now, set()) == 0.25  # twins
    assert w(row(), now, {"f"}) == pytest.approx(0.05)  # just released by someone
    assert w(row(best_score=40), now, set(), closest=True) > w(row(best_score=400), now, set(), closest=True)


def test_the_pick_is_random_and_mostly_spares_stuck_functions(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "fresh", "n_insns": 40}, {"name": "stuck", "n_insns": 4, "stale_visits": 40}])
    net._rng = random.Random(7)
    proj = net.project_row("game")
    rows = net.store.all("SELECT * FROM functions")
    picks = [net._pick(net.store.conn, proj, rows, time.time())["name"] for _ in range(400)]
    assert picks.count("fresh") > 380 and "stuck" in picks  # the smallest isn't first any more; stuck still comes


def test_shuffle_is_the_default_and_hands_out_everything_in_time(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": f"f{i}", "n_insns": 10 + i} for i in range(6)])
    assert net.default_strategy(None, "game") == "shuffle"
    got = {net.claim_row(_agent(net, f"a{i}"), "game")["name"] for i in range(6)}
    assert got == {f"f{i}" for i in range(6)}


# ---------------------------------------------------------------------------------------- what stays out

def test_human_handoffs_never_come_and_the_permuters_near_misses_come_last(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [
        {"name": "plain", "n_insns": 30},
        {"name": "regs", "n_insns": 30, "best_score": 20, "best_source": "int a;", "best_kind": "regalloc"},
        {"name": "for_you", "n_insns": 30, "handoff": "human", "handoff_note": "a stack switch"},
    ])
    first, second, third = (_agent(net, n) for n in ("a", "b", "c"))
    assert net.claim_row(first, "game")["name"] == "plain"
    assert net.claim_row(second, "game")["name"] == "regs"  # nothing else left: the permuter's one after all
    assert net.claim_row(third, "game") is None  # a human's, never automatically
    assert net.claim_row(third, "game", "for_you")["name"] == "for_you"  # by name: yours, and the handoff is gone
    assert net.store.one("SELECT handoff FROM functions WHERE name='for_you'")["handoff"] is None


def test_a_helper_started_on_the_permuters_near_misses_gets_only_those(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [
        {"name": "plain", "n_insns": 30},
        {"name": "near", "n_insns": 30, "best_score": 20, "best_source": "int a;", "best_kind": "regalloc"},
        {"name": "far", "n_insns": 30, "best_score": 900, "best_source": "int b;", "best_kind": "regalloc"},
    ])
    net._rng = random.Random(3)
    pins = _agent(net, "pins")
    assert net.claim_row(pins, "game", None, "order-only")["name"] in ("near", "far")
    other = _agent(net, "pins-2")
    assert net.claim_row(other, "game", None, "order-only")["name"] in ("near", "far")
    assert net.claim_row(_agent(net, "pins-3"), "game", None, "order-only") is None  # never "plain"


def test_a_size_limit_keeps_a_helper_to_small_functions(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "small", "n_insns": 30}, {"name": "big", "n_insns": 400}])
    weak = _agent(net, "local", 80)
    assert net.agent_limit(weak) == 80 and net.left_for(weak, "game") == 1
    assert net.claim_row(weak, "game")["name"] == "small"
    assert net.claim_row(_agent(net, "weak-2", 80), "game") is None
    assert net.claim_row(_agent(net, "strong"), "game")["name"] == "big"


# ---------------------------------------------------------------------------------------- visits without progress

def test_visits_without_progress_are_counted_and_progress_resets_them(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "f", "n_insns": 30}])
    a = _agent(net, "a")
    stale = lambda: net.store.one("SELECT stale_visits, best_at FROM functions WHERE name='f'")  # noqa: E731
    for _ in range(2):
        net.claim_row(a, "game", "f")
        net.release(a, "f", "", "game")
    assert stale()["stale_visits"] == 2
    fn = net.claim_row(a, "game", "f")
    net._record(fn, a, 500, "int f;", 1)  # a better score during this claim
    assert stale()["stale_visits"] == 0 and stale()["best_at"]
    net.release(a, "f", "", "game")
    assert stale()["stale_visits"] == 0  # that visit made progress
    net.claim_row(a, "game", "f")
    with net.store.tx() as db:
        db.execute("UPDATE claims SET expires_at=0")
        net._expire_claims(db)  # a lease that ran out is a visit too
    assert stale()["stale_visits"] == 1


def test_visits_are_counted_from_history_once(tmp_path):
    from decompnet.store import Store

    net = RomNetwork(tmp_path / "app.db", tmp_path)
    pid = _project(net, tmp_path, [{"name": "f", "n_insns": 30, "best_score": 100}, {"name": "g", "n_insns": 30}])
    a = _agent(net, "a")
    fid = net._function("f", "game")["id"]
    with net.store.tx() as db:
        for t, score in [(10, 300), (20, 100), (30, 100)]:
            db.execute("INSERT INTO attempts(function_id,agent_id,score,compiled,created_at) VALUES(?,?,?,1,?)",
                       (fid, a.id, score, t))
        for t, name in [(15, "f"), (25, "f"), (35, "f"), (40, "g"), (41, "g")]:
            db.execute("INSERT INTO events(project_id,agent_id,kind,detail,created_at) VALUES(?,?,?,?,?)",
                       (pid, a.id, "release" if t != 41 else "expired", name, t))
        db.execute("UPDATE functions SET best_at=NULL, stale_visits=0")
        db.execute("DELETE FROM settings WHERE key='stale_visits_counted'")  # as in a database from before
    store = Store(tmp_path / "app.db")
    rows = {r["name"]: (r["best_at"], r["stale_visits"]) for r in store.all("SELECT * FROM functions")}
    assert rows == {"f": (20, 2), "g": (None, 2)}  # f reached 100 at 20, then two releases; g never improved


# ---------------------------------------------------------------------------------------- the permuter

def test_left_for_the_permuter_and_its_two_rounds(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    pid = _project(net, tmp_path, [
        {"name": "regs", "n_insns": 40, "best_score": 25, "best_source": "int a;", "best_kind": "regalloc"},
        {"name": "said_so", "n_insns": 40, "best_score": 90, "best_source": "int b;", "best_kind": "structure"},
        {"name": "asked", "n_insns": 40, "best_score": 100, "best_source": "int c;", "best_kind": "structure"},
        {"name": "too_far", "n_insns": 40, "best_score": 900, "best_source": "int f;", "best_kind": "structure"},
        {"name": "close", "n_insns": 100, "best_score": 50, "best_source": "int d;", "best_kind": "structure"},
        {"name": "data_only", "n_insns": 40, "best_score": 1, "best_source": "int e;", "best_kind": "data"},
        {"name": "fresh", "n_insns": 12},
    ])
    agent = _agent(net, "claude-1")
    for name, notes, leave in [("regs", "tried two orders", ""), ("said_so", "regalloc only: permuter", ""),
                               ("asked", "registers left", "permuter"), ("data_only", "jtbl reads wrong", "human")]:
        net.claim_row(agent, "game", name)
        assert "Left for " in net.release(agent, name, notes, "game", leave)
    net.claim_row(agent, "game", "too_far")
    assert "Not left for the permuter" in net.release(agent, "too_far", "permuter only, I give up", "game", "permuter")
    hand = {r["name"]: r["handoff"] for r in net.store.all("SELECT name, handoff FROM functions WHERE project_id=?", pid)}
    assert hand == {"regs": "permuter", "said_so": "permuter", "asked": "permuter", "close": None,
                    "data_only": "human", "fresh": None, "too_far": None}
    other = _agent(net, "claude-2")
    assert net.claim_row(other, "game", None, "closest")["name"] == "close"  # handed-off ones never come first
    net.release(other, "close", "", "game")
    assert net.claim_row(other, "game", None, "closest")["name"] in ("too_far", "fresh")
    net.release(other, _held(net, other), "", "game")
    proj = net.project_row("game")
    queue = [r["name"] for r in net.permuter_queue(proj)]
    assert queue[:3] == ["regs", "said_so", "asked"] and "data_only" not in queue, queue
    permuter, _ = net.local_agent("permuter")
    abandons = net.store.one("SELECT abandons FROM functions WHERE name='regs'")["abandons"]
    for _ in range(network.PERMUTER_TURNS):
        net.permuter_done(permuter, net._function("regs", "game")["id"], "")
    row = net.store.one("SELECT * FROM functions WHERE name='regs'")
    assert row["handoff"] == "permuter" and "regs" in [r["name"] for r in net.permuter_queue(proj)]  # a second round
    for _ in range(network.PERMUTER_RUNS - network.PERMUTER_TURNS):
        net.permuter_done(permuter, net._function("regs", "game")["id"], "")
    row = net.store.one("SELECT * FROM functions WHERE name='regs'")
    assert row["handoff"] is None and row["permuter_turns"] == network.PERMUTER_RUNS  # back to the agents
    assert row["abandons"] == abandons  # a permuter run isn't an agent giving up
    assert "regs" not in [r["name"] for r in net.permuter_queue(proj)]


def test_handoffs_never_strand_a_function(tmp_path):
    """The permuter had both its rounds: a note saying "permuter" doesn't send it back there (nobody would take it)."""
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [
        {"name": "done_permuting", "n_insns": 30, "best_score": 40, "best_source": "int a;", "best_kind": "regalloc",
         "permuter_turns": network.PERMUTER_RUNS},
    ])
    a = _agent(net, "opus-1")
    net.claim_row(a, "game", "done_permuting")
    assert "Left for" not in net.release(a, "done_permuting", "regalloc only: permuter", "game")
    assert net.store.one("SELECT handoff FROM functions WHERE name='done_permuting'")["handoff"] is None


def test_notes_from_before_handoffs_hand_their_functions_to_the_permuter_once(tmp_path):
    from decompnet.store import Store

    net = RomNetwork(tmp_path / "app.db", tmp_path)
    pid = _project(net, tmp_path, [{"name": "said_permuter", "n_insns": 40, "best_score": 100, "best_source": "int a;"},
                                   {"name": "far_but_said", "n_insns": 40, "best_score": 900, "best_source": "int c;"},
                                   {"name": "no_attempt", "n_insns": 40},
                                   {"name": "other_note", "n_insns": 40, "best_score": 90, "best_source": "int b;"}])
    with net.store.tx() as db:
        db.execute("UPDATE functions SET handoff=NULL")
        db.execute("DELETE FROM settings WHERE key='handoffs_from_notes'")  # as in a database from before
        for name, text in [("said_permuter", "Best 100 stays; regalloc only: permuter. Queue keeps re-serving this."),
                           ("far_but_said", "regalloc only: permuter (but still 77% there)"),
                           ("no_attempt", "permuter only"), ("other_note", "tried a for loop instead")]:
            fid = net._function(name, "game")["id"]
            db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,0)",
                       (pid, fid, None, text))
    store = Store(tmp_path / "app.db")
    hand = {r["name"]: r["handoff"] for r in store.all("SELECT name, handoff FROM functions")}
    assert hand == {"said_permuter": "permuter", "far_but_said": None, "no_attempt": None, "other_note": None}


# ---------------------------------------------------------------------------------------- the priority list, the stall stop

def test_the_priority_list_never_hands_back_a_function_just_released(tmp_path, monkeypatch):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "stuck", "n_insns": 30}, {"name": "other", "n_insns": 50},
                             {"name": "third", "n_insns": 60}], {"priority": ["stuck"]})
    a = _agent(net, "claude-1")
    assert net.claim_row(a, "game", None, "easiest")["name"] == "stuck"
    net.release(a, "stuck", "permuter only; regalloc", "game")  # (not left for the permuter: no best attempt)
    assert net.claim_row(a, "game", None, "easiest")["name"] == "other"  # not the one it just released
    b = _agent(net, "claude-2")
    assert net.claim_row(b, "game", None, "easiest")["name"] == "third"  # released by someone just now: cooling
    monkeypatch.setattr(network, "RELEASE_COOLDOWN", 0)
    net.release(b, "third", "", "game")
    c = _agent(net, "claude-3")
    assert net.claim_row(c, "game", None, "easiest")["name"] == "stuck"  # cooled down: the priority again
    net.release(c, "stuck", "", "game")
    net.set_handoff("game", "stuck", "permuter")
    d = _agent(net, "claude-4")
    assert net.claim_row(d, "game", None, "easiest")["name"] != "stuck"  # left for the permuter: skipped


def test_every_agent_is_stopped_after_three_times_the_nudge(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "f", "n_insns": 10}])
    agent = _agent(net, "sonnet-1")
    fn = net.claim_row(agent, "game", "f")
    with net.store.tx() as db:
        for score in [50] + [60] * 11:
            db.execute("INSERT INTO attempts(function_id,agent_id,score,compiled,created_at) VALUES(?,?,?,1,0)",
                       (fn["id"], agent.id, score))
    since, after = net._stall(fn["id"], agent.id)
    assert (since, after) == (11, 4) and since < network.HARD_STOP * after
    with net.store.tx() as db:
        db.execute("INSERT INTO attempts(function_id,agent_id,score,compiled,created_at) VALUES(?,?,60,1,0)",
                   (fn["id"], agent.id))
    with pytest.raises(NetworkError, match="stop here"):
        net.attempt(agent, "f", "int f(void) { return 0; }", "game")  # refused before compiling anything


# ---------------------------------------------------------------------------------------- unchanged from before

def test_the_reader_never_mixes_up_overlays_at_the_same_address(tmp_path):
    rom = bytes([0x11]) * 0x100 + bytes([0xAA]) * 0x100 + bytes([0xBB]) * 0x100
    (tmp_path / "rom.bin").write_bytes(rom)
    segs = [{"name": "main", "rom_start": 0, "rom_end": 0x100, "vram": 0x80010000},
            {"name": "ovl_a", "rom_start": 0x100, "rom_end": 0x200, "vram": 0x80100000},
            {"name": "ovl_b", "rom_start": 0x200, "rom_end": 0x300, "vram": 0x80100000}]
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "in_b", "n_insns": 4, "vram": 0x80100010, "rom_offset": 0x210},
                             {"name": "in_main", "n_insns": 4, "vram": 0x80010010, "rom_offset": 0x10}],
             {"rom": {"file": "rom.bin"}, "segments": segs})
    proj = net.project_row("game")
    in_b, in_main = net._function("in_b", "game"), net._function("in_main", "game")
    assert net.reader(proj, in_b)(0x80100040, 4) == bytes([0xBB]) * 4  # its own overlay's bytes, not ovl_a's
    assert net.reader(proj, in_b)(0x80010000, 4) == bytes([0x11]) * 4  # the main program is always there
    assert net.reader(proj, in_main)(0x80100040, 4) is None  # which overlay? can't tell: nothing, not a guess
    order = split.visible_segments([split.Segment(**s) for s in segs], 0x150)
    assert [s.name for s in order] == ["ovl_a", "main"]


def test_what_a_diff_says_is_left():
    row = lambda m: (m, "a", "b")  # noqa: E731
    assert diff_kind(DiffResult(10, 100, [row(" "), row("r"), row("o")], False)) == "regalloc"
    assert diff_kind(DiffResult(5, 100, [row(" "), row("s")], False)) == "regalloc"
    assert diff_kind(DiffResult(100, 100, [row("r"), row("<")], False)) == "structure"
    assert diff_kind(DiffResult(1, 100, [row(" ")], False), verdict=False) == "data"
    assert diff_kind(DiffResult(0, 100, [row(" ")], True)) is None


def test_instructions_are_kept_whole(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "f", "n_insns": 10}])
    net.set_instructions("game", "rule. " * 1500)
    assert len(net.instructions("game")) > 4000


def test_the_exported_symbol_list_pins_each_function_to_its_overlay(tmp_path):
    from decompnet.rom import export

    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "func_8009E820", "n_insns": 4, "vram": 0x8009E820, "rom_offset": 0x8D9A8},
                             {"name": "ovl_story_func_8009E820", "n_insns": 4, "vram": 0x8009E820, "rom_offset": 0xACCE8}])
    lines = export._symbol_lines(net, net.project_row("game"))
    assert "func_8009E820 = 0x8009E820; // type:func rom:0x8D9A8" in lines
    assert "ovl_story_func_8009E820 = 0x8009E820; // type:func rom:0xACCE8" in lines


def test_big_function_sections_and_calls():
    """compile_and_diff on a big function says which sections already match and where the calls stop agreeing."""
    from decompnet import scoring

    rows = [(" ", f"addiu $v0, $zero, {i}", f"addiu $v0, $zero, {i}") for i in range(10)]
    rows += [("r", "move $s0, $a0", "move $s1, $a0"), (" ", "jal func_A", "jal func_A"),
             ("|", "jal func_B", "jal func_C"), ("<", "nop", ""), (" ", "jr $ra", "jr $ra")]
    text = scoring.section_report(rows, 0x80001000, [0x80001000 + 4 * 8], min_insns=5)
    assert "0x80001000-0x8000101C" in text and "8 insns: 8 exact, 0 close, 0 wrong" in text
    assert "first difference at 0x80001028" in text  # the `move` row
    assert "calls: target 2, yours 2; the first 1 are the same (next in the target: func_B" in text
