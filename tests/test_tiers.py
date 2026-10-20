"""Which agent gets which function (tiers.py): small models small work, large ones what the others can't do, and
whatever only differs in registers goes to the permuter first. Plus the overlay-aware reader and the exported
symbol list, which used to mix up overlays loaded at the same address."""

import json

import pytest

from decompnet import network, tiers
from decompnet.app.engine import RomNetwork
from decompnet.network import NetworkError
from decompnet.rom import split
from decompnet.scoring import DiffResult


def _project(net, root, rows, cfg=None):
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(root), json.dumps({"kind": "rom", **(cfg or {})}))).lastrowid
        for r in rows:
            cols = {"project_id": pid, "status": "open", "unit": "a", **r}
            cols.setdefault("size", cols["n_insns"] * 4)
            db.execute(f"INSERT INTO functions({','.join(cols)}) VALUES({','.join('?' * len(cols))})", tuple(cols.values()))
    return pid


def _agent(net, name, tier):
    agent, _ = net.local_agent(name)
    net.set_agent_tier(agent.id, tier)
    return agent


def test_the_tier_comes_from_the_model():
    lan = lambda base, resolve=False: "192.168." in base  # noqa: E731
    d = lambda kind, **o: tiers.detect(kind, o, lan)  # noqa: E731
    assert d("claude", model="haiku") == d("claude", model="claude-haiku-4-5") == "small"
    assert d("claude", model="sonnet") == "medium"
    assert d("claude", model="opus") == d("claude", model="claude-opus-5-5") == "large"
    assert d("claude") is None  # Claude Code's default model: unknown, no routing
    assert d("opencode", model="mbp/qwen3.8:27b") == "local"  # an Ollama tag behind a provider of the user's
    assert d("opencode", model="ollama/qwen3-coder") == "local"
    assert d("opencode", model="openrouter/qwen/qwen3.8-flash-next") is None
    assert d("api", model="qwen3.8:27b", base_url="http://192.168.1.226:11434/v1") == "local"
    assert d("claude", model="haiku", tier="large") == "large"  # the helper's own setting wins
    assert d("claude", model="haiku", tier="any") is None  # and can switch routing off


def test_each_tier_gets_what_suits_it_and_large_takes_what_small_cant(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [
        {"name": "tiny", "n_insns": 10}, {"name": "small", "n_insns": 60}, {"name": "big", "n_insns": 200},
        {"name": "huge", "n_insns": 600}, {"name": "given_up", "n_insns": 30, "abandons": 3},
        {"name": "near_regalloc", "n_insns": 40, "best_score": 20, "best_source": "int a;", "best_kind": "regalloc"},
        {"name": "left", "n_insns": 20, "handoff": "permuter", "best_score": 5, "best_source": "int b;"},
    ])
    s1, s2, s3 = (_agent(net, f"haiku-{i}", "small") for i in range(3))
    assert net.claim_row(s1, "game", None, "easiest")["name"] == "tiny"
    assert net.claim_row(s2, "game", None, "easiest")["name"] == "small"
    assert net.claim_row(s3, "game", None, "easiest") is None  # the rest is too big, too hard or the permuter's
    big = _agent(net, "opus-1", "large")
    # small helpers are working: the large one starts on what they can't do (given up on 3 times), small first
    assert net.claim_row(big, "game", None, net.default_strategy(big, "game"))["name"] == "given_up"
    mid = _agent(net, "sonnet-1", "medium")
    assert net.claim_row(mid, "game", None, net.default_strategy(mid, "game"))["name"] == "big"
    mid2 = _agent(net, "sonnet-2", "medium")
    assert net.claim_row(mid2, "game", None, "finish") is None  # 600 instructions: a large model is around for it
    big2 = _agent(net, "opus-2", "large")
    assert net.claim_row(big2, "game", None, "frontier")["name"] == "huge"
    big3 = _agent(net, "opus-3", "large")
    assert net.claim_row(big3, "game", None, "frontier")["name"] == "near_regalloc"  # large only, and last


def test_alone_a_large_model_works_as_before_and_a_medium_one_takes_big_work(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "tiny", "n_insns": 10}, {"name": "huge", "n_insns": 600}])
    big = _agent(net, "opus-1", "large")
    assert net.claim_row(big, "game", None, "frontier")["name"] == "tiny"  # nobody weaker: the usual order
    mid = _agent(net, "sonnet-1", "medium")
    assert net.left_for(mid, "game") == 1  # tiny (opus-1 holds it); 600 instructions are for the large model
    assert net.claim_row(mid, "game", None, "finish") is None  # opus-1 is working: the big one waits for it
    net.release(big, "tiny", "", "game")
    with net.store.tx() as db:
        db.execute("DELETE FROM events")  # opus-1 is gone
    assert net.claim_row(mid, "game", None, "finish")["name"] in ("huge", "tiny")


def test_small_models_stop_when_nothing_left_suits_them(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "huge", "n_insns": 600},
                             {"name": "regalloc", "n_insns": 20, "best_score": 10, "best_source": "int a;",
                              "best_kind": "regalloc"}])
    small = _agent(net, "haiku-1", "small")
    assert net.left_for(small, "game") == 0
    assert net.left_for(_agent(net, "opus-1", "large"), "game") == 2
    assert net.left_for(_agent(net, "you", None), "game") == 2


def test_left_for_the_permuter_and_its_queue(tmp_path):
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
    agent = _agent(net, "claude-1", None)
    for name, notes, leave in [("regs", "tried two orders", ""), ("said_so", "regalloc only: permuter", ""),
                               ("asked", "registers left", "permuter"), ("data_only", "jtbl reads wrong", "human")]:
        net.claim_row(agent, "game", name)
        assert "Left for " in net.release(agent, name, notes, "game", leave)
    # 77% there with real instruction differences: the permuter can't fix that, so it stays with the agents
    net.claim_row(agent, "game", "too_far")
    assert "Not left for the permuter" in net.release(agent, "too_far", "permuter only, I give up", "game", "permuter")
    hand = {r["name"]: r["handoff"] for r in net.store.all("SELECT name, handoff FROM functions WHERE project_id=?", pid)}
    assert hand == {"regs": "permuter", "said_so": "permuter", "asked": "permuter", "close": None,
                    "data_only": "human", "fresh": None, "too_far": None}
    other = _agent(net, "claude-2", None)
    assert net.claim_row(other, "game", None, "closest")["name"] == "close"  # handed-off ones never come automatically
    net.release(other, "close", "", "game")
    assert net.claim_row(other, "game", None, "closest")["name"] in ("too_far", "fresh")  # too_far stayed with agents
    net.release(other, net.store.one("SELECT f.name FROM claims c JOIN functions f ON f.id=c.function_id "
                                     "WHERE c.agent_id=?", other.id)["name"], "", "game")
    proj = net.project_row("game")
    queue = [r["name"] for r in net.permuter_queue(proj)]
    assert queue[:3] == ["regs", "said_so", "asked"] and "data_only" not in queue, queue
    permuter, _ = net.local_agent("permuter")
    abandons = net.store.one("SELECT abandons FROM functions WHERE name='regs'")["abandons"]
    for _ in range(network.PERMUTER_TURNS):
        net.permuter_done(permuter, net._function("regs", "game")["id"], "")
    row = net.store.one("SELECT * FROM functions WHERE name='regs'")
    assert row["handoff"] is None and row["permuter_turns"] == network.PERMUTER_TURNS  # back to the agents
    assert row["abandons"] == abandons  # a permuter run isn't an agent giving up
    assert "regs" not in [r["name"] for r in net.permuter_queue(proj)]
    assert "permuter" in net.list_functions("game", "permuter") and "said_so" in net.list_functions("game", "permuter")


def test_the_priority_list_never_hands_back_a_function_just_released(tmp_path, monkeypatch):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [{"name": "stuck", "n_insns": 30}, {"name": "other", "n_insns": 50},
                             {"name": "third", "n_insns": 60}], {"priority": ["stuck"]})
    a = _agent(net, "claude-1", None)
    assert net.claim_row(a, "game", None, "easiest")["name"] == "stuck"
    net.release(a, "stuck", "permuter only; regalloc", "game")  # (left for the permuter: it has no best attempt)
    assert net.claim_row(a, "game", None, "easiest")["name"] == "other"  # not the one it just released
    b = _agent(net, "claude-2", None)
    assert net.claim_row(b, "game", None, "easiest")["name"] == "third"  # released by someone just now: cooling
    monkeypatch.setattr(network, "RELEASE_COOLDOWN", 0)
    net.release(b, "third", "", "game")
    c = _agent(net, "claude-3", None)
    assert net.claim_row(c, "game", None, "easiest")["name"] == "stuck"  # cooled down: the priority again
    net.release(c, "stuck", "", "game")
    net.set_handoff("game", "stuck", "permuter")
    d = _agent(net, "claude-4", None)
    assert net.claim_row(d, "game", None, "easiest")["name"] != "stuck"  # left for the permuter: skipped


def test_cheap_tiers_are_stopped_after_twice_the_nudge(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    pid = _project(net, tmp_path, [{"name": "f", "n_insns": 10}])
    small = _agent(net, "haiku-1", "small")
    fn = net.claim_row(small, "game", "f")
    with net.store.tx() as db:
        for score in [50] + [60] * 8:
            db.execute("INSERT INTO attempts(function_id,agent_id,score,compiled,created_at) VALUES(?,?,?,1,0)",
                       (fn["id"], small.id, score))
    with pytest.raises(NetworkError, match="stop here"):
        net.attempt(small, "f", "int f(void) { return 0; }", "game")  # refused before compiling anything
    assert pid


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
    assert tiers.diff_kind(DiffResult(10, 100, [row(" "), row("r"), row("o")], False)) == "regalloc"
    assert tiers.diff_kind(DiffResult(5, 100, [row(" "), row("s")], False)) == "regalloc"
    assert tiers.diff_kind(DiffResult(100, 100, [row("r"), row("<")], False)) == "structure"
    assert tiers.diff_kind(DiffResult(1, 100, [row(" ")], False), verdict=False) == "data"
    assert tiers.diff_kind(DiffResult(0, 100, [row(" ")], True)) is None


def test_instructions_are_kept_whole_and_go_with_the_tier_guidance(tmp_path):
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


def test_handoffs_never_strand_a_function(tmp_path):
    """The permuter had its runs: a note saying "permuter" doesn't send it back there (nobody would take it). Left for
    a human: claiming it by name means somebody is looking at it, and it's everyone's again."""
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    _project(net, tmp_path, [
        {"name": "done_permuting", "n_insns": 30, "best_score": 40, "best_source": "int a;", "best_kind": "regalloc",
         "permuter_turns": network.PERMUTER_TURNS},
        {"name": "for_you", "n_insns": 30, "handoff": "human", "handoff_note": "tool problem?"},
    ])
    a = _agent(net, "opus-1", "large")
    net.claim_row(a, "game", "done_permuting")
    assert "Left for" not in net.release(a, "done_permuting", "regalloc only: permuter", "game")
    assert net.store.one("SELECT handoff FROM functions WHERE name='done_permuting'")["handoff"] is None
    you = _agent(net, "you", None)
    assert net.claim_row(you, "game", None, "easiest")["name"] != "for_you"  # never automatically
    net.release(you, net.store.one("SELECT f.name FROM claims c JOIN functions f ON f.id=c.function_id "
                                   "WHERE c.agent_id=?", you.id)["name"], "", "game")
    assert net.claim_row(you, "game", "for_you")["name"] == "for_you"  # by name: yours, and the handoff is gone
    assert net.store.one("SELECT handoff FROM functions WHERE name='for_you'")["handoff"] is None


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


def test_tiers_disabled_helper_gets_what_the_tiers_hold_back(tmp_path):
    """A helper started with tiers disabled ("any") takes big functions and ones left for the permuter, which a
    small-tier helper never gets; what it released lately comes last."""
    import json

    from decompnet.app.engine import RomNetwork

    net = RomNetwork(tmp_path / "app.db", tmp_path)
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(tmp_path), json.dumps({"kind": "rom"}))).lastrowid
        for name, n, handoff in [("big", 400, None), ("handed", 20, "permuter")]:
            db.execute("INSERT INTO functions(project_id,name,size,n_insns,status,unit,handoff) VALUES(?,?,?,?,?,?,?)",
                       (pid, name, n * 4, n, "open", "a", handoff))
    small, _ = net.local_agent("small")
    net.set_agent_tier(small.id, "small")
    assert net.left_for(small, "game") == 0
    free, _ = net.local_agent("free")
    net.set_agent_tier(free.id, "any")
    assert net.left_for(free, "game") == 2
    assert net.claim_row(free, "game", None, "easiest")["name"] == "big"  # the permuter's one comes after
    free2, _ = net.local_agent("free2")
    net.set_agent_tier(free2.id, "any")
    assert net.claim_row(free2, "game", None, "easiest")["name"] == "handed"
