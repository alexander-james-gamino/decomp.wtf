"""Briefs that fit in one tool result (assembly in pages, capped attempts and notes), the status digest at the top of
a brief, and a session prompt whose stable part comes first."""

import json

import pytest

from decompnet import network
from decompnet.app import agents
from decompnet.network import NetworkError


def _project(net, root, rows):
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(root), json.dumps({"kind": "rom"}))).lastrowid
        db.execute("INSERT INTO context_versions(project_id,version,header,agent_id,message,created_at,unit) "
                   "VALUES(?,1,?,NULL,'',0,'')", (pid, "typedef int s32;\n"))
        for r in rows:
            cols = {"project_id": pid, "status": "open", "unit": "a", "target": bytes(r["size"]), **r}
            db.execute(f"INSERT INTO functions({','.join(cols)}) VALUES({','.join('?' * len(cols))})", tuple(cols.values()))
    return pid


def test_a_huge_function_comes_in_pages(tmp_path):
    net = network.Network(tmp_path / "app.db")
    asm = "\n".join(f"    addiu      $v0, $v0, {i}" for i in range(1000))
    _project(net, tmp_path, [{"name": "huge", "n_insns": 1000, "size": 4000, "asm": asm,
                              "best_score": 9000, "best_source": "x" * 20000}])
    brief = net.get_function("huge", "game")
    assert "page 1 of 3" in brief and 'page=2' in brief
    assert "$v0, 449" in brief and "$v0, 450" not in brief  # the first page only
    assert len(brief) < 40000  # fits in one tool result (the uncut brief was ~50,000 characters)
    assert "more characters: get_function with page=-1" in brief  # the 20,000-character attempt is cut short
    page2 = net.get_function("huge", "game", page=2)
    assert "page 2 of 3" in page2 and "$v0, 450" in page2 and "$v0, 899" in page2 and "$v0, 900" not in page2
    assert "$v0, 999" in net.get_function("huge", "game", page=3)
    with pytest.raises(NetworkError, match="3 pages"):
        net.get_function("huge", "game", page=4)
    assert len(net.get_function("huge", "game", page=-1)) > 20000  # the whole attempt


def test_the_status_digest_heads_the_brief(tmp_path):
    net = network.Network(tmp_path / "app.db")
    pid = _project(net, tmp_path, [{"name": "f", "n_insns": 10, "size": 40, "asm": "    nop"}])
    fid = net._function("f", "game")["id"]
    with net.store.tx() as db:
        db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,1)",
                   (pid, fid, None, "DIGEST: best 40; tried u8 vs s32 temps; next: pin the base in $s0"))
        for i in range(10):  # newer notes than the digest: it must not scroll away
            db.execute("INSERT INTO notes(project_id,function_id,agent_id,text,created_at) VALUES(?,?,?,?,2)",
                       (pid, fid, None, f"note {i} " + "y" * 2000))
    brief = net.get_function("f", "game")
    assert brief.index("Status digest") < brief.index("Target assembly")
    assert "next: pin the base in $s0" in brief
    assert brief.count("DIGEST:") == 1  # only in the instruction to keep it current, not again among the notes
    assert "note 9" in brief and "note 1 " not in brief  # the latest 8 ordinary notes
    assert "search_notes has the full note" in brief  # each note capped


def test_the_session_prompt_keeps_what_changes_for_last():
    class Net:
        def overview(self, slug):
            return {"title": "Game", "platform": "psx", "compiler": {"toolchain": "gcc-2.8.1-psx"}}

        def learnings(self, slug):
            return "- NEWEST NOTE"

        def instructions(self, slug):
            return "USER INSTRUCTIONS"

    m = agents.Manager.__new__(agents.Manager)
    m.net = Net()
    m.tiers_on = lambda: True  # the role text only appears when the Model tiers setting is on
    s = agents.Session("claude-1", "claude", "c", "game", "claude-1", {"model": "haiku"})
    prompt = agents.Manager.prompt(m, s)
    assert prompt.index("USER INSTRUCTIONS") < prompt.index("Your role here (small model") < prompt.index("NEWEST NOTE")
    assert prompt.rstrip().endswith("- NEWEST NOTE")
    assert "DIGEST:" in prompt  # the work prompt asks for the digest before a release
    assert network.ASM_PAGE_LINES == 450
