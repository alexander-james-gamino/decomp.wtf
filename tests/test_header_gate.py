"""A header edit is blamed only for what it breaks. slus-00844's context.h froze for days: about 16 matched functions
no longer reproduced with the header as it was (a declaration changed since they matched), so every edit was
rejected, even a comment, and no agent could add a macro or a type."""

import json

from decompnet.app.engine import RomNetwork


def _setup(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(tmp_path), json.dumps({"kind": "rom"}))).lastrowid
        db.execute("INSERT INTO context_versions(project_id,version,header,agent_id,message,created_at,unit) "
                   "VALUES(?,1,?,NULL,'',0,'')", (pid, "typedef int s32;\n"))
        for name in ("fine", "drifted", "victim"):
            db.execute("INSERT INTO functions(project_id,name,unit,status,n_insns,size,matched_source,matched_at) "
                       "VALUES(?,?,'','matched',4,16,?,1)", (pid, name, f"void {name}(void) {{}}"))
    calls = []

    def fake_regression(proj, header, sources=None, only=None, header_for=None):
        """'drifted' fails with any header (it doesn't reproduce today); 'victim' fails once the header says BREAK."""
        calls.append(header)
        out = []
        for fn in only:
            h = header_for(fn) if header_for else header
            if fn["name"] == "drifted" or (fn["name"] == "victim" and "BREAK" in h):
                out.append((fn["name"], "compiles but no longer matches"))
        return out

    net.regression_test = fake_regression
    net._require_level = lambda agent, what: None
    agent, _ = net.local_agent("helper")
    return net, agent, calls


def test_an_edit_is_not_blamed_for_functions_that_already_fail(tmp_path):
    net, agent, calls = _setup(tmp_path)
    ok = net.edit_context(agent, "game", "", "#define GTE_SET_ZSF3(r) /* ctc2 $29 */", "SetGeomOffset macros")
    assert ok.startswith("✅ header is now v2") and "drifted" in ok and "already fail" in ok
    bad = net.edit_context(agent, "game", "", "/* BREAK */", "breaks victim")
    assert bad.startswith("❌ REJECTED: the edit breaks 1 ") and "- victim" in bad and "- drifted" not in bad
    assert "drifted" in bad  # still named, as not counted


def test_the_baseline_is_computed_once_per_header_version(tmp_path):
    net, agent, calls = _setup(tmp_path)
    net.edit_context(agent, "game", "", "/* BREAK */", "one")      # edit + baseline at v1: rejected
    net.edit_context(agent, "game", "", "/* BREAK too */", "two")  # edit; the v1 baseline comes from the cache
    assert len(calls) == 3
    net.edit_context(agent, "game", "", "/* fine */", "three")     # edit + baseline at v1 (cached): accepted, now v2
    net.edit_context(agent, "game", "", "/* BREAK */", "four")     # header v2: a new baseline
    assert len(calls) == 6


def test_a_clean_project_still_rejects_what_breaks(tmp_path):
    net, agent, calls = _setup(tmp_path)
    with net.store.tx() as db:
        db.execute("UPDATE functions SET status='open' WHERE name='drifted'")
    assert net.edit_context(agent, "game", "", "/* fine */", "ok").startswith("✅")
    assert net.edit_context(agent, "game", "", "/* BREAK */", "no").startswith("❌ REJECTED: the edit breaks 1 ")
