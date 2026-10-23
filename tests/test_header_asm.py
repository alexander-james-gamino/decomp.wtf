"""context.h macros may drive the GTE and pin registers, nothing else. On slus-00844 a local model "matched" 43
functions in a day by pasting each one's assembly into ASM_BODY("...") and helpers like it, defined in context.h
(the rule against inline asm only looked at function sources)."""

import json

from decompnet.app.engine import RomNetwork
from decompnet.compiler import CompilerConfig, SourceRejected, check_header_use, compile_source, header_asm_problems

GTE = '''
#define GTE_SET_R0(r) __asm__ volatile("ctc2 %0, $0" :: "r"(r))
#define GTE_AVSZ4() __asm__ volatile("nop; .word 0x4B68002E")
#define GTE_SQR_ALL(src, dst) __asm__ volatile(".set noreorder; lh $2, 0(%0); lh $3, 2(%0); mtc2 $2, $9; \\
    mtc2 $3, $10; lwc2 $11, 4(%0); nop; .word 0x4AA80428; swc2 $9, 0(%1); .set reorder" :: "r"(src), "r"(dst) : "$2")
#define A0_VAR(n) register s32 n asm("$4")
'''
BAD = '''
#define ASM_BODY(s) __asm__ volatile(s :: "m"(D_8007B0A0))
#define WRAP(fn) __asm__ volatile("sw $31, 0x10($29); jal " #fn "; nop")
#define SRA_ADDI_M1(x) __asm__ volatile("sra %0, %0, 2; addi %0, %0, -1" : "=r"(x) : "0"(x))
#define LOOP() __asm__ volatile("1: nop; mtc2 $2, $9")
#define WORD() __asm__ volatile(".word 0x27BDFFE8")
#define LOAD(x) __asm__ volatile("lw %0, 0($4)" : "=r"(x))
'''


def test_gte_macros_and_pins_pass_and_pasted_assembly_does_not():
    assert header_asm_problems(GTE) == {}
    bad = header_asm_problems(BAD)
    assert set(bad) == {"ASM_BODY", "WRAP", "SRA_ADDI_M1", "LOOP", "WORD", "LOAD"}
    assert "macro argument" in bad["ASM_BODY"] and "#x" in bad["WRAP"] and "sra" in bad["SRA_ADDI_M1"]
    assert "label" in bad["LOOP"] and "GTE command" in bad["WORD"] and "without a GTE" in bad["LOAD"]


def test_a_source_using_a_pasting_macro_is_refused_before_compiling():
    src = 'void f(void) { s32 dummy[6]; ASM_BODY("addiu $v0, $zero, 1; jr $ra"); }'
    try:
        check_header_use(src, BAD)
        raise AssertionError("not refused")
    except SourceRejected as e:
        assert "ASM_BODY" in str(e)
    check_header_use("void f(void) { GTE_AVSZ4(); }", GTE + BAD)  # the GTE macro is fine, even in a dirty header
    res = compile_source(CompilerConfig(cmd=["false"]), BAD, src)
    assert not res.ok and res.log.startswith("rejected: ASM_BODY")


def test_header_edits_may_not_add_one(tmp_path):
    net = RomNetwork(tmp_path / "app.db", tmp_path)
    with net.store.tx() as db:
        pid = db.execute("INSERT INTO projects(slug,name,description,arch,big_endian,root,config) VALUES(?,?,?,?,?,?,?)",
                         ("game", "Game", "", "mips", 0, str(tmp_path), json.dumps({"kind": "rom"}))).lastrowid
        db.execute("INSERT INTO context_versions(project_id,version,header,agent_id,message,created_at,unit) "
                   "VALUES(?,1,?,NULL,'',0,'')", (pid, GTE))
    net.regression_test = lambda *a, **k: []
    net._require_level = lambda agent, what: None
    agent, _ = net.local_agent("helper")
    no = net.edit_context(agent, "game", "", '#define ASM_BODY(s) __asm__ volatile(s)', "a generic helper")
    assert no.startswith("❌ REJECTED: context.h macros may emit GTE") and "- ASM_BODY" in no
    ok = net.edit_context(agent, "game", "", '#define GTE_SET_ZSF3(r) __asm__ volatile("ctc2 %0, $29" :: "r"(r))',
                          "SetGeomOffset")
    assert ok.startswith("✅")
