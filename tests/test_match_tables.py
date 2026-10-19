"""How a `.L`-labelled data reference is placed when its ROM relocation lives at another instruction:

`target_from_split` must not lose refs to `.L<hex>` names (disassembly labels no one labels), and a switch's
own table goes where the ROM keeps it even when the compiler moved the `lui`/`lw`: pair candidate references
with the original's still-unused relocations of the same kind, in the original's order.
"""

from decompnet.rom import match
from decompnet.rom.match import _Linker
from decompnet.rom.split import AsmLine, RomFunction


def linker(rels, local=None):
    lk = _Linker.__new__(_Linker)  # _place only needs the placed state; no object to read
    lk.target_at = {off: (rtype, addr) for off, rtype, addr in rels}
    lk.local_map = local or {}
    lk._used = set()
    return lk


def fn(lines):
    return RomFunction("func_80098F80", 0x80098F80, 4 * len(lines), "main", lines, None, set())


def test_dot_l_labels_resolve_as_their_address():
    # splat names a disassembly's own labels after their vram; the symbol table never lists them
    huge = match.target_from_split(fn([AsmLine(0x1004, 0x80098F84, 0x3C018009, "lui", "$at, %hi(.L80098F84)")]),
                                   lambda name: None)
    assert huge.relocs == [(4, match.R_MIPS_HI16, 0x80098F84)]
    still_unknown = match.target_from_split(fn([AsmLine(0x1004, 0x80098F84, 0x3C018009, "lui", "$at, %hi(other_name)")]),
                                            lambda name: None)
    assert still_unknown.relocs == []


def test_a_reference_the_original_has_at_the_same_instruction():
    lk = linker([(8, match.R_MIPS_HI16, 0x80012688)])
    assert lk._place({"off": 8, "type": match.R_MIPS_HI16}, 3, 0) == 0x80012688
    assert 8 in lk._used  # explained once: a later own-data reference pairs with something else


def test_moved_references_pair_with_unused_relocations_in_order():
    # two switch tables, the compiler emits every reference elsewhere than the original does
    lk = linker([(8, match.R_MIPS_HI16, 0x80012688), (16, match.R_MIPS_HI16, 0x80012690),
                 (24, match.R_MIPS_LO16, 0x80012688), (32, match.R_MIPS_LO16, 0x80012690)])
    assert lk._place({"off": 100, "type": match.R_MIPS_HI16}, 3, 0) == 0x80012688
    assert lk._place({"off": 104, "type": match.R_MIPS_HI16}, 4, 0) == 0x80012690
    assert lk._place({"off": 108, "type": match.R_MIPS_LO16}, 3, 0) == 0x80012688
    assert lk._place({"off": 112, "type": match.R_MIPS_LO16}, 4, 0) == 0x80012690
    assert lk._place({"off": 116, "type": match.R_MIPS_HI16}, 5, 0) is None


def test_paired_place_prefers_one_the_datum_already_uses():
    lk = linker([(8, match.R_MIPS_HI16, 0x80012688), (16, match.R_MIPS_HI16, 0x80012690)], {(3, 0x10): 0x80012690})
    assert lk._place({"off": 100, "type": match.R_MIPS_HI16}, 3, 0x10) == 0x80012690


def test_a_conflict_still_falls_through_for_the_check_to_report():
    # nothing free agrees with where the datum is placed: keep the original's order (the first unused)
    # so `address` reports the datum "used for two different ROM addresses" instead of hiding it here
    lk = linker([(8, match.R_MIPS_HI16, 0x80012688)], {(3, 0x10): 0x80012690})
    assert lk._place({"off": 100, "type": match.R_MIPS_HI16}, 3, 0x10) == 0x80012688


def test_references_of_another_kind_do_not_get_mixed_in():
    lk = linker([(8, match.R_MIPS_HI16, 0x80012688)])
    assert lk._place({"off": 100, "type": match.R_MIPS_32}, 3, 0) is None
