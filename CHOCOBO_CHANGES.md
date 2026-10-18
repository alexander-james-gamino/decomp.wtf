# Changes in this fork: model tiers, target routing, permuter handoffs, overlay fixes

2026-10-17. On top of `decomp.wtf-chocobo`. Tested on macOS with the project's test suite: 117 pass (102 before;
the new `tests/test_tiers.py` plus two outdated update tests fixed). No test that passed before fails now. The
remaining failures are the same as before: the demo projects need Linux GCC and the period compilers.

## 1. Model tiers: "if you're Haiku, do this" (`src/decompnet/tiers.py`)

Every helper gets a tier when its session starts (`agents.tier` column), detected from how it runs:

| Tier | Detected from | Gets | Batch | Claims/session | Move-on nudge | Hard stop |
|---|---|---|---|---|---|---|
| local | `api` helper on a LAN/localhost endpoint; opencode model with an Ollama tag (`mbp/qwen3.8:27b`) or an ollama/lmstudio/... provider | ≤ 40 instructions, never given up on | 1 | 1 | 1 compile sooner | yes |
| small | model name contains `haiku` | ≤ 80 instructions, given up on ≤ 1 time | 4 | 2 | size-based (4–10) | yes |
| medium | `sonnet` | ≤ 250 instructions, given up on ≤ 3 times (more if no large model is around) | 3 | 2 | size-based | no |
| large | `opus`, `fable`, `mythos` | anything; the "frontier" order | 2 | 2 | 2 compiles later | no |
| none | anything else (the agent you talk to, a default model) | everything, as before | as asked | 2 | size-based | no |

A helper option `tier` (local / small / medium / large) overrides the detection, and `tier: "any"` turns routing off
for that helper. There's no UI field for it yet: detection from the model name covers Claude and local endpoints.

Per tier:
- **Target selection is on the server** (`Network._eligible` / `_next_for`): what a tier is handed automatically,
  in what order, and when nothing is left for it.
- **Division of labour:** a tier above one that's working right now (claims, or activity in the last 20 minutes)
  takes what that one can't first. With Haiku helpers running, an Opus helper starts on big functions and ones
  Haiku gave up on, not on functions Haiku would match anyway. Alone, it works in the usual order.
- **Stopping:** a small or local helper stops when nothing left suits it ("nothing left that suits a small model:
  N functions are open"). Start a stronger model for the rest.
- **Hard stop (local, small):** after twice the move-on nudge without a better score, compile_and_diff refuses and
  says to release. That saves compiles and tokens on functions the model won't get.
- **Prompt:** each tier gets a short role description appended to the session prompt. Local helpers on the built-in
  `api` kind get 8 tools instead of 15.

## 2. Leave it for the permuter / a human

- `release_function(function, notes, leave_for="permuter" | "human")`. Without `leave_for`, a function still goes
  to the permuter when its best attempt differs only in registers/order, or when the release note says
  "regalloc only" / "permuter".
- **Detected on every compile:** each best attempt's diff is classified (`functions.best_kind`):
  - `regalloc`: only registers, order or stack offsets differ;
  - `data`: every instruction matches, only data differs;
  - `structure`: real code differences.

  The compile response tells the agent which handoff applies.
- Handed-off functions are never claimed automatically. Claiming one by name still works, and claiming a "human"
  one by name gives it back to everyone. `list_functions(status="permuter" | "human")` shows them with the reason.
- **One-time migration:** open functions whose notes already say "regalloc only" / "permuter only" (with a best
  attempt to start from) are handed to the permuter on first start. `func_800A6238` and its 256 "queue keeps
  re-serving this" notes are covered by this.

## 3. The permuter's priority queue (`Network.permuter_queue`, `permuter_done`)

The order is: functions agents left for it, then `regalloc` near misses, then fewer permuter runs so far, then the
closest score, then the smallest. Data-only diffs and human handoffs are excluded. After 3 permuter runs without a
match (`PERMUTER_TURNS`), the handoff is cleared and the function goes back to the agents, where the large tier
takes it. A permuter run no longer counts as an agent giving up (`abandons`).
- The built-in permuter (`app/permute.py`) uses this queue.
- Your `tools/permute/dp_worker.py` (in the slus-00844 zip) uses it too, with a fallback for older versions.

## 4. Bug fixes from the review

- **Overlays at the same address:** `reader(proj, fn)` reads from the function's own segment and non-overlapping
  ones, never another overlay (`rom/split.visible_segments`). This applies to the judge's data check, the jump
  tables fed to m2c, the agents' `read_data` (as seen by the claimed function), and the export's re-check. The
  import's data previews are read by ROM offset.
- **Export:** `symbol_addrs.txt` gets one line per function, pinned with `rom:`, so names no longer apply to every
  overlay. Functions are matched to database rows by address and ROM offset, falling back to the address alone for
  old projects. No more fake functions or C copied into the wrong overlay.
- **`build.py`** (copied from `rom/export_build.py`): wrapper files and objects are keyed by (segment, name).
  `check SEG/NAME` works, a bare name in several segments is refused, and its reads and names are per segment.
- **The priority list** follows the same rules as every claim: not something this agent released, nor something
  released within the cooldown, nor a handoff, and only what fits the tier. That ends the endless re-serving.
- **Instructions** are kept up to 12,000 characters (was 4,000, cut silently).
- **The work prompt** says where case numbers come from (the jump table, via read_data) and when to use each
  `leave_for`.

## Deploying (Linux box)

1. Install this fork the way the chocobo branch is installed now (`uv tool install --force ...`, then
   `decomp setup`), and restart decomp.wtf. The new database columns and the one-time migration apply on start.
2. Copy `tools/toolchain/cc.py` and `tools/permute/dp_worker.py` from the slus-00844 zip into
   `~/Decomp/slus-00844/tools/`. `sync_repo.py` copies tools from there into the repo, so edits made only in the
   repo would be overwritten.
3. Paste `tools/agent-instructions.md` (slus-00844 zip) into Setup → Instructions, and finish its [continue] part
   with the rest of your original playbook.
4. Start helpers with explicit models (`haiku`, `sonnet`, `opus`) so they get tiers. A local model works best as
   the built-in `api` kind pointed at the box's endpoint.
5. Re-export (`sync_repo.py --no-push`) and check:
   - about 2,376 functions;
   - no duplicate names: `python3 -c "import json,collections; c=collections.Counter(f['name'] for f in json.load(open('build.json'))['functions']); print([n for n in c if c[n]>1][:5])"`;
   - no `ovl_*` C files that are copies of another overlay's;
   - then `python build.py`.

   The `rom:` symbol attribute is splat's mechanism for overlays sharing addresses, but it couldn't be run here (no
   ROM). If splat complains, report it.
6. Rescore `func_8009E820` and the other jump-table functions in `ovl_m_chedit`/`ovl_m_mdsel`. The corrected C for
   `func_8009E820` is in the review (`SLUS-00844_REVIEW.md`).

## Not done (yet)

- **Per-overlay symbol table** (`rom/symbols.SymbolTable` keeps one name per address). Scoring is right now
  because it reads by segment, but labels like `func_80098FEC+0x18` stay in existing assembly text until a
  re-import.
- **splat's `exclusive_ram_id`** for the overlay segments.
- **UI controls** for tiers and handoffs.
- **`best_kind` for old attempts.** Existing best attempts have none until they're compiled again; a rescore fills
  it in.

---

## 3. Session-limit auto-resume, and smaller UI things (2026-10-17, later)

- **Session limit** (`agents.py`): a helper stopped by the account's plan cap ("5-hour limit reached · resets
  5:00pm") now waits out the reset instead of dying after three tries. `throttle_wait()` parses the reset time
  (wall clock with its IANA zone, or "try again in 47m"; 12 h cap), and the throttled branch waits that long.
  Plain 429s keep the old 60-second retry.
- **Hover info per attempt** (`app.js`): the diff view's "Latest attempt / Best" buttons show who·when·score on
  hover; the functions table shows each function's compile count with its best score, matcher and claim; the
  "now working" cards list the last compiles under the tries line. The per-compile chart already had its own
  tooltips.
- **Long text can't bleed out of its box** (`app.css`): notes, diff problems, the try message and job cards now
  have `overflow-wrap: anywhere` (long URLs/addresses in a note can no longer push text across the layout).
- **The notes can no longer cover the assembly and C panes** (`app.css`): in the function drawer, Chrome sizes
  the `auto` grid rows of `.drawer-body` against the card's fixed height, so the row with the 58vh-tall
  assembly/C panes was squeezed to ~314px and the panes painted over the rows below (xref, notes). Grid rows
  are now `max-content`, so the drawer body scrolls instead of the rows compressing; reproduced and verified
  with headless Chromium against the running app (notes at y 862 below panes at 713, no intersection).
- **Average completeness** (`engine.py` + `app.js`/`index.html`): under "150.9 KB of 620.6 KB · 91 matched today"
  a second gray line now reads "the average function is 87.5% there · 61.9% weighted (by instructions)" — the mean
  best-attempt completeness over all functions (matched = 100%, hand-written assembly = 100%, untried = 0%). The
  plain mean weighs every function once; the weighted mean weighs each by its instructions, so it shifts toward
  the big (usually hardest) functions and sits far below the plain one when small functions are near-matched but
  big ones are barely attempted. Also visible: it drags toward the byte-exact % as (nearly-)complete attempts
  become full matches.
- **Model tiers off by default, with a GUI toggle** (`agents.py`, `server.py`, settings page): the tier routing
  (cheap models get small functions, strong models the hard ones) is now disabled unless the user turns on
  **Model tiers** in Settings. Off: every helper can claim anything (the size-based move-on nudge stays).
  A helper only learns its tier when it starts while the toggle is on; switching it off clears the stored
  tiers of helpers already working.
- **Finishing the ROM-map SDK work** (`server.py` + `app.js`/`app.css`/`index.html`): the map's `/api/map` payload
  forgot the `sdk` field (`MAP_KEYS`), so the red squares and the "Sony SDK" badge never reached the browser —
  added; the map legend now shows "Sony SDK" (red) and "assembly" chips when anything is set aside; and the
  matched-code line spells out the breakdown: "51.3 KB set aside: 38.5 KB Sony SDK, 12.8 KB assembly".
