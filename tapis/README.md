# Tapis App artifacts

For Sina / whoever's porting this to Vista.

## `fnas-lcd-train-0.1.3.app.json`

The real, live Tapis App definition (`GET /v3/apps/fnas-lcd-train/0.1.3`),
pulled 2026-10-07. This is what Tapis actually has registered — not
something hand-written, so it's authoritative for what the App expects:
`containerImage`, `appArgs`, `containerArgs`, resource limits, etc.

The App is **not public** (`isPublic: false`), owned by `jinghuayan96`.
You likely can't query this yourself yet — either ask to be added to its
share list, or treat this file as your copy of it for now.

## `trainer.def`

**Reconstructed, not the original.** The real build file for
`trainer.sif` (the `.sif` this App's `containerImage` points at, on
Pitzer's filesystem — not a registry image) could not be found on any
machine or backup as of 2026-10-07. What's here is a best-effort rebuild
from `fnas-lcd-train-0.1.3.app.json`'s documented CLI/runtime and this
repo's own `src/training/timm_trainer.py` — see the comment block at the
top of the file for exactly what's confirmed vs. reconstructed.

**Before relying on this for Vista:** build it and run a short smoke-test
job (`--max_iters 5`, matching how local testing was done) against Pitzer
first, where there's something to compare against, before assuming it's
equivalent to what's actually been running in production. If it behaves
differently, that's real signal the reconstruction missed something, not
a Vista-specific problem.
