# PERF_PROPOSALS — state file (RLGymSim_CPP wrapper engagement)

> ## ⚠ STATUS — READ THIS FIRST
> This file tracks perf work on the **standalone RLGymSim_CPP C++ gym wrapper** (the Phase-1 sim
> layer that RLGymPPO_CPP will vendor). It is SEPARATE from the parent `D:/RLBotTraining/PERF_PROPOSALS.md`
> (the L1 library-selection engagement, already resolved: L1 approved, sim-port-first).
>
> **Blocking finding (2026-11-12): the library does not compile at HEAD.** The wrapper layer was
> mid-migrated from a flat `FList`/`FList2` obs interface to a `(q, kv, m)` tuple interface and left
> incoherent. There is therefore NO working baseline to benchmark, so no Tier A perf change can be
> verified as "provably equivalent." Tier A is EMPTY until the build is fixed. The build fix is
> Tier B (public interface / return-shape + obs-content decision) and is escalated below, not applied.

## Resource budget (measured 2026-11-12, this session)
- CPU: Intel i9-10850K, 20 logical cores (10 physical)
- RAM: 65,382 MB total; ~40,243 MB available at pre-flight
- GPU: RTX 3090 24576 MiB, ~2063 MiB used → ~22.5 GiB live VRAM headroom
- Page file: system-managed (virtual max ~69,478 MB). Pinned ceilings: RAM 64 GB, GPU 24 GB.
- Toolchain present: CLion-bundled MinGW g++ + cmake 4.1 (win/x64). No system cmake/cl on PATH.

## Ground truth established this session
- `git log -1` = `e3f694d feat: port custom features from RocketSim fork to RLGymSim_CPP`. That port
  commit is what broke the build.
- The bundled engine (`RocketSim/`) builds clean: `Built target RocketSim`. Only the `src/RLGymSim_CPP`
  wrapper fails.
- The **working reference** is the vendored copy at
  `D:/RLBotTraining/RLGymPPO_CPP/RLGymPPO_CPP/RLGymSim_CPP/src/RLGymSim_CPP/`. There:
  - `OBSBuilder::BuildOBS(...) -> FList` (flat)
  - `Match::BuildObservations(...) -> FList2`
  - `Gym::Reset()/Gym::Step() -> FList2`
  The standalone fork diverged: `Match::BuildObservations -> std::vector<ObsTuple>` and
  `TemporalObsBuilder::BuildOBS -> std::tuple<FList,FList,FList>`, with `Gym::obsTupleToFList2`
  flattening the tuple back to `FList2`. The two halves of that migration do not agree, so it won't build.

## Exact compile errors (from `cmake --build build` at HEAD, GCC 14 / C++20)
1. `Utils/Gamestates/GameState.cpp:25` — `arena->_boostPads[j]->pos` → BoostPad has no `pos`;
   should be `->config.pos` (see `RocketSim/src/Sim/BoostPad/BoostPad.h`: field is `config.pos`).
2. `Utils/Gamestates/GameState.cpp:59` — `attachInfo = ballState.attachInfo;` assigns
   `RocketSim::BallState::AttachInfo` into `RLGSC::AttachInfo`; no conversion/assignment exists.
3. `Envs/Match.cpp:18` — `results[i] = obsBuilder->BuildOBS(...)` assigns `FList` into `ObsTuple`
   (the tuple-vs-flat conflict, see item 4).
4. `Utils/OBSBuilders/TemporalObsBuilder.h:89` — `BuildOBS` overrides base `OBSBuilder::BuildOBS`
   (returns `FList`) with a `std::tuple<...>` return type → **invalid covariant return type**. This is
   the root interface conflict.
5. `Utils/OBSBuilders/TemporalObsBuilder.cpp:17` — `slotType.resize(entities, EntityType::BALL)`
   but `slotType` is `std::vector<int>`; the fill value is an unscoped-enum mismatch.
6. `TemporalObsBuilder.cpp:64/70/77` vs `:158/216/277` — `extractBallFeatures/extractEntityFeatures/
   extractPadFeatures` are DEFINED with extra `(float* out, int E, int kvDims)` params but DECLARED
   in the header with fewer params → "no declaration matches".
7. `TemporalObsBuilder.cpp:141-143` — `Vec::dot` → actual method is `Vec::Dot` (capital D).
8. `TemporalObsBuilder.cpp:386` — `Vec::len` → actual method is `Vec::Length`.
9. `TemporalObsBuilder.cpp:400` — `CommonValues::BALL_MAX_ANG_VEL` does not exist (only `CAR_MAX_ANG_VEL`).

## Open (awaiting user decision — all Tier B, NOT self-applied)
- [ ] **R1 — Fix the obs-tuple interface so the wrapper compiles** | Tier B: it is a return-shape /
      public-interface decision AND it changes obs CONTENT (the fork's `TemporalObsBuilder` emits a
      30-dim-per-entity temporal feature layout, totally different from `DefaultOBS`). | expected
      gain: unblocks ALL perf work (nothing can be measured until it builds) | effort: M (mechanical
      once the interface is chosen) | risk: obs vector layout changes → any checkpoint/policy width
      tuned to the old flat layout is invalid | rollback: revert the port commit `e3f694d` |
      **recommendation: pick ONE interface and finish the migration. Given the Perceiver needs
      (q,kv,m), the fork's intent is clearly the tuple path — so make `OBSBuilder::BuildOBS` return
      `ObsTuple` (or add a separate virtual), keep `Gym` flattening to `FList2` via `obsTupleToFList2`,
      and fix items 1,2,5,6,7,8,9 mechanically. But this is your call because it defines the obs schema
      the learner will consume.**
- [ ] **R2 — `obsTupleToFList2` + `buildKeyValue`/`buildMask` rebuild the whole (F×E×kv) buffer every
      step** | Tier B (perf, but can't be validated until R1 compiles): `Match::BuildObservations`
      calls `BuildOBS` per player, and each `BuildOBS` → `buildKeyValue` copies the entire temporal
      buffer into a fresh `FList` (13 frames × 43 slots × 30 dims ≈ 16,770 floats) AND `buildMask`
      rebuilds a constant all-ones mask every call. For an N-player match that's N full buffer copies
      per step, and the mask is invariant. | expected gain: large reduction in per-step allocation +
      memcpy once it builds | effort: M | risk: must prove flattened output is byte-identical |
      rollback: git revert | **recommendation: after R1, hoist the kv/mask build out of the per-player
      loop (build once per step, slice per player) and cache the constant mask. Do NOT do this before
      R1 — there is no baseline to diff against yet.**

## Approved - implementing
- (none yet — R1/R2 are unapproved Tier B items awaiting your decision)

## Done
- (none yet)

## Declined
- (none yet)

## Notes
- The engine hot path (`Arena::Step`, `Car::Update`, `Ball` attach logic) is untouched and builds; no
  Tier A engine perf item is proposed there because there is nothing to measure against yet.
- Increasing env/worker count is Tier B always (ordering + RAM multiply) — not raised as a candidate.
- Once R1 lands and a golden obs baseline exists at fixed seed, R2 becomes a verifiable Tier A-style
  equivalence change (byte-identical flattened obs), and further hot-path wins (e.g. the per-step
  `memmove` shift in `PreStep`, the per-frame `memset` zeroing) can be measured properly.
