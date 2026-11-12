# Phase 1 — RLGymSim_CPP: sim + obs/reward port (standalone-testable)

> ## ⚠ STATUS
> Approved 2026-11-11 (part of the L1 migration tracked in
> `D:/RLBotTraining/PERF_PROPOSALS.md`). This repo is the **sim half** of the migration and is
> fully testable WITHOUT the learner: build the Gym, step arenas, diff obs against the Python
> stack. Do not touch `RLGymPPO_CPP` until this plan is done and gated.

## Goal
Make this repo produce, in pure C++, exactly what the Python stack
(`custombot` + rlgym-fork + RocketSim-fork) produces today:
1. The two custom game modes (`SPIKE_RUSH`, `GRIDIRON`) inside the bundled `RocketSim/` submodule.
2. custombot's obs builders, reward functions, terminal conditions, state setters, and action
   parsers ported onto this repo's `OBSBuilder`/`RewardFunction`/`TerminalCondition`/`StateSetter`/
   `ActionParser` interfaces.

## Source of truth (read these first)
- Game-mode mechanic (C++, the fork's own source — transplant target is `RocketSim/src/Sim/`):
  - [`rocketsim/src/Sim/Ball/Ball.cpp`](../../../rocketsim/src/Sim/Ball/Ball.cpp) + `Ball.h`
    — `AttachInfo` (attachedCarId, localOffset, active, engageTimer, releaseCooldown),
    `_OnAttach` engage/steal, release-on-jump/powerup, kinematic weld follow
  - [`rocketsim/src/Sim/Arena/Arena.cpp`](../../../../rocketsim/src/Sim/Arena/Arena.cpp)
    — collision routing to `_OnAttach`, bump steal (SPIKE_RUSH demo+transfer) / fumble (GRIDIRON
    free ball), carrier boost lock, GRIDIRON kickoff-after-goal handoff (conceding team gets ball)
  - [`rocketsim/src/RLConst.h`](../../../../rocketsim/src/RLConst.h)
    — `ATTACH_INVULN_TIME`, `SpikeRush::{ACTIVATION_DELAY, ATTACH_RADIUS, RELEASE_COOLDOWN, ...}`,
    `Gridiron::{ROOF_LOCAL_OFFSET, REACQUIRE_COOLDOWN, ...}`, GRIDIRON spawn/respawn tables
  - `rocketsim/src/Sim/GameMode.h` (enum), `CarControls.h` (`powerup` button),
    `MutatorConfig.cpp` (GRIDIRON dropshot-style boost recharge when not carrying)
- Obs/reward/terminal/action semantics (Python, port to C++ interfaces here):
  - `custombot/custombot/obs.py` (32KB — the pad-to-max 8-slot entity obs + 13-frame temporal
    builder producing the `(q, kv, mask)` tuple; the tuple's element layout is the Phase-2 contract)
  - `custombot/custombot/reward.py`, `reward_terms.py`, `reward_geometry.py`
  - `custombot/custombot/conditions.py` (terminal), `mode_setters.py` (state setters),
    `custombot/custombot/constants.py` (BIN_CENTERS etc.), `shared_info.py`, `scoreboard.py`
- Interfaces to implement against (this repo):
  `src/RLGymSim_CPP/Utils/{OBSBuilders,RewardFunctions,TerminalConditions,StateSetters,ActionParsers}/*.h`,
  `src/RLGymSim_CPP/Envs/Match.{h,cpp}`, `src/RLGymSim_CPP/Gym.{h,cpp}`

## Design decisions already made (do not re-litigate)
- Obs stays **pad-to-max fixed 8 slots**; the temporal builder emits a `(q, kv, mask)` tuple —
  `Match::BuildObservations`/`FList2` must carry that tuple through (extend the FList2/GameState
  plumbing or introduce a typed obs variant; Phase 2 consumes it verbatim).
- `ActionParser`: custombot's CURRENT actor is **continuous** — the (9,) control vector passes
  through `decode_controls` (clip cols 0..4 to [-1,1]; threshold cols 5..8 at 0) and repeats
  `repeats` times into the engine block (see `ContinuousActionParser` in
  `custombot/custombot/agent.py`). Port a continuous action parser; the repo's existing
  `DiscreteAction.cpp` is still under testing to see if it outperforms the continuous model and might be worth porting as well.
- Player-count is fixed per env (Match already errors on mid-episode player-count change).

## Tasks (execution order)
1. [ ] Port `SPIKE_RUSH`/`GRIDIRON` into `RocketSim/` submodule (this fork's RocketSim copy).
       Keep the fork's Python-binding (`python-mtheall`) OUT — this repo needs C++ only.
2. [ ] Extend `GameState`/`PlayerData` to carry attach-info fields the obs builders need
       (attachedCarId, active, etc.).
3. [ ] Port obs builders → `OBSBuilder` subclass producing the `(q, kv, mask)` tuple.
4. [ ] Port reward stack → `RewardFunction`/`CombinedReward`/`ZeroSumReward` equivalents.
5. [ ] Port terminal conditions + state setters + discrete action parser.
6. [ ] Equivalence gate (the whole point): fixed-seed scenario script that steps the Python stack
       and this C++ Gym on identical initial states; assert obs arrays match within float tolerance
       and goals/demos/terminations match exactly. Include at least: soccar kickoff, spike_rush
       pre-arm window (pre-activation hits behave like soccar), engage, steal-with-demo, release,
       gridiron fumble + kickoff handoff.
7. [ ] Merge into `RLGymPPO_CPP`'s vendored `RLGymSim_CPP` copy (plain directory, not submodule)
       OR repoint RLGymPPO_CPP's CMake at this fork. Decide at merge time; do not pre-decide.

## Non-goals
- No learner/torch work (Phase 2 / RLGymPPO_CPP repo).
- No changes to `custombot/` (gated to the other agent) or to the Python `rlgym/`/`rocketsim/`
  forks beyond reading them as spec.

## Risks
- Numerics: bullet3 version drift between the fork and the bundled RocketSim → pin the submodule
  to the same bullet3 (3.24) and diff a soccar arena first, before touching custom modes.
- Obs tuple plumbing through `FList2` is the ugliest mechanical edit; keep the Python obs as the
  golden reference, never "improve" it during the port.

## Validation
- `cmake --build` clean on MSVC; the scenario runner binary steps N ticks and prints obs hashes.
- Gate passes: |C++ obs − Python obs| < 1e-5 elementwise on ≥3 seeded matches per mode.
