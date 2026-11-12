# RLGymSim_CPP
A lightning-fast C++ implementation of [rlgym_sim](https://github.com/AechPro/rocket-league-gym-sim)

## Dependencies 
 - RocketSim (as a submodule)

## RocketSim submodule

The `RocketSim/` submodule tracks the `cpp-standalone` branch of
[justoboy/RocketSim](https://github.com/justoboy/RocketSim) — the C++-only distribution of the
custom RocketSim fork (SPIKE_RUSH / GRIDIRON / DROPSHOT support). Its `src/` and `libsrc/` are
byte-identical to the fork's `main`; only the Python binding layer is stripped and the
`CMakeLists.txt` builds a plain static library. Simulation changes are made on the fork's `main`
first, then synced onto `cpp-standalone` (see `RocketSim/BRANCHES.md`).

