#!/usr/bin/env python
"""Cross-language obs-equivalence gate (Python half).

Parses the SAME hand-authored, RNG-free fixture files the C++ ParityGate consumes, builds rlgym
GameState/Car/PhysicsObject objects, drives the REAL reference TemporalObsBuilder + ProfileTracker
with the identical frame-0-direct / frame-N-after-step semantics, flattens each agent's obs to
q ++ kv ++ m, and dumps one line per (frame, agent) to <outDir>/<name>.py.txt.

Then diffs each Python dump against the matching C++ dump elementwise; the gate passes iff every
element agrees within float tolerance (default 1e-5). This is the contract check: BC trains on
obs.py-built obs from replays and RL fine-tunes the SAME checkpoint in the C++ env, so the two
builders must be elementwise identical or the BC checkpoint is out-of-distribution at RL step 0.

Usage:
    python scripts/parity/run_parity.py <fixturesDir> <cppOutDir> [--tol 1e-5]

Expects <cppOutDir>/<name>.cpp.txt (produced by the ParityGate binary). Writes <name>.py.txt next
to the C++ dumps and prints a per-fixture PASS/FAIL with the max absolute diff.
"""
import os
import sys
import glob

import numpy as np

# Make the reference package importable when run from the C++ repo.
_REFERENCE = os.environ.get("REFERENCE_SRC")
if _REFERENCE:
    sys.path.insert(0, _REFERENCE)

from reference.bot_spec import BotSpec
from reference.constants import as_holder, BLUE_TEAM, ORANGE_TEAM
from reference.obs import TemporalObsBuilder as PyTemporalObsBuilder
from reference.profile_tracker import ProfileTracker, PROFILES_KEY
from reference.scoreboard import Scoreboard, SCORE_KEY, TICKS_KEY
from rlgym.rocket_league.api import GameState as PyGameState
from rlgym.rocket_league.api import Car as PyCar
from rlgym.rocket_league.api import PhysicsObject as PyPhysicsObject


class CarRow:
    __slots__ = ("team", "pos", "vel", "ang", "fw", "up",
                 "boost", "demoed", "onground", "hasflip", "hasjump", "balltouch", "bumped")


class Frame:
    __slots__ = ("ball", "cars")


class Fixture:
    pass


def _split(line):
    h = line.find("#")
    if h != -1:
        line = line[:h]
    return line.split()


def parse_fixture(path):
    fx = Fixture()
    fx.mode = 0
    fx.tick_skip = 8
    fx.n_active_pads = 0
    fx.blue_score = 0
    fx.orange_score = 0
    fx.ticks_left = 0.0
    fx.is_overtime = 0
    fx.finished = 0
    fx.mut = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fx.hs = [0.0, 0.0, 0.0]
    fx.charge = 0
    fx.pad_timers = []
    fx.tiles = None
    fx.frames = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            t = _split(raw)
            if not t:
                continue
            k = t[0]
            num = lambda i: float(t[i])
            if k == "MODE":
                fx.mode = int(t[1])
            elif k == "TICK_SKIP":
                fx.tick_skip = int(t[1])
            elif k == "N_ACTIVE_PADS":
                fx.n_active_pads = int(t[1])
            elif k == "BLUE_SCORE":
                fx.blue_score = int(t[1])
            elif k == "ORANGE_SCORE":
                fx.orange_score = int(t[1])
            elif k == "TICKS_LEFT":
                fx.ticks_left = float(t[1])
            elif k == "IS_OVERTIME":
                fx.is_overtime = int(t[1])
            elif k == "FINISHED":
                fx.finished = int(t[1])
            elif k == "MUT":
                fx.mut = [num(1 + i) for i in range(6)]
            elif k == "HS":
                fx.hs = [num(1 + i) for i in range(3)]
            elif k == "CHARGE":
                fx.charge = int(t[1])
            elif k == "PAD_TIMERS":
                fx.pad_timers = [float(x) for x in t[1:]]
            elif k == "TILES":
                fx.tiles = np.array([float(t[1 + i]) for i in range(140)], dtype=np.float32)
            elif k == "FRAME":
                fr = Frame()
                fr.cars = []
                fr.ball = None
                fx.frames.append(fr)
            elif k == "BALL":
                fx.frames[-1].ball = np.array([num(1 + i) for i in range(9)], dtype=np.float32)
            elif k == "CAR":
                c = CarRow()
                c.team = int(t[1])
                c.pos = [num(2 + i) for i in range(3)]
                c.vel = [num(5 + i) for i in range(3)]
                c.ang = [num(8 + i) for i in range(3)]
                c.fw = [num(11 + i) for i in range(3)]
                c.up = [num(14 + i) for i in range(3)]
                c.boost = num(17)
                c.demoed = num(18)
                c.onground = num(19)
                c.hasflip = num(20)
                c.hasjump = num(21)
                c.balltouch = num(22)
                c.bumped = num(23)
                fx.frames[-1].cars.append(c)
    return fx


def build_state(fx, fr):
    """Construct a rlgym GameState from a fixture frame, cars in agent order (slot 1+i)."""
    gs = PyGameState()
    ball = PyPhysicsObject()
    ball.position = np.array(fr.ball[0:3], dtype=np.float32)
    ball.linear_velocity = np.array(fr.ball[3:6], dtype=np.float32)
    ball.angular_velocity = np.array(fr.ball[6:9], dtype=np.float32)
    # heatseeker / dropshot ball info (mode-gated inside obs.py)
    ball.heatseeker_target_dir = fx.hs[0]
    ball.heatseeker_target_speed = fx.hs[1]
    ball.heatseeker_time_since_hit = fx.hs[2]
    ball.dropshot_charge_level = fx.charge
    gs.ball = ball
    gs.game_mode = fx.mode
    gs.boost_pad_timers = np.array(fx.pad_timers, dtype=np.float32)
    gs.boost_pad_positions = _boost_positions(fx.n_active_pads)
    gs.dropshot_tile_states = fx.tiles.reshape(2, 70) if fx.tiles is not None else None
    gs.tick_count = 0
    gs.goal_scored = False
    gs.scoring_team = None

    cars = {}
    for i, c in enumerate(fr.cars):
        ph = PyPhysicsObject()
        ph.position = np.array(c.pos, dtype=np.float32)
        ph.linear_velocity = np.array(c.vel, dtype=np.float32)
        ph.angular_velocity = np.array(c.ang, dtype=np.float32)
        fwd = np.array(c.fw, dtype=np.float32)
        up = np.array(c.up, dtype=np.float32)
        left = np.cross(up, fwd)
        ph.rotation_mtx = np.stack([fwd, left, up], axis=-1).astype(np.float32)
        car = PyCar()
        car.team_num = c.team
        car.physics = ph
        car.boost_amount = float(c.boost)
        car.demo_respawn_timer = 1.0 if c.demoed != 0 else 0.0
        car.on_ground = bool(c.onground)
        car.has_flipped = bool(c.hasflip)
        car.has_jumped = bool(c.hasjump)
        car.ball_touches = int(c.balltouch)
        car.bump_victim_id = "bumped" if c.bumped != 0 else None
        # agent id string encodes team (obs.py derives team from the id prefix) and a stable slot id
        agent_id = ("orange" if c.team == ORANGE_TEAM else "blue") + f"-{i}"
        cars[agent_id] = car
    gs.cars = cars
    return gs


from reference.constants import BOOST_LOCATIONS  # noqa: E402


def _boost_positions(n_active):
    return np.array(BOOST_LOCATIONS[:n_active], dtype=np.float32)


def run_fixture(in_path, out_path):
    fx = parse_fixture(in_path)
    holder = as_holder(fx.mode)
    holder.gravity = fx.mut[0]
    holder.boost_consumption = fx.mut[1]
    holder.boost_strength = fx.mut[2]
    holder.ball_max_speed = fx.mut[3]
    holder.ball_size = fx.mut[4]
    holder.ball_bounciness = fx.mut[5]

    spec = BotSpec(name="parity", profile=True, n_frames=13)
    obs = PyTemporalObsBuilder(spec, n_players=2 * 4, tick_skip=fx.tick_skip, mode=holder)
    tracker = ProfileTracker(tick_skip=fx.tick_skip)

    state0 = build_state(fx, fx.frames[0])
    agents = list(state0.cars.keys())

    shared_info = {}
    shared_info[SCORE_KEY] = (fx.blue_score, fx.orange_score)
    shared_info[TICKS_KEY] = float(fx.ticks_left)
    # Seed neutral profiles (mirrors C++ Reset seeding wasDemoed/wasBumped from frame 0).
    shared_info = tracker.set_state(agents, state0, shared_info)
    obs.reset(agents, state0, shared_info)

    lines = []

    def emit(obs_map):
        for agent in agents:
            q, kv, m = obs_map[agent]
            flat = np.concatenate([q.reshape(-1), kv.reshape(-1), m.reshape(-1)])
            lines.append(" ".join(f"{v:.9g}" for v in flat))

    # frame 0: read the broadcast buffer directly (neutral profiles); shift is a no-op (all frames equal)
    emit(obs.build_obs(agents, state0, shared_info))
    # frames 1..N-1: advance profiles with the new state, shift+encode, then read
    for f in range(1, len(fx.frames)):
        sf = build_state(fx, fx.frames[f])
        agents = list(sf.cars.keys())
        shared_info[SCORE_KEY] = (fx.blue_score, fx.orange_score)
        shared_info[TICKS_KEY] = float(fx.ticks_left)
        shared_info = tracker.step(agents, sf, shared_info)
        emit(obs.build_obs(agents, sf, shared_info))

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    return len(fx.frames), len(fx.frames[0].cars)


def main():
    if len(sys.argv) < 3:
        print("usage: run_parity.py <fixturesDir> <cppOutDir> [--tol 1e-5]", file=sys.stderr)
        return 1
    fixtures_dir = sys.argv[1]
    cpp_dir = sys.argv[2]
    tol = 1e-5
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])

    files = sorted(glob.glob(os.path.join(fixtures_dir, "*.txt")))
    print(f"=== run_parity: {len(files)} fixture(s) ===")
    all_pass = True
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        py_path = os.path.join(cpp_dir, name + ".py.txt")
        cpp_path = os.path.join(cpp_dir, name + ".cpp.txt")
        nf, na = run_fixture(f, py_path)
        if not os.path.exists(cpp_path):
            print(f"  [MISS] {name}: no C++ dump at {cpp_path}")
            all_pass = False
            continue
        py = np.loadtxt(py_path, dtype=np.float64)
        cpp = np.loadtxt(cpp_path, dtype=np.float64)
        if py.shape != cpp.shape:
            print(f"  [FAIL] {name}: shape mismatch py={py.shape} cpp={cpp.shape}")
            all_pass = False
            continue
        diff = np.abs(py - cpp)
        max_diff = float(diff.max()) if diff.size else 0.0
        ok = max_diff < tol
        all_pass = all_pass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} "
              f"({nf} frames x {na} agents) max|diff|={max_diff:.3e} tol={tol:.1e}")
    print(f"=== run_parity {'PASS' if all_pass else 'FAIL'} ===")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
