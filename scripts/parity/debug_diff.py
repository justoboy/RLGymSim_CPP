import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("REFERENCE_SRC", r"D:\RLBotTraining\reference"))

from reference.bot_spec import BotSpec
from reference.constants import as_holder, BLUE_TEAM, ORANGE_TEAM, ENTITY_FEATURES
from reference.obs import TemporalObsBuilder as PyTemporalObsBuilder
from reference.profile_tracker import ProfileTracker, PROFILES_KEY
from reference.scoreboard import SCORE_KEY, TICKS_KEY

# reuse the harness builders
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_parity import parse_fixture, build_state

name = "soccar"
fx = parse_fixture(os.path.join("fixtures", name + ".txt"))
holder = as_holder(fx.mode)
holder.gravity, holder.boost_consumption, holder.boost_strength = fx.mut[0], fx.mut[1], fx.mut[2]
holder.ball_max_speed, holder.ball_size, holder.ball_bounciness = fx.mut[3], fx.mut[4], fx.mut[5]
spec = BotSpec(name="parity", profile=True, n_frames=13)
obs = PyTemporalObsBuilder(spec, n_players=8, tick_skip=fx.tick_skip, mode=holder)
tracker = ProfileTracker(tick_skip=fx.tick_skip)
state0 = build_state(fx, fx.frames[0])
agents = list(state0.cars.keys())
shared = {SCORE_KEY: (fx.blue_score, fx.orange_score), TICKS_KEY: float(fx.ticks_left)}
shared = tracker.set_state(agents, state0, shared)
obs.reset(agents, state0, shared)

lines = []
def emit(om):
    for a in agents:
        q, kv, m = om[a]
        lines.append(np.concatenate([q.reshape(-1), kv.reshape(-1), m.reshape(-1)]))
emit(obs.build_obs(agents, state0, shared))
for f in range(1, len(fx.frames)):
    sf = build_state(fx, fx.frames[f])
    agents = list(sf.cars.keys())
    shared[SCORE_KEY] = (fx.blue_score, fx.orange_score)
    shared[TICKS_KEY] = float(fx.ticks_left)
    shared = tracker.step(agents, sf, shared)
    emit(obs.build_obs(agents, sf, shared))
py = np.array(lines)

cpp = np.loadtxt(os.path.join("build", "parity_out", name + ".cpp.txt"), dtype=np.float64)
print("py", py.shape, "cpp", cpp.shape)
diff = np.abs(py - cpp)
flat_idx = int(np.argmax(diff))
print("argmax flat idx", flat_idx, "diff", diff.ravel()[flat_idx])
# decode: q is [0:208], kv [208: 208+151*69], m after
Q = 208
if flat_idx < Q:
    print("QUERY col", flat_idx)
else:
    r = flat_idx - Q
    tok = r // 69
    col = r % 69
    print(f"KV token {tok} col {col}")
    # token -> frame, entity
    nbc = 1 + 8
    if tok < 13 * nbc:
        frame = tok // nbc; ent = tok % nbc
        print(f"  entity {ent} (0=ball,1..8=car) col {col}")
    else:
        print(f"  pad {tok - 13*nbc}")
# show values at that index across all rows
print("py vals", py.ravel()[flat_idx])
print("cpp vals", cpp.ravel()[flat_idx])
# also print a few top diffs
order = np.argsort(diff.ravel())[::-1][:12]
for oi in order:
    fi = int(oi)
    if fi < Q:
        dec = f"QUERY col {fi}"
    else:
        r = fi - Q; tok = r//69; col = r%69
        nbc = 9
        if tok < 13*nbc:
            dec = f"KV ent {tok%nbc} col {col} (frame {tok//nbc})"
        else:
            dec = f"PAD {tok-13*nbc} col {col}"
    print(f"flat {fi}: diff {diff.ravel()[oi]:.4e} py={py.ravel()[oi]:.5f} cpp={cpp.ravel()[oi]:.5f} {dec}")
