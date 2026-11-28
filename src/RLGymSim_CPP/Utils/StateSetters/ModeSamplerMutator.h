#pragma once
#include "StateSetter.h"
#include <random>

// State setters ported from the reference mode_setters.py:
// TeamConfigMutator (active-set spawner + matchup sampler + optional spawn-curriculum override),
// ModeSamplerMutator (per-episode mode sampler).
// NOTE: uses RocketSim::GameMode (via Framework.h's `using namespace RocketSim`).

namespace RLGSC {

	// Mutable holder for the active per-episode team config (n_blue, n_orange).
	struct TeamConfigHolder {
		int n_blue = 1;
		int n_orange = 1;
		TeamConfigHolder(int nb = 1, int no = 1) : n_blue(nb), n_orange(no) {}
	};

	// Mutable holder for the active mode.
	struct ModeHolder {
		GameMode mode = GameMode::SOCCAR;
	};

	// ---- Data-driven drill curriculum (replaces the old fixed two-wheel WheelHolder) ----
	//
	// The driver owns the curriculum's phase state machine and advances it per EPOCH, then pushes the
	// resolved drill specs onto every live env via Learner::SetDrillConfig. The mutator evaluates the
	// spec list PER EPISODE (the C++ learner auto-resets games internally, so the per-episode decision
	// lives here, not in the driver): the FIRST enabled spec spawns the episode; none enabled => native
	// kickoff. The C++ side is intentionally dumb: it never interpolates difficulty, never rolls fire
	// probabilities (always-drill: an enabled spec fires), and never knows what a "phase" is. Geometry
	// ranges, thresholds, and phase ordering are all driver/JSON concerns.
	enum class DrillSpawnType : int {
		NATIVE_KICKOFF = 0,  // arena->ResetToRandomKickoff (the duel phase uses this with a clock cap)
		TOUCH = 1,           // ball at center; each car at car_dist along the kickoff axis facing it
		GOAL_BREAKAWAY = 2,  // 1v0: ball at +ball_y on the lone car's attack axis, car car_dist behind it
	};

	struct DrillSpec {
		DrillSpawnType spawn_type = DrillSpawnType::NATIVE_KICKOFF;
		bool  enabled     = false;   // false => skipped this epoch (the driver gates phases here)
		int   n_blue      = -1;      // lineup override; -1 => keep the sampled lineup
		int   n_orange    = -1;
		float car_dist    = 400.0f;  // TOUCH: car->ball gap; GOAL_BREAKAWAY: car stand-off behind ball
		                             // (each drill owns its OWN distance param — no cross-drill sharing)
		float ball_y      = 0.0f;    // GOAL_BREAKAWAY: ball |y| on the lone car's attack axis
		float clock_cap   = -1.0f;   // >= 0: per-episode clock cap (s); < 0 => holder native_time
		bool  hard_finish = false;   // capped native-type episode hard-stops on expiry (no OT)
	};

	struct DrillHolder {
		std::vector<DrillSpec> drills;  // evaluated in order; first enabled wins
		float native_time     = 300.0f; // clock cap (s) on a native (non-drill) episode
		float native_fraction = 0.0f;   // fraction of drill-phase episodes that use native kickoff
		                                // instead of the active drill (0 = always drill; the default)
	};

	// Initial car speed (uu/s) aimed at the ball on custom spawns.
	inline constexpr float SPAWN_APPROACH_SPEED = 1000.0f;

	// TeamConfigMutator: per-episode active-set spawner + matchup sampler + data-driven drill override.
	//
	// The C++ learner auto-resets each game internally (the driver only sees per-EPOCH aggregate reports),
	// so the per-episode spawn decision lives HERE in the mutator, not in the driver: on every reset it
	// (1) samples one (n_blue, n_orange) from the weighted pool (the native / touch-drill lineup),
	// (2) picks the FIRST enabled DrillSpec in the holder (always-drill: an enabled spec fires; none
	//     enabled => native kickoff), honoring the holder's native_fraction escape hatch,
	// (3) applies the spawn: NATIVE_KICKOFF leaves the sampled lineup + random kickoff intact; TOUCH keeps
	//     the lineup and places each car at car_dist from the center ball facing it; GOAL_BREAKAWAY forces
	//     the spec's lineup (1v0) and writes the ball at +ball_y on the lone car's attack axis, car
	//     car_dist behind the ball facing it (the car distance is the spec's OWN param — no cross-drill
	//     sharing),
	// (4) stamps the per-episode clock cap (spec clock_cap, else holder native_time) + drillActive flag
	//     (drillActive for TOUCH/GOAL so goals respawn instead of terminating; hard_finish for a capped
	//     native-type episode, e.g. the duel, so it ends on expiry with no overtime),
	// (5) stamps the per-episode owner map into GameState::slotOwners.
	//
	// A single-bot self-play run is the degenerate case: one member, siblingFraction 0, every slot owned by
	// member 0; the goal drill's lone car is BLUE (attacks +y), the touch drill keeps the sampled matchup.
	class TeamConfigMutator : public StateSetter {
	public:
		// Weighted pool of (n_blue, n_orange) team configs (mirrors DEFAULT_TEAM_CONFIGS/WEIGHTS).
		std::vector<std::pair<int, int>> teamConfigs;
		std::vector<float> teamWeights;
		// Number of league members (bots). 1 => pure self-play (every slot owned by member 0).
		int memberCount = 1;
		// Fraction of episodes that draw a mixed-lineage (sibling) matchup vs. all-slots-one-bot self.
		float siblingFraction = 0.0f;
		// Drill curriculum holder (nullptr => always native kickoff).
		DrillHolder* drill = nullptr;
		std::mt19937 rng;
		// The spec used on the most recent ResetState (nullptr = native kickoff); RespawnAfterGoal
		// re-applies the same spawn so a mid-episode drill goal returns the ball to the drill spawn
		// instead of resting in the net.
		const DrillSpec* lastSpec = nullptr;

		TeamConfigMutator(const std::vector<std::pair<int, int>>& configs,
		                  const std::vector<float>& weights,
		                  int memberCount, float siblingFraction, uint32_t seed,
		                  DrillHolder* drill = nullptr)
			: teamConfigs(configs), teamWeights(weights),
			  memberCount(memberCount), siblingFraction(siblingFraction), drill(drill), rng(seed) {
			assert(teamConfigs.size() == teamWeights.size());
			assert(memberCount >= 1);
		}

		virtual GameState ResetState(Arena* arena) override {
			// 1. Sample one (n_blue, n_orange) from the weighted pool (the native / touch-drill lineup).
			std::discrete_distribution<int> cfgDist(teamWeights.begin(), teamWeights.end());
			auto [nBlue, nOrange] = teamConfigs[cfgDist(rng)];

			// 2. Pick the active spec: the FIRST enabled one, unless the holder's native_fraction roll
			//    (default 0 => always drill) escapes to a native kickoff this episode.
			const DrillSpec* spec = nullptr;
			if (drill) {
				for (const auto& s : drill->drills)
					if (s.enabled) { spec = &s; break; }
				if (spec && drill->native_fraction > 0.0f
					&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < drill->native_fraction)
					spec = nullptr;
			}

			// 3. Lineup override from the spec (e.g. the goal breakaway forces 1v0); -1 keeps the sample.
			if (spec) {
				if (spec->n_blue   >= 0) nBlue   = spec->n_blue;
				if (spec->n_orange >= 0) nOrange = spec->n_orange;
			}
			int n = nBlue + nOrange;

			// 4. Build the per-slot owner list (blue-first slot order). Self-play: every slot -> member 0.
			//    Sibling: cycle member ids across the n slots then shuffle (mirrors League.draw_sibling).
			std::vector<int> botIds(n, 0);
			bool sibling = memberCount > 1
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < siblingFraction;
			if (sibling) {
				for (int j = 0; j < n; j++)
					botIds[j] = j % memberCount;   // cycle member ids as evenly as possible
				std::shuffle(botIds.begin(), botIds.end(), rng);
			} // else: self-play, botIds stays all-zero.

			// 5. The Arena is built FRESH and EMPTY on every Reset() (see Gym::Reset), so we only ADD this
			//    episode's active set: blue cars first (ascending id), then orange (ascending id), so the
			//    deterministic (team,id) sort in GameState::UpdateFromArena reproduces blue-first slot order.
			for (int i = 0; i < nBlue; i++)
				arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE);
			for (int i = 0; i < nOrange; i++)
				arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);

			// 6. Native random kickoff repositions the now-existing cars + ball (overridden below on a
			//    TOUCH/GOAL spec; left intact for NATIVE_KICKOFF specs like the duel and for native
			//    episodes). It also sets each car's boost to the engine's kickoff default.
			arena->ResetToRandomKickoff();

			bool drillActive = false;
			if (spec && spec->spawn_type != DrillSpawnType::NATIVE_KICKOFF) {
				ApplyDrillSpawn(arena, *spec);
				drillActive = true;
			}
			lastSpec = drillActive ? spec : nullptr;

			// 7. Capture the state, stamp the per-episode owner map + clock cap + hard-finish flag.
			//    drill (TOUCH/GOAL) -> spec clock cap (drillActive already forces the hard-stop on expiry);
			//    capped native-type spec (the duel) -> hard_finish, no OT, goals respawn to kickoff;
			//    uncapped native -> full match with OT on a tie.
			GameState state(arena);
			state.slotOwners = botIds;
			state.drillActive = drillActive;
			if (spec && spec->clock_cap >= 0.0f) {
				state.maxTimeSeconds = spec->clock_cap;
				state.hardFinish = !drillActive && spec->hard_finish;
			}
			else {
				state.maxTimeSeconds = drill ? drill->native_time : CommonValues::kMaxTimeSeconds;
				state.hardFinish = false;
			}
			return state;
		}

		// Mid-episode respawn after a drill goal: re-apply the same spawn (ball + cars) so the ball leaves
		// the net. GameState::UpdateFromArena detects goals by ball POSITION, so a ball resting in the net
		// would re-increment scoreLine every tick; returning it to the drill spawn both fixes that and lets
		// the curriculum observe repeated goal events. Boost is RESET to the engine's kickoff value on
		// respawn (see ApplyDrillSpawn), matching a live kickoff: preserving the car's drained boost would
		// create a train/inference mismatch. No-op when the last reset was not a drill (a native kickoff
		// goal is respawned by Gym::Step).
		virtual void RespawnAfterGoal(Arena* arena) override {
			if (!lastSpec)
				return;
			ApplyDrillSpawn(arena, *lastSpec);
		}

	private:
		// Shared drill spawn geometry, used by ResetState and RespawnAfterGoal. Cars spawn with the
		// engine's kickoff boost, so a fresh spawn and a mid-episode respawn both reset boost.
		void ApplyDrillSpawn(Arena* arena, const DrillSpec& spec) {
			if (spec.spawn_type == DrillSpawnType::GOAL_BREAKAWAY) {
				// GOAL drill: ball on the lone (BLUE) car's attack axis at +ball_y, still; car behind the
				// ball (smaller y) facing +y toward it at the spec's OWN car_dist. The opponent's absence
				// (spec lineup 1v0) is the binary difficulty.
				float by = spec.ball_y;
				BallState bs = {};
				bs.pos = Vec(0.0f, by, CommonValues::BALL_RADIUS);
				bs.vel = Vec(0.0f, 0.0f, 0.0f);
				bs.angVel = Vec(0.0f, 0.0f, 0.0f);
				arena->ball->SetState(bs);

				Car* car = arena->_cars.empty() ? nullptr : *arena->_cars.begin();
				if (car) {
					Vec fwd(0.0f, 1.0f, 0.0f);                 // lone BLUE car attacks +y (toward the ball)
					Vec up(0.0f, 0.0f, 1.0f);
					Vec right = up.Cross(fwd);
					RotMat rot(fwd, right, up);
					CarState cs = {};
					cs.pos = Vec(0.0f, by - spec.car_dist, RLConst::CAR_SPAWN_REST_Z); // behind the ball
					cs.vel = fwd * SPAWN_APPROACH_SPEED;
					cs.angVel = Vec(0.0f, 0.0f, 0.0f);
					cs.rotMat = rot;
					cs.boost = arena->GetMutatorConfig().carSpawnBoostAmount; // kickoff boost, read from engine
					car->SetState(cs);
				}
			} else if (spec.spawn_type == DrillSpawnType::TOUCH) {
				// TOUCH drill: ball at center, still; each car at spec.car_dist along the kickoff axis
				// facing it.
				BallState bs = {};
				bs.pos = Vec(0.0f, 0.0f, CommonValues::BALL_RADIUS);
				bs.vel = Vec(0.0f, 0.0f, 0.0f);
				bs.angVel = Vec(0.0f, 0.0f, 0.0f);
				arena->ball->SetState(bs);

				for (Car* car : arena->_cars) {
					bool blue = (car->team == Team::BLUE);
					float sgn = blue ? -1.0f : 1.0f;        // blue defends -y (spawns at -d, faces +y)
					Vec fwd(0.0f, -sgn, 0.0f);              // unit vector from car toward center ball
					Vec up(0.0f, 0.0f, 1.0f);
					Vec right = up.Cross(fwd);
					RotMat rot(fwd, right, up);
					CarState cs = {};
					cs.pos = Vec(0.0f, sgn * spec.car_dist, RLConst::CAR_SPAWN_REST_Z);
					cs.vel = fwd * SPAWN_APPROACH_SPEED;
					cs.angVel = Vec(0.0f, 0.0f, 0.0f);
					cs.rotMat = rot;
					cs.boost = arena->GetMutatorConfig().carSpawnBoostAmount; // kickoff boost, read from engine
					car->SetState(cs);
				}
			}
		}
	};

	// ModeSamplerMutator: per-episode mode sampler. The arena's game mode is fixed at
	// Arena::Create() time and cannot be switched at runtime, so the sampled mode is
	// stored in the ModeHolder for the caller to construct the next episode's Gym with.
	class ModeSamplerMutator : public StateSetter {
	public:
		ModeHolder* modeHolder;
		std::vector<GameMode> modes;
		std::vector<float> weights;
		std::mt19937 rng;

		ModeSamplerMutator(ModeHolder* holder,
		                   const std::vector<GameMode>& modes,
		                   const std::vector<float>& weights,
		                   uint32_t seed = 0)
			: modeHolder(holder), modes(modes), weights(weights), rng(seed) {
			assert(modes.size() == weights.size());
		}

		virtual GameState ResetState(Arena* arena) override {
			// Sample a mode with the configured weights
			std::discrete_distribution<int> dist(weights.begin(), weights.end());
			int idx = dist(rng);
			modeHolder->mode = modes[idx];

			return GameState(arena);
		}
	};

} // namespace RLGSC
