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

	// Mutable holder for the TWO-WHEEL spawn curriculum (read by TeamConfigMutator on every reset).
	// Two INDEPENDENT wheels (touch + goal), each with its own state machine living in the driver. The
	// driver advances each wheel's streak/score per EPOCH and writes the resulting per-wheel geometry +
	// fire-probability here; the mutator then rolls each drill's Bernoulli PER EPISODE (the C++ learner
	// auto-resets games internally, so the per-episode decision lives in the mutator, not the driver).
	//   touch_enabled/touch_d/touch_prob: touch wheel ON, car->ball spawn distance, per-episode fire prob.
	//   goal_enabled/goal_ball_y/goal_prob: goal wheel ON, ball |y| on the lone car's attack axis, fire prob.
	//   drill_time: episode clock cap (seconds) while any drill is active (short = cheap collection).
	// When a wheel is OFF (enabled=false) the mutator never fires that drill (native kickoff on that axis).
	struct WheelHolder {
		bool  touch_enabled = false;
		float touch_d       = 400.0f;
		float touch_prob    = 0.5f;
		bool  goal_enabled  = false;
		float goal_ball_y   = 0.0f;
		float goal_prob     = 0.5f;
		float drill_time    = 10.0f;  // clock cap (s) while a drill is active
		float native_time   = 300.0f; // clock cap (s) on a native (non-drill) episode
	};

	// Custom-spawn geometry (mirrors the reference scaffold constants).
	inline constexpr float SPAWN_MIN_DIST = 400.0f;     // ball basically on the car (touch g=1)
	inline constexpr float SPAWN_MAX_DIST = 4000.0f;    // real kickoff gap (touch g=0)
	inline constexpr float SPAWN_APPROACH_SPEED = 1000.0f; // initial car speed (uu/s) aimed at the ball

	// TeamConfigMutator: per-episode active-set spawner + matchup sampler + two-wheel drill override.
	//
	// Faithful port of the reference TeamConfigMutator + SpawnCurriculumMutator. The C++ learner auto-resets
	// each game internally (the driver only sees per-EPOCH aggregate reports), so the per-episode drill
	// Bernoulli lives HERE in the mutator, not in the driver: on every reset it
	// (1) samples one (n_blue, n_orange) from the weighted pool (the native/touch lineup),
	// (2) rolls each wheel's per-episode Bernoulli from the holder's prob,
	// (3) on a GOAL drill forces a 1v0 lineup (single BLUE car) and writes the ball at +goal_ball_y on the
	//     lone car's attack axis, car behind the ball facing it; on a TOUCH drill keeps the sampled lineup
	//     and places each car at distance d from the center ball facing it; otherwise native kickoff,
	// (4) stamps the per-episode clock cap (drill_time on a drill, native_time otherwise) + drillActive flag,
	// (5) stamps the per-episode owner map into GameState::slotOwners.
	//
	// The driver advances each wheel's streak/score state machine per EPOCH and writes the resulting geometry
	// + fire-probability into the shared WheelHolder (SetWheelConfig); the mutator reads that holder here.
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
		// Two-wheel curriculum holder (nullptr => always native kickoff).
		WheelHolder* wheel = nullptr;
		std::mt19937 rng;
		// Which drill fired on the most recent ResetState; RespawnAfterGoal re-applies the same spawn so a
		// mid-episode drill goal returns the ball to the drill spawn instead of resting in the net.
		bool lastGoalFire = false;
		bool lastTouchFire = false;

		TeamConfigMutator(const std::vector<std::pair<int, int>>& configs,
		                  const std::vector<float>& weights,
		                  int memberCount, float siblingFraction, uint32_t seed,
		                  WheelHolder* wheel = nullptr)
			: teamConfigs(configs), teamWeights(weights),
			  memberCount(memberCount), siblingFraction(siblingFraction), wheel(wheel), rng(seed) {
			assert(teamConfigs.size() == teamWeights.size());
			assert(memberCount >= 1);
		}

		virtual GameState ResetState(Arena* arena) override {
			// 1. Sample one (n_blue, n_orange) from the weighted pool (the native / touch-drill lineup).
			std::discrete_distribution<int> cfgDist(teamWeights.begin(), teamWeights.end());
			auto [nBlue, nOrange] = teamConfigs[cfgDist(rng)];

			// 2. Roll each wheel's per-episode Bernoulli from the holder's fire probability.
			bool touchFire = wheel && wheel->touch_enabled
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < wheel->touch_prob;
			bool goalFire  = wheel && wheel->goal_enabled
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < wheel->goal_prob;

			// A goal drill forces a 1v0 lineup (the lone car is the owner's; BLUE so it attacks +y).
			if (goalFire) { nBlue = 1; nOrange = 0; }
			int n = nBlue + nOrange;

			// 3. Build the per-slot owner list (blue-first slot order). Self-play: every slot -> member 0.
			//    Sibling: cycle member ids across the n slots then shuffle (mirrors League.draw_sibling).
			std::vector<int> botIds(n, 0);
			bool sibling = memberCount > 1
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < siblingFraction;
			if (sibling) {
				for (int j = 0; j < n; j++)
					botIds[j] = j % memberCount;   // cycle member ids as evenly as possible
				std::shuffle(botIds.begin(), botIds.end(), rng);
			} // else: self-play, botIds stays all-zero.

			// 4. The Arena is built FRESH and EMPTY on every Reset() (see Gym::Reset), so we only ADD this
			//    episode's active set: blue cars first (ascending id), then orange (ascending id), so the
			//    deterministic (team,id) sort in GameState::UpdateFromArena reproduces blue-first slot order.
			for (int i = 0; i < nBlue; i++)
				arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE);
			for (int i = 0; i < nOrange; i++)
				arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);

			// 5. Native random kickoff repositions the now-existing cars + ball (overridden below on a drill).
			arena->ResetToRandomKickoff();

			bool drill = goalFire || touchFire;
			lastGoalFire = goalFire;
			lastTouchFire = touchFire;
			if (drill)
				ApplyDrillSpawn(arena, goalFire, touchFire);

			// 6. Capture the state, stamp the per-episode owner map + clock cap (drill vs native).
			GameState state(arena);
			state.slotOwners = botIds;
			state.drillActive = drill;
			state.maxTimeSeconds = (drill && wheel) ? wheel->drill_time
				: (wheel ? wheel->native_time : CommonValues::kMaxTimeSeconds);
			return state;
		}

		// Mid-episode respawn after a drill goal: re-apply the same drill spawn (ball + cars) so the ball
		// leaves the net. GameState::UpdateFromArena detects goals by ball POSITION, so a ball resting in
		// the net would re-increment scoreLine every tick; returning it to the drill spawn both fixes that
		// and lets the curriculum wheels observe repeated goal events. Boost is RESET to the kickoff value
		// (BOOST_SPAWN_AMOUNT ~33) on respawn, matching a live kickoff: preserving the car's drained boost
		// would create a train/inference mismatch (the bot sees boost it never has at kickoff in live play).
		// No-op when the last reset was not a drill.
		virtual void RespawnAfterGoal(Arena* arena) override {
			if (!lastGoalFire && !lastTouchFire)
				return;
			ApplyDrillSpawn(arena, lastGoalFire, lastTouchFire);
		}

	private:
		// Shared drill spawn geometry, used by ResetState and RespawnAfterGoal. Cars spawn with the default
		// CarState boost (BOOST_SPAWN_AMOUNT ~33), so a fresh spawn and a mid-episode respawn both reset
		// boost to the kickoff value. Exactly mirrors the original inline spawn.
		void ApplyDrillSpawn(Arena* arena, bool goalFire, bool touchFire) {
			if (goalFire) {
				// GOAL drill: ball on the lone (BLUE) car's attack axis at +goal_ball_y, still; car behind
				// the ball (smaller y) facing +y toward it. The opponent's absence is the binary difficulty.
				float by = wheel->goal_ball_y;
				BallState bs = {};
				bs.pos = Vec(0.0f, by, CommonValues::BALL_RADIUS);
				bs.vel = Vec(0.0f, 0.0f, 0.0f);
				bs.angVel = Vec(0.0f, 0.0f, 0.0f);
				arena->ball->SetState(bs);

				Car* car = arena->_cars.empty() ? nullptr : *arena->_cars.begin();
				if (car) {
					float d = wheel->touch_d;
					Vec fwd(0.0f, 1.0f, 0.0f);                 // lone BLUE car attacks +y (toward the ball)
					Vec up(0.0f, 0.0f, 1.0f);
					Vec right = up.Cross(fwd);
					RotMat rot(fwd, right, up);
					CarState cs = {};
					cs.pos = Vec(0.0f, by - d, RLConst::CAR_SPAWN_REST_Z); // behind the ball (farther from goal)
					cs.vel = fwd * SPAWN_APPROACH_SPEED;
					cs.angVel = Vec(0.0f, 0.0f, 0.0f);
					cs.rotMat = rot;
					car->SetState(cs);
				}
			} else if (touchFire) {
				// TOUCH drill: ball at center, still; each car at distance d along the kickoff axis facing it.
				float d = wheel->touch_d;
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
					cs.pos = Vec(0.0f, sgn * d, RLConst::CAR_SPAWN_REST_Z);
					cs.vel = fwd * SPAWN_APPROACH_SPEED;
					cs.angVel = Vec(0.0f, 0.0f, 0.0f);
					cs.rotMat = rot;
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
