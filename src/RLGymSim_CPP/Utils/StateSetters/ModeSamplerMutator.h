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

	// Mutable holder for the spawn-curriculum config (read by TeamConfigMutator on every reset).
	//   enabled: when false the spawn override is a no-op (every episode uses the native kickoff).
	//   ratio:   scaffold blend 1->0. d = MIN + (MAX-MIN)*(1-ratio).
	//   prob:    fraction of episodes that use the custom (easy) spawn; the rest stay native kickoff.
	struct SpawnHolder {
		bool enabled = false;
		float ratio = 1.0f; // scaffold blend 1->0
		float prob = 0.5f;  // fraction of episodes using custom spawn
	};

	// Custom-spawn geometry (mirrors the reference scaffold constants).
	inline constexpr float SPAWN_MIN_DIST = 400.0f;     // ball basically on the car (ratio=1)
	inline constexpr float SPAWN_MAX_DIST = 4000.0f;    // real kickoff gap (ratio=0)
	inline constexpr float SPAWN_APPROACH_SPEED = 1000.0f; // initial car speed (uu/s) aimed at the ball

	// TeamConfigMutator: per-episode active-set spawner + matchup sampler (+ optional spawn curriculum).
	//
	// Faithful port of the reference TeamConfigMutator + sample_team_config + League.draw_sibling/
	// self_matchup (and the SpawnCurriculumMutator override that runs after it). On every reset it:
	// (1) samples one (n_blue, n_orange) from the weighted pool, (2) spawns exactly n_blue blue +
	// n_orange orange cars into the fresh empty arena (blue-first by id), (3) runs the native random
	// kickoff to reposition them + the ball, (4) on a fraction of episodes overrides the live ball +
	// car states with a custom (easy) spawn, and (5) stamps the per-episode owner map into
	// GameState::slotOwners for the learner to route each slot's trajectory to its owning member.
	//
	// The spawn override writes to the LIVE arena (ball->SetState / car->SetState), mirroring
	// RandomState, so the engine actually simulates the custom spawn (the old snapshot-only override
	// never reached the engine). A single-bot self-play run is the degenerate case: one member,
	// siblingFraction 0, every config (1,1) -> every slot owned by member 0.
	class TeamConfigMutator : public StateSetter {
	public:
		// Weighted pool of (n_blue, n_orange) team configs (mirrors DEFAULT_TEAM_CONFIGS/WEIGHTS).
		std::vector<std::pair<int, int>> teamConfigs;
		std::vector<float> teamWeights;
		// Number of league members (bots). 1 => pure self-play (every slot owned by member 0).
		int memberCount = 1;
		// Fraction of episodes that draw a mixed-lineage (sibling) matchup vs. all-slots-one-bot self.
		float siblingFraction = 0.0f;
		// Optional spawn-curriculum holder (nullptr => always native kickoff).
		SpawnHolder* spawn = nullptr;
		std::mt19937 rng;

		TeamConfigMutator(const std::vector<std::pair<int, int>>& configs,
		                  const std::vector<float>& weights,
		                  int memberCount, float siblingFraction, uint32_t seed,
		                  SpawnHolder* spawn = nullptr)
			: teamConfigs(configs), teamWeights(weights),
			  memberCount(memberCount), siblingFraction(siblingFraction), spawn(spawn), rng(seed) {
			assert(teamConfigs.size() == teamWeights.size());
			assert(memberCount >= 1);
		}

		virtual GameState ResetState(Arena* arena) override {
			// 1. Sample one (n_blue, n_orange) from the weighted pool.
			std::discrete_distribution<int> cfgDist(teamWeights.begin(), teamWeights.end());
			auto [nBlue, nOrange] = teamConfigs[cfgDist(rng)];
			int n = nBlue + nOrange;

			// 2. Build the per-slot owner list (blue-first slot order: first nBlue slots blue, rest orange).
			//    Self-play: every slot -> member 0. Sibling: cycle member ids across the n slots then shuffle
			//    so each bot lands on each team over time (mirrors League.draw_sibling).
			std::vector<int> botIds(n, 0);
			bool sibling = memberCount > 1
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < siblingFraction;
			if (sibling) {
				for (int j = 0; j < n; j++)
					botIds[j] = j % memberCount;   // cycle member ids as evenly as possible
				std::shuffle(botIds.begin(), botIds.end(), rng);
			} // else: self-play, botIds stays all-zero.

			// 3. The Arena is built FRESH and EMPTY on every Reset() (see Gym::Reset), so we only ADD this
			//    episode's active set: blue cars first (ascending id), then orange (ascending id), so the
			//    deterministic (team,id) sort in GameState::UpdateFromArena reproduces blue-first slot order.
			//    We never RemoveCar on a live world (that corrupts this fork's Bullet state).
			for (int i = 0; i < nBlue; i++)
				arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE);
			for (int i = 0; i < nOrange; i++)
				arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);

			// 4. Native random kickoff repositions the now-existing cars + ball.
			arena->ResetToRandomKickoff();

			// 5. Optional spawn-curriculum override: on a fraction of episodes, replace the native kickoff
			//    with a custom (easy) spawn written to the LIVE arena (ball center+still, each car at distance
			//    d along the kickoff axis facing and moving toward the ball).
			if (spawn && spawn->enabled
				&& std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < spawn->prob) {
				float d = SPAWN_MIN_DIST + (SPAWN_MAX_DIST - SPAWN_MIN_DIST) * (1.0f - spawn->ratio);

				// Ball at center, still.
				BallState bs = {};
				bs.pos = Vec(0.0f, 0.0f, CommonValues::BALL_RADIUS);
				bs.vel = Vec(0.0f, 0.0f, 0.0f);
				bs.angVel = Vec(0.0f, 0.0f, 0.0f);
				arena->ball->SetState(bs);

				// Each car at distance d along the kickoff axis, facing + moving toward the center ball.
				for (Car* car : arena->_cars) {
					bool blue = (car->team == Team::BLUE);
					float sgn = blue ? -1.0f : 1.0f;        // blue defends -y (spawns at -d, faces +y)
					Vec fwd(0.0f, -sgn, 0.0f);              // unit vector from car toward center ball
					Vec up(0.0f, 0.0f, 1.0f);
					Vec right = up.Cross(fwd);
					RotMat rot(fwd, right, up);             // col0=forward, col2=up (what obs reads)

					CarState cs = {};
					cs.pos = Vec(0.0f, sgn * d, RLConst::CAR_SPAWN_REST_Z);
					cs.vel = fwd * SPAWN_APPROACH_SPEED;
					cs.angVel = Vec(0.0f, 0.0f, 0.0f);
					cs.rotMat = rot;
					car->SetState(cs);
				}
			}

			// 6. Capture the state and stamp the per-episode owner map (already in blue-first slot order).
			GameState state(arena);
			state.slotOwners = botIds;
			return state;
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
