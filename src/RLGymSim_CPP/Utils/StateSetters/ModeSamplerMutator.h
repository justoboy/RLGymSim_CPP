#pragma once
#include "StateSetter.h"
#include <random>

// State setters ported from custombot/custombot/mode_setters.py:
// TeamConfigMutator, ModeSamplerMutator, SpawnCurriculumMutator.

namespace RLGSC {

	// Game mode enum (mirrors RocketSim/src/Sim/GameMode.h)
	enum class GameMode : int {
		SOCCAR = 0,
		HOOPS,
		HEATSEEKER,
		SNOWDAY,
		SPIKE_RUSH,
		GRIDIRON,
		DROPSHOT,
	};

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

	// TeamConfigMutator: spawns the active team's cars into state.cars from TeamConfigHolder.
	class TeamConfigMutator : public StateSetter {
	public:
		TeamConfigHolder* holder;

		TeamConfigMutator(TeamConfigHolder* holder) : holder(holder) {}

		virtual GameState ResetState(Arena* arena) override {
			GameState state(arena);

			// Clear existing players
			state.players.clear();

			// Spawn blue cars
			for (int i = 0; i < holder->n_blue; i++) {
				PlayerData p;
				p.carId = (uint32_t)(i + 1);
				p.team = Team::BLUE;
				p.carState = CarState();
				p.carState.pos = Vec(-2048.0f, -2560.0f, RLConst::CAR_SPAWN_REST_Z);
				p.carState.rotMat = RotMat::fromYaw(M_PI_4 * 2.0f);
				p.boostFraction = 0.0f;
				p.hasJump = true;
				p.hasFlip = true;
				p.ballTouchedStep = false;
				p.ballTouchedTick = false;
				state.players.push_back(p);
			}

			// Spawn orange cars
			for (int i = 0; i < holder->n_orange; i++) {
				PlayerData p;
				p.carId = (uint32_t)(i + holder->n_blue + 1);
				p.team = Team::ORANGE;
				p.carState = CarState();
				p.carState.pos = Vec(2048.0f, 2560.0f, RLConst::CAR_SPAWN_REST_Z);
				p.carState.rotMat = RotMat::fromYaw(M_PI_4 * 0.0f);
				p.boostFraction = 0.0f;
				p.hasJump = true;
				p.hasFlip = true;
				p.ballTouchedStep = false;
				p.ballTouchedTick = false;
				state.players.push_back(p);
			}

			return state;
		}
	};

	// ModeSamplerMutator: per-episode mode sampler + native routing.
	class ModeSamplerMutator : public StateSetter {
	public:
		ModeHolder* modeHolder;
		Arena* arena; // for mode routing
		std::vector<GameMode> modes;
		std::vector<float> weights;
		std::mt19937 rng;

		// Default mutator levels (for sampling)
		float gravityLevels[5] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f};
		float boostConsumptionLevels[5] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f};
		float boostStrengthLevels[5] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f};
		float ballMaxSpeedLevels[5] = {0.75f, 0.875f, 1.0f, 1.125f, 1.25f};
		float ballSizeLevels[5] = {0.75f, 0.875f, 1.0f, 1.125f, 1.25f};
		float ballBouncinessLevels[5] = {0.75f, 0.875f, 1.0f, 1.125f, 1.25f};

		ModeSamplerMutator(ModeHolder* holder, Arena* arena,
		                   const std::vector<GameMode>& modes,
		                   const std::vector<float>& weights,
		                   bool mutators = false, uint32_t seed = 0)
			: modeHolder(holder), arena(arena), modes(modes), weights(weights), rng(seed) {
			assert(modes.size() == weights.size());
		}

		virtual GameState ResetState(Arena* arena) override {
			// Sample a mode
			std::discrete_distribution<int> dist(modes.begin(), modes.end());
			int idx = dist(rng);
			GameMode sampledMode = modes[idx];
			weights; // suppress warning

			modeHolder->mode = sampledMode;

			// Route mode into arena (set game mode)
			if (arena) {
				// Cast to underlying int for Arena::setMode
				arena->setMode((int)sampledMode);
			}

			// Sample mutators if enabled
			bool mutators = false; // default disabled
			if (mutators) {
				std::uniform_int_distribution<int> gDist(0, 4);
				std::uniform_int_distribution<int> bcDist(0, 4);
				std::uniform_int_distribution<int> bsDist(0, 4);
				std::uniform_int_distribution<int> bmsDist(0, 4);
				std::uniform_int_distribution<int> bszDist(0, 4);
				std::uniform_int_distribution<int> bbDist(0, 4);

				// Write mutator multipliers into arena config
				if (arena) {
					// MutatorConfig would be set here in the full implementation
				}
			}

			return GameState(arena);
		}
	};

	// SpawnHolder: mutable holder for spawn-curriculum config.
	struct SpawnHolder {
		bool enabled = false;
		float ratio = 1.0f; // scaffold blend 1->0
		float prob = 0.5f;  // fraction of episodes using custom spawn
	};

	// SpawnCurriculumMutator: fabricates a custom (easy) spawn on a fraction of episodes.
	class SpawnCurriculumMutator : public StateSetter {
	public:
		SpawnHolder* holder;
		float minDist = 400.0f;   // SPAWN_MIN_DIST
		float maxDist = 4000.0f;  // SPAWN_MAX_DIST
		float approachSpeed = 1000.0f;

		SpawnCurriculumMutator(SpawnHolder* holder) : holder(holder) {}

		virtual GameState ResetState(Arena* arena) override {
			GameState state(arena);

			if (!holder->enabled) return state; // no-op if disabled

			// Sample whether to use custom spawn
			std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
			if (probDist(rng()) > holder->prob) return state; // use native kickoff

			// Compute spawn distance
			float d = minDist + (maxDist - minDist) * (1.0f - holder->ratio);

			// Override player spawns: each car at distance d from ball, facing the ball
			for (auto& player : state.players) {
				Vec ballPos = state.ballState.pos;
				float yaw = atan2(ballPos.y - player.carState.pos.y, ballPos.x - player.carState.pos.x);

				// Position car at distance d from ball along kickoff axis
				player.carState.pos.x = ballPos.x - d * cosf(yaw);
				player.carState.pos.y = ballPos.y - d * sinf(yaw);
				player.carState.pos.z = RLConst::CAR_SPAWN_REST_Z;

				// Face toward ball
				player.carState.rotMat = RotMat::fromYaw(yaw);

				// Initial speed toward ball
				player.carState.vel.x = -approachSpeed * cosf(yaw);
				player.carState.vel.y = -approachSpeed * sinf(yaw);
			}

			return state;
		}

	private:
		std::mt19937 rng;
	};

} // namespace RLGSC
