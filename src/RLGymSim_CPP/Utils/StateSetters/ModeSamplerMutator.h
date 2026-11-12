#pragma once
#include "StateSetter.h"
#include <random>

// State setters ported from custombot/custombot/mode_setters.py:
// TeamConfigMutator, ModeSamplerMutator, SpawnCurriculumMutator.
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

	// TeamConfigMutator: resets to a random kickoff and captures the resulting state.
	// (Cars are already spawned into the arena by the Gym constructor based on teamSize,
	//  so this setter only handles the kickoff reset itself.)
	class TeamConfigMutator : public StateSetter {
	public:
		TeamConfigHolder* holder;

		TeamConfigMutator(TeamConfigHolder* holder) : holder(holder) {}

		virtual GameState ResetState(Arena* arena) override {
			(void)holder; // team sizes are fixed at Gym construction (Match::playerAmount check)
			arena->ResetToRandomKickoff();
			return GameState(arena);
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
			if (probDist(rng) > holder->prob) return state; // use native kickoff

			// Compute spawn distance
			float d = minDist + (maxDist - minDist) * (1.0f - holder->ratio);

			// Override player spawns: each car at distance d from ball, facing the ball
			for (auto& player : state.players) {
				Vec ballPos = state.ballState.pos;
				float yaw = atan2f(ballPos.y - player.carState.pos.y, ballPos.x - player.carState.pos.x);

				// Position car at distance d from ball along kickoff axis
				player.carState.pos.x = ballPos.x - d * cosf(yaw);
				player.carState.pos.y = ballPos.y - d * sinf(yaw);
				player.carState.pos.z = RLConst::CAR_SPAWN_REST_Z;

				// Face toward ball
				player.carState.rotMat = Angle(yaw, 0, 0).ToRotMat();

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
