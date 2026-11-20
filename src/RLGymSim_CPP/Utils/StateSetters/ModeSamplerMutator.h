#pragma once
#include "StateSetter.h"
#include <random>

// State setters ported from reference/reference/mode_setters.py:
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

	// TeamConfigMutator: per-episode active-set spawner + matchup sampler.
	//
	// Faithful port of the reference TeamConfigMutator + sample_team_config + League.draw_sibling/
	// self_matchup. On every reset it: (1) samples one (n_blue, n_orange) from the weighted pool,
	// (2) clears every car currently in the arena, (3) spawns exactly n_blue blue + n_orange orange
	// cars, (4) runs the native random kickoff to reposition them + the ball, and (5) stamps the
	// per-episode owner map (which league member controls each active slot, in blue-first order) into
	// GameState::slotOwners for the learner to route each slot's trajectory to its owning member.
	//
	// A single-bot self-play run is the degenerate case: one member, siblingFraction 0, every config
	// (1,1) -> every slot owned by member 0, byte-identical to the old fixed-topology path.
	class TeamConfigMutator : public StateSetter {
	public:
		// Weighted pool of (n_blue, n_orange) team configs (mirrors DEFAULT_TEAM_CONFIGS/WEIGHTS).
		std::vector<std::pair<int, int>> teamConfigs;
		std::vector<float> teamWeights;
		// Number of league members (bots). 1 => pure self-play (every slot owned by member 0).
		int memberCount = 1;
		// Fraction of episodes that draw a mixed-lineage (sibling) matchup vs. all-slots-one-bot self.
		float siblingFraction = 0.0f;
		std::mt19937 rng;

		TeamConfigMutator(const std::vector<std::pair<int, int>>& configs,
		                  const std::vector<float>& weights,
		                  int memberCount, float siblingFraction, uint32_t seed)
			: teamConfigs(configs), teamWeights(weights),
			  memberCount(memberCount), siblingFraction(siblingFraction), rng(seed) {
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

			// 5. Capture the state and stamp the per-episode owner map (already in blue-first slot order).
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
