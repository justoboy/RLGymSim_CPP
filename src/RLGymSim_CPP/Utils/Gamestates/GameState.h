#pragma once
#include "PlayerData.h"
#include "../CommonValues.h"
#include <array>

namespace RLGSC {
	struct ScoreLine {
		int teamGoals[2] = { 0,0 };

		int operator[](size_t index) const {
			return teamGoals[index];
		}

		int& operator[](size_t index) {
			return teamGoals[index];
		}
	};

	// Ball attach info for SPIKE_RUSH / GRIDIRON game modes.
	// Mirrors RocketSim/src/Sim/Ball/Ball.h:BallState::AttachInfo.
	struct AttachInfo {
		uint32_t attachedCarId = 0;       // 0 = free ball, not attached
		uint32_t lastCarrierId = 0;       // Car that most recently carried (persists after release)
		Vec localOffset{};                // Ball offset in carrier's local space
		float engageTimer = 0;            // Seconds ball has been attached to current carrier
		float releaseCooldown = 0;        // Cooldown after release before re-attach allowed
		bool active = false;              // True once the mode's attach mechanic is live
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/game_state.py
	struct GameState {
		float deltaTime = 0; // Time that has passed since last update

		ScoreLine scoreLine;
		int lastTouchCarID = -1;
		std::vector<PlayerData> players;

		BallState ballState;
		PhysObj ball, ballInv;

		std::array<bool, CommonValues::BOOST_LOCATIONS_AMOUNT> boostPads;
		std::array<bool, CommonValues::BOOST_LOCATIONS_AMOUNT> boostPadsInv;

		std::array<float, CommonValues::BOOST_LOCATIONS_AMOUNT> boostPadTimers;
		std::array<float, CommonValues::BOOST_LOCATIONS_AMOUNT> boostPadTimersInv;

		// Last arena we updated with
		// Can be used to determine current arena from within reward function, for example
		// NOTE: Could be null
		Arena* lastArena = NULL;

		// Last tick count when updated
		uint64_t lastTickCount = 0;

		// --- Custom features (SPIKE_RUSH / GRIDIRON) ---
		AttachInfo attachInfo{};            // Ball attach state for custom modes
		bool goalScored[2] = { false, false }; // Latch: whether a goal was scored this tick
		int scoringTeam = -1;               // Team that scored last (-1 = none)
		uint64_t tickCount = 0;             // Global tick counter

		GameState() = default;
		explicit GameState(Arena* arena) {
			UpdateFromArena(arena);
		}

		const PhysObj& GetBallPhys(bool inverted) const {
			return inverted ? ballInv : ball;
		}

		const auto& GetBoostPads(bool inverted) const {
			return inverted ? boostPadsInv : boostPads;
		}

		void UpdateFromArena(Arena* arena);
	};
}