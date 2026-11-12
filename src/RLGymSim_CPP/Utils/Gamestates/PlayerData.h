#pragma once
#include "../../Framework.h"
#include "PhysObj.h"

namespace RLGSC {
	// Ball attach info for a specific player (mirrors CarState::ballAttachInfo from the fork).
	struct PlayerBallAttachInfo {
		bool isCarrier = false;         // True if this car is currently carrying the ball
		uint32_t carrierId = 0;         // ID of the current carrier (same as carId when isCarrier)
		Vec localOffset{};              // Ball offset in player's local space
		float engageTimer = 0;          // Seconds carrying
		float releaseCooldown = 0;      // Cooldown after release
		bool attachActive = false;      // True once the mode's attach mechanic is live
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py
	struct PlayerData {
		uint32_t carId;
		Team team;

		PhysObj phys, physInv;
		CarState carState;

		// matchAssists: being the passer to a teammate who shot and scored
		// matchBumps: any bump against an opponent, including demos
		int
			matchGoals = 0,
			matchSaves = 0,
			matchAssists = 0,
			matchShots = 0,
			matchShotPasses = 0,
			matchBumps = 0, 
			matchDemos = 0,
			boostPickups = 0;

		bool hasJump;
		bool hasFlip;
		float boostFraction; // From 0 to 1

		bool ballTouchedStep; // True if the player touched the ball during any of tick of the step
		bool ballTouchedTick; // True if the player is touching the ball on the final tick of the step

		// --- Custom features (SPIKE_RUSH / GRIDIRON) ---
		PlayerBallAttachInfo ballAttach{};  // Ball attach state for this player

		void UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip);

		const PhysObj& GetPhys(bool inverted) const {
			return inverted ? physInv : phys;
		}
	};
}