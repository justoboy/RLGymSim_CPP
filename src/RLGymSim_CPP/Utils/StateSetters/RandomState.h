#pragma once
#include "StateSetter.h"

namespace RLGSC {
	// Self-play (single-member) resetter. The Arena is built fresh and EMPTY each episode (see Gym::Reset),
	// so this setter is responsible for spawning the active set itself: teamSize blue cars plus (optionally)
	// an equal number of orange opponents, then randomizing their kickoff state. Every slot is owned by
	// member 0 (self-play), so slotOwners is stamped all-zero.
	class RandomState : public StateSetter {
	public:
		bool
			randBallSpeed, randCarSpeed, carsOnGround;
		int teamSize = 1;
		bool spawnOpponents = true;
		CarConfig carConfig = CAR_CONFIG_OCTANE;

		RandomState(bool randBallSpeed, bool randCarSpeed, bool carsOnGround,
		            int teamSize = 1, bool spawnOpponents = true, CarConfig carConfig = CAR_CONFIG_OCTANE) :
			randBallSpeed(randBallSpeed), randCarSpeed(randCarSpeed), carsOnGround(carsOnGround),
			teamSize(teamSize), spawnOpponents(spawnOpponents), carConfig(carConfig) {
		}

		virtual GameState ResetState(Arena* arena);
	};
}