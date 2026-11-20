#pragma once
#include "Envs/Match.h"

namespace RLGSC {
	class Gym {
	public:
		// The active Arena. A FRESH Arena is built on every Reset() (mirroring the reference engine, which
		// hands each episode's mutator an empty car set and builds a brand-new Arena per episode). We never
		// mutate a live bullet world's car set at runtime; that path corrupted memory. Arena collision meshes
		// are globally cached, so rebuilding per episode is bounded.
		Arena* arena = nullptr;
		GameEventTracker eventTracker; // persistent across arena rebuilds; shot/goal/save callbacks live here
		Match* match;
		int tickSkip;
		GameState prevState;

		// Stored so Reset() can rebuild the Arena with the same configuration each episode.
		CarConfig carConfig;
		GameMode gameMode;
		MutatorConfig mutatorConfig;

		int totalTicks = 0;
		int totalSteps = 0;

		Gym(Match* match, int tickSkip, CarConfig carConfig = CAR_CONFIG_OCTANE, GameMode gameMode = GameMode::SOCCAR, MutatorConfig mutatorConfig = MutatorConfig(GameMode::SOCCAR));

		RG_NO_COPY(Gym);

		virtual FList2 Reset();

		struct StepResult {
			FList2 obs;
			FList reward;
			bool done;
			GameState state;
		};
		virtual StepResult Step(const ActionParser::Input& actionsData);

		virtual ~Gym() {
			delete arena;
		}
	};
}