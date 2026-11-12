#pragma once
#include "TerminalCondition.h"

// ScoreboardGoalCondition: terminal when a goal is scored by either team.
// Extends the existing GoalScoreCondition with scoreboard-based detection.
namespace RLGSC {

	class ScoreboardGoalCondition : public TerminalCondition {
	public:
		int targetGoals; // number of goals to reach (0 = never terminate on score)

		ScoreboardGoalCondition(int targetGoals = 0) : targetGoals(targetGoals) {}

		void Reset(const GameState& initialState) override {
			TerminalCondition::Reset(initialState);
		}

		virtual bool IsTerminal(const GameState& currentState) override {
			if (targetGoals > 0) {
				for (int i = 0; i < 2; i++) {
					if (currentState.scoreLine[i] >= targetGoals) {
						return true;
					}
				}
			}
			return false;
		}
	};

	// ScoreboardTimeCondition: terminal when time runs out
	class ScoreboardTimeCondition : public TerminalCondition {
	public:
		int targetTime; // target time in ticks (default: 60s * 120fps = 7200)

		ScoreboardTimeCondition(int targetTime = 7200) : targetTime(targetTime) {}

		void Reset(const GameState& initialState) override {
			TerminalCondition::Reset(initialState);
		}

		virtual bool IsTerminal(const GameState& currentState) override {
			return currentState.tickCount >= (uint64_t)targetTime;
		}
	};

} // namespace RLGSC
