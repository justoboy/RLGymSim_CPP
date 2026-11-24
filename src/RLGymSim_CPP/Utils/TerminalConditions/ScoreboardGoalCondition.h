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

	// ScoreboardTimeCondition: terminal when the (pseudo-)match clock runs out, or overtime exceeds
	// maxOvertimeSeconds. Mirrors the reference ScoreboardTimeCondition: the clock's `finished` latch
	// (ticksLeft == -inf) is an immediate terminal; while `isOvertime` (ticksLeft == +inf) the episode runs
	// until maxOvertimeSeconds elapse since overtime began. The clock's per-episode cap (drill vs native) is
	// what makes drill episodes cheap and lets 1v0 drill episodes hard-stop on drill-time.
	class ScoreboardTimeCondition : public TerminalCondition {
	public:
		float maxOvertimeSeconds; // overtime grace before a tied game is force-truncated
		uint64_t overtimeStartTick = 0;
		bool overtimeStarted = false;

		ScoreboardTimeCondition(float maxOvertimeSeconds = 300.0f) : maxOvertimeSeconds(maxOvertimeSeconds) {}

		void Reset(const GameState& initialState) override {
			TerminalCondition::Reset(initialState);
			overtimeStarted = false;
			overtimeStartTick = 0;
		}

		virtual bool IsTerminal(const GameState& currentState) override {
			if (currentState.finished)
				return true; // regulation expired with a non-tie, or an overtime goal ended it
			if (currentState.isOvertime) {
				if (!overtimeStarted) {
					overtimeStarted = true;
					overtimeStartTick = currentState.tickCount;
				}
				float elapsed = (float)(currentState.tickCount - overtimeStartTick) / CommonValues::kTicksPerSecond;
				return elapsed >= maxOvertimeSeconds;
			}
			return false;
		}
	};

} // namespace RLGSC
