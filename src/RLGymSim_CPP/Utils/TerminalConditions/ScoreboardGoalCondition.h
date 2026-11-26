#pragma once
#include "TerminalCondition.h"

// ScoreboardGoalCondition: terminal when a goal is scored by either team.
// Extends the existing GoalScoreCondition with scoreboard-based detection.
namespace RLGSC {

	class ScoreboardGoalCondition : public TerminalCondition {
	public:
		int targetGoals; // number of goals to reach (0 = never terminate on score)
		bool fullMatch;  // native (non-drill) episodes play to the clock: a goal respawns to kickoff (see
		                 // Gym::Step) instead of terminating, so the match runs to expiry with multiple goals.
		                 // This is what lets the profile tracker accumulate multi-goal history (its EMAs +
		                 // pseudo-count features need several goals to leave the neutral prior). Default false
		                 // keeps the cheap goal-terminating objective available when the curriculum is off.

		ScoreboardGoalCondition(int targetGoals = 0, bool fullMatch = false)
			: targetGoals(targetGoals), fullMatch(fullMatch) {}

		void Reset(const GameState& initialState) override {
			TerminalCondition::Reset(initialState);
		}

		virtual bool IsTerminal(const GameState& currentState) override {
			// On a curriculum DRILL episode the goal is the OBJECTIVE, not the end condition: terminating on
			// a goal lets the agent avoid scoring to keep banking the dense per-step reward (the goal-terminate
			// reward hack). Drill episodes instead run to the drill clock (ScoreboardTimeCondition hard-stops
			// on ticksLeft <= 0), and a scored drill goal respawns to kickoff mid-episode (see Gym::Step) so
			// the wheels see repeated goal events.
			if (currentState.drillActive)
				return false;
			// Native FULL-MATCH episode: a goal respawns to kickoff (see Gym::Step) and the match runs to the
			// clock, so the goal must NOT terminate. This is what lets the profile tracker accumulate
			// multi-goal history. When fullMatch is false, native episodes keep the goal-terminates rule.
			if (fullMatch)
				return false;
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
