#pragma once
#include "TerminalCondition.h"

namespace RLGSC {
	class NoTouchCondition : public TerminalCondition {
	public:

		int stepsSinceTouch = 0;
		int maxSteps;

		NoTouchCondition(int maxSteps) : maxSteps(maxSteps) {
		}

		virtual void Reset(const GameState& initialState) {
			stepsSinceTouch = 0;
		};

		virtual bool IsTerminal(const GameState& currentState) {
			// On a curriculum DRILL episode the drill clock (ScoreboardTimeCondition) is the sole end
			// condition; the no-touch timeout must not truncate a breakaway before the bot reaches the ball.
			if (currentState.drillActive)
				return false;
			for (auto& player : currentState.players) {
				if (player.ballTouchedStep) {
					stepsSinceTouch = 0;
					return false;
				}
			}

			stepsSinceTouch++;
			return stepsSinceTouch >= maxSteps;
		}
	};
}