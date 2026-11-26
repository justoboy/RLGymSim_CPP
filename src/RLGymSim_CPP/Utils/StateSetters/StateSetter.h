#pragma once
#include "../Gamestates/GameState.h"

namespace RLGSC {
	class StateSetter {
	public:

		// NOTE: Applies reset state to arena
		virtual GameState ResetState(Arena* arena) = 0;

		// Mid-episode respawn after a goal on a curriculum DRILL episode. Drill episodes do not terminate
		// on a goal (see ScoreboardGoalCondition), so the ball must be returned to the drill spawn or the
		// position-based goal detection in GameState::UpdateFromArena would re-increment scoreLine every tick
		// while the ball rests in the net. Default no-op: native matches and non-drill setters are unaffected.
		virtual void RespawnAfterGoal(Arena* arena) {}
	};
}