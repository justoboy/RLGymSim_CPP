#include "Match.h"

namespace RLGSC {
	void Match::EpisodeReset(const GameState& initialState) {
		prevActions = ActionSet(initialState.players.size());
		for (auto cond : terminalConditions)
			cond->Reset(initialState);
		rewardFn->Reset(initialState);
		obsBuilder->Reset(initialState);
	}

	FList2 Match::BuildObservations(const GameState& state) {
		FList2 results(state.players.size());

		obsBuilder->PreStep(state);

		for (int i = 0; i < (int)state.players.size(); i++) {
			results[i] = obsBuilder->BuildOBS(state.players[i], state, prevActions[i]);
		}

		return results;
	}

	FList Match::GetRewards(const GameState& state, bool done) {
		auto result = FList(state.players.size());

		rewardFn->PreStep(state);
		return rewardFn->GetAllRewards(state, prevActions, done);
	}

	bool Match::IsDone(const GameState& state) {
		for (auto& cond : terminalConditions)
			if (cond->IsTerminal(state))
				return true;

		return false;
	}

	ScoreLine Match::GetScoreLine(const GameState& state) {
		return state.scoreLine;
	}

	ActionSet Match::ParseActions(const ActionParser::Input& actionsData, const GameState& gameState) {
		ActionSet actions = actionParser->ParseActions(actionsData, gameState);

		for (int i = 0; i < (int)gameState.players.size(); i++)
			if (gameState.players[i].carState.isDemoed)
				actions[i] = {};

		return actions;
	}

	GameState Match::ResetState(Arena* arena) {
		// The active-set setter spawns a per-episode number of cars (variable team size + uneven lobbies),
		// so the player count legitimately changes between resets. The old fixed-topology guard that
		// hard-errored on a player-count change is removed; everything downstream (obs build, rewards,
		// action parse, learner routing) iterates state.players.size() at runtime.
		GameState newState = stateSetter->ResetState(arena);

		for (auto& pad : arena->_boostPads)
			pad->SetState({});

		return newState;
	}
}