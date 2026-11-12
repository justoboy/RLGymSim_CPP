#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../BasicTypes/Lists.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGSC {
	// Input is a flat float vector; DiscreteAction casts to int indices,
	// ContinuousActionParser reads it as [nAgents * N_CONTROLS] control floats.
	class ActionParser {
	public:
		typedef FList Input;

		virtual ActionSet ParseActions(const Input& actionsData, const GameState& gameState) = 0;
		virtual int GetActionAmount() = 0;
	};
}