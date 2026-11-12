#include "Gym.h"

namespace RLGSC {

	template<int PlayerData::* DATA_VAR>
	void IncPlayerCounter(Car* car, void* userInfo) {
		if (!car)
			return;

		Gym* gym = (Gym*)userInfo;
		for (auto& player : gym->prevState.players)
			if (player.carId == car->id)
				(player.*DATA_VAR)++;
	}

	void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchShots>(shooter, userInfo);
		IncPlayerCounter<&PlayerData::matchShotPasses>(passer, userInfo);
	}

	void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchGoals>(scorer, userInfo);
		IncPlayerCounter<&PlayerData::matchAssists>(passer, userInfo);
	}

	void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchSaves>(saver, userInfo);
	}

	void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
		if (bumper->team == victim->team)
			return;

		IncPlayerCounter<&PlayerData::matchBumps>(bumper, userInfo);

		if (isDemo)
			IncPlayerCounter<&PlayerData::matchDemos>(bumper, userInfo);
	}

	FList2 Gym::obsTupleToFList2(const std::vector<ObsTuple>& tuples) {
		// Flatten obs tuple: for each player, concatenate q, kv, m into a single FList
		FList2 result(tuples.size());
		for (int i = 0; i < (int)tuples.size(); i++) {
			const auto& [q, kv, m] = tuples[i];
			result[i] = q;
			result[i].insert(result[i].end(), kv.begin(), kv.end());
			result[i].insert(result[i].end(), m.begin(), m.end());
		}
		return result;
	}

	Gym::Gym(Match* match, int tickSkip, CarConfig carConfig, GameMode gameMode, MutatorConfig mutatorConfig) :
		match(match), tickSkip(tickSkip) {
		arena = Arena::Create(gameMode);
		arena->SetMutatorConfig(mutatorConfig);

		for (int i = 0; i < match->teamSize; i++) {
			arena->AddCar(Team::BLUE, carConfig);
			if (match->spawnOpponents)
				arena->AddCar(Team::ORANGE, carConfig);
		}

		eventTracker.SetShotCallback(_ShotEventCallback, this);
		eventTracker.SetGoalCallback(_GoalEventCallback, this);
		eventTracker.SetSaveCallback(_SaveEventCallback, this);

		arena->SetCarBumpCallback(_BumpCallback, this);
	}

	FList2 Gym::Reset() {
		GameState resetState = match->ResetState(arena);
		match->EpisodeReset(resetState);
		prevState = resetState;
		eventTracker.ResetPersistentInfo();

		auto obsTuples = match->BuildObservations(resetState);
		FList2 obs = obsTupleToFList2(obsTuples);
		return obs;
	}

	Gym::StepResult Gym::Step(const ActionParser::Input& actionsData) {
		ActionSet actions = match->ParseActions(actionsData, prevState);
		match->prevActions = actions;

		GameState state;

		{ // Step arena with actions
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < (int)actions.size(); i++) {
				(*carItr)->controls = (CarControls)actions[i];
				carItr++;
			}

			arena->Step(1);
			if (arena->gameMode != GameMode::HEATSEEKER)
				eventTracker.Update(arena);
			state = prevState; // All callbacks have been hit
			state.UpdateFromArena(arena);
			arena->Step(tickSkip - 1);
			totalTicks += tickSkip;
			totalSteps++;
		}

		auto obsTuples = match->BuildObservations(state);
		FList2 obs = obsTupleToFList2(obsTuples);
		bool done = match->IsDone(state);
		FList rewards = match->GetRewards(state, done);
		prevState = state;

		return StepResult {
			obs,
			rewards,
			done,
			state
		};
	}
}