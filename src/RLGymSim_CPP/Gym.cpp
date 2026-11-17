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

		return match->BuildObservations(resetState);
	}

	Gym::StepResult Gym::Step(const ActionParser::Input& actionsData) {
		ActionSet actions = match->ParseActions(actionsData, prevState);
		match->prevActions = std::move(actions);

		{ // Step arena with actions
			auto carItr = arena->_cars.begin();
			const auto& actionsRef = match->prevActions;
			for (int i = 0; i < (int)actionsRef.size(); i++) {
				(*carItr)->controls = (CarControls)actionsRef[i];
				carItr++;
			}

			arena->Step(1);
			if (arena->gameMode != GameMode::HEATSEEKER)
				eventTracker.Update(arena); // callbacks mutate prevState.players counters
			// Update prevState IN PLACE: players.resize() is a no-op at equal size and
			// UpdateFromCar overwrites every physics field while preserving the counters,
			// so in-place update is value-identical to the old copy-then-update path.
			prevState.UpdateFromArena(arena);
			arena->Step(tickSkip - 1);
			totalTicks += tickSkip;
			totalSteps++;
		}

		FList2 obs = match->BuildObservations(prevState);
		bool done = match->IsDone(prevState);
		FList rewards = match->GetRewards(prevState, done);

		return StepResult {
			std::move(obs),
			std::move(rewards),
			done,
			prevState // one copy for the caller (prevState stays canonical for next step)
		};
	}
}