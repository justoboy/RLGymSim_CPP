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
		match(match), tickSkip(tickSkip), carConfig(carConfig), gameMode(gameMode), mutatorConfig(mutatorConfig) {
		// NO Arena is created in the constructor. A FRESH Arena is built on every Reset() (see Reset below),
		// mirroring the reference engine, which hands each episode's mutator an EMPTY car set and builds a
		// brand-new Arena per episode. We deliberately never mutate a live bullet world's car set at runtime:
		// RemoveCar()/AddCar() on a running world corrupts this fork's Bullet state (the removed rigid body can
		// still be referenced by the broadphase/pair cache). Whole-object create/destroy is safe. Arena collision
		// meshes are globally cached, so rebuilding per episode is bounded. The bump callback is arena-bound, so
		// it is (re-)registered on each fresh arena; shot/goal/save callbacks live on the persistent eventTracker.
		eventTracker.SetShotCallback(_ShotEventCallback, this);
		eventTracker.SetGoalCallback(_GoalEventCallback, this);
		eventTracker.SetSaveCallback(_SaveEventCallback, this);
	}

	FList2 Gym::Reset() {
		// Build a brand-new, EMPTY Arena for this episode (the StateSetter then spawns this episode's active set
		// into it, so the active car count can vary per episode: 1v1..4v4 even + uneven lobbies). Destroy the
		// previous episode's arena first.
		if (arena)
			delete arena;
		arena = Arena::Create(gameMode);
		arena->SetMutatorConfig(mutatorConfig);
		arena->SetCarBumpCallback(_BumpCallback, this);

		GameState resetState = match->ResetState(arena);
		match->EpisodeReset(resetState);
		prevState = resetState;
		eventTracker.ResetPersistentInfo();

		FList2 obs = match->BuildObservations(resetState);
		return obs;
	}

	Gym::StepResult Gym::Step(const ActionParser::Input& actionsData) {
		ActionSet actions = match->ParseActions(actionsData, prevState);
		match->prevActions = std::move(actions);

		{ // Step arena with actions
			// Bind each action to its car by STABLE car id (prevState.players[i].carId), not by
			// arena->_cars iteration order. The active set can change per reset, so the unordered_set
			// order is not a reliable slot<->action mapping; id lookup is. prevState.players is already
			// in deterministic slot order (blue-first by id), so index i here == slot i everywhere else.
			const auto& actionsRef = match->prevActions;
			for (int i = 0; i < (int)actionsRef.size(); i++) {
				Car* car = arena->GetCar(prevState.players[i].carId);
				if (car)
					car->controls = (CarControls)actionsRef[i];
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

		// Curriculum drill: a goal does NOT end the episode (the goal is the objective, not a terminal
		// condition). GameState::UpdateFromArena detects goals by ball POSITION, so a ball resting in the
		// net would re-increment scoreLine every tick; return it to the drill spawn (which also lets the
		// curriculum wheels see repeated goal events) and re-sync prevState so the next step starts from
		// the spawn. The clock fields are saved/restored so the extra UpdateFromArena does not double-
		// decrement the drill clock. obs/rewards above already saw the ball in the net, so the goal is
		// both observed and credited before the respawn.
		if (prevState.drillActive && (prevState.goalScored[0] || prevState.goalScored[1])) {
			float savedTicksLeft = prevState.ticksLeft;
			bool savedFinished = prevState.finished;
			bool savedOvertime = prevState.isOvertime;
			uint64_t savedLastTickCount = prevState.lastTickCount;
			match->stateSetter->RespawnAfterGoal(arena);
			prevState.UpdateFromArena(arena);
			prevState.ticksLeft = savedTicksLeft;
			prevState.finished = savedFinished;
			prevState.isOvertime = savedOvertime;
			prevState.lastTickCount = savedLastTickCount;
		}

		return StepResult {
			std::move(obs),
			std::move(rewards),
			done,
			prevState // one copy for the caller (prevState stays canonical for next step)
		};
	}
}