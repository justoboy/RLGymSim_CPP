#include "GameState.h"

#include <algorithm>

#include "../../Math.h"

using namespace RLGSC;

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};
void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
	RG_LOG("Building boost pad index map...");

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

	RG_LOG(" > Done");
	boostPadIndexMapBuilt = true;
}

void RLGSC::GameState::UpdateFromArena(Arena* arena) {
	lastArena = arena;
	int tickSkip = RS_MAX(arena->tickCount - lastTickCount, 0);

	deltaTime = tickSkip * (1 / 120.f);

	ballState = arena->ball->GetState();
	ball = PhysObj(ballState);
	ballInv = ball.Invert();

	// Copy attach info for custom modes (SPIKE_RUSH / GRIDIRON)
	// (RLGSC::AttachInfo is a field-for-field mirror of RocketSim::BallState::AttachInfo)
	attachInfo.attachedCarId = ballState.attachInfo.attachedCarId;
	attachInfo.lastCarrierId = ballState.attachInfo.lastCarrierId;
	attachInfo.localOffset   = ballState.attachInfo.localOffset;
	attachInfo.engageTimer   = ballState.attachInfo.engageTimer;
	attachInfo.releaseCooldown = ballState.attachInfo.releaseCooldown;
	attachInfo.active        = ballState.attachInfo.active;

	// Build players in a DETERMINISTIC slot order (blue-first, then by ascending car id) so that
	// slot index i always maps to the same logical agent across episodes. arena->_cars is an
	// unordered_set, so iterating it directly would scramble the slot<->agent binding that the obs
	// builder and the learner's per-slot routing depend on. The active-set setter spawns blue cars
	// before orange ones, so a (team, id) sort reproduces the reference's blue-then-orange slot order.
	std::vector<Car*> orderedCars(arena->_cars.begin(), arena->_cars.end());
	std::sort(orderedCars.begin(), orderedCars.end(), [](const Car* a, const Car* b) {
		int ta = (int)a->team, tb = (int)b->team;
		if (ta != tb) return ta < tb;       // BLUE(0) before ORANGE(1)
		return a->id < b->id;               // then by stable id
	});

	players.resize(orderedCars.size());

	for (int i = 0; i < (int)players.size(); i++) {
		auto& player = players[i];
		player.UpdateFromCar(orderedCars[i], arena->tickCount, tickSkip);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;
	}

	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		// Check again? This seems stupid but also makes sense to me
		//	Without this, we could lock as the index map is building, then go build again
		//	I would like to keep the mutex inside the if statement so it is only checked a few times
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		int idx = boostPadIndexMap[i];
		int invIdx = boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];

		auto state = arena->_boostPads[idx]->GetState();
		auto stateInv = arena->_boostPads[invIdx]->GetState();

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Active pad count (hoops=20 vs soccar-family=34); obs masks pad slots beyond this out.
	nActivePads = (int)arena->_boostPads.size();

	// Dropshot tile damage grid: flatten [team][tile] raw damage levels (FULL=0/DAMAGED=1/BROKEN=2).
	// Zeroed for non-dropshot modes so the soccar query tail stays clean.
	if (arena->gameMode == GameMode::DROPSHOT) {
		DropshotTilesState tiles = arena->GetDropshotTilesState();
		for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
			for (int t = 0; t < RLConst::Dropshot::NUM_TILES_PER_TEAM; t++)
				tileStates[team * RLConst::Dropshot::NUM_TILES_PER_TEAM + t] =
					(float)tiles.states[team][t].damageState;
	} else {
		tileStates.fill(0.0f);
	}

	// Reset goal-scored latches at the start of each update
	goalScored[0] = goalScored[1] = false;
	scoringTeam = -1;

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	if (Math::IsBallScored(ball.pos)) {
		int team = (int)RS_TEAM_FROM_Y(ball.pos.y);
		scoreLine[team]++;
		goalScored[team] = true;
		scoringTeam = team;
	}

	// --- Pseudo-match clock (mirrors reference Scoreboard SharedInfoProvider) ---
	// First update (lastTickCount == 0): initialize the clock to full regulation time. Subsequent
	// updates count it down by the elapsed ticks; on expiry a tie -> overtime (+inf sentinel) and a
	// non-tie -> finished (-inf sentinel). A goal while overtime ends the game (finished).
	if (lastTickCount == 0) {
		ticksLeft = CommonValues::kMaxTimeSeconds * CommonValues::kTicksPerSecond;
		isOvertime = false;
		finished = false;
	} else {
		if (!isOvertime && !finished) {
			ticksLeft -= (float)tickSkip;
			if (ticksLeft <= 0.0f) {
				if (scoreLine[0] == scoreLine[1]) { ticksLeft = CommonValues::kInf; isOvertime = true; }
				else { ticksLeft = -CommonValues::kInf; finished = true; }
			}
		}
		if (goalScored[0] || goalScored[1]) {
			if (isOvertime && scoreLine[0] != scoreLine[1]) { ticksLeft = -CommonValues::kInf; finished = true; }
		}
	}

	lastTickCount = arena->tickCount;
	tickCount = arena->tickCount;
}