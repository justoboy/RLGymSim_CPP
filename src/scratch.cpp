// scratch.cpp — fixed-seed benchmark harness for RLGymSim_CPP (pre-authorized scratch space).
// Deterministic: fixed kickoff seed + fixed action sequence => byte-identical obs/reward stream.
// Emits: steps/s (throughput metric) and a golden checksum (equivalence gate for Tier A edits).

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>

#include "RLGymSim_CPP/Framework.h"
#include "RLGymSim_CPP/Gym.h"
#include "RLGymSim_CPP/Envs/Match.h"
#include "RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/ZeroSumReward.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/RewardTerms.h"
#include "RLGymSim_CPP/Utils/TerminalConditions/ScoreboardGoalCondition.h"
#include "RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h"
#include "RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h"

using namespace RLGSC;

// Deterministic state setter: spawns the single blue car into the fresh empty arena, then a fixed-seed
// kickoff. No RNG elsewhere. (The Gym constructor no longer spawns cars; the setter owns the active set.)
class FixedKickoffMutator : public StateSetter {
public:
	int seed;
	explicit FixedKickoffMutator(int seed) : seed(seed) {}
	virtual GameState ResetState(Arena* arena) override {
		arena->AddCar(Team::BLUE, CAR_CONFIG_OCTANE); // single self-play car (spawnOpponents=false)
		arena->ResetToRandomKickoff(seed);
		return GameState(arena);
	}
};

int main(int argc, char** argv) {
	int steps = (argc > 1) ? atoi(argv[1]) : 2000;
	int seed = (argc > 2) ? atoi(argv[2]) : 12345;

	RocketSim::Init("collision_meshes", true);

	auto obsBuilder = new TemporalObsBuilder();
	obsBuilder->activeMode = GameMode::SOCCAR;

	// Single-member ComposedReward (self-play): one member selects these 4 terms with the given weights,
	// no team blending (plain weighted sum, matching the old addTerm path).
	std::vector<RewardTerm*> terms;
	terms.push_back(new GoalTerm());
	terms.push_back(new WinProbTerm());
	terms.push_back(new ProximityTerm());
	terms.push_back(new AlignmentTerm());
	MemberRewardSpec spec;
	spec.termWeights = { {"goal", 5.0f}, {"win_prob", 1.0f}, {"proximity", 0.5f}, {"alignment", 0.3f} };
	spec.teamSpirit = 0.0f;
	spec.opponentPunish = 0.0f;
	auto rewardFn = new ComposedReward(std::move(terms), std::vector<MemberRewardSpec>{ std::move(spec) });

	auto terminalConds = std::vector<TerminalCondition*>{
		new ScoreboardGoalCondition(1000000) // effectively never; keep episode alive
	};

	auto actionParser = new ContinuousActionParser(8);
	auto stateSetter = new FixedKickoffMutator(seed);

	// spawnOpponents=false: a single car means no car-car demos, and demo respawn is the only
	// mid-episode RNG consumer (Math::RandInt with seed=-1). With no demos, the episode is fully
	// deterministic: fixed kickoff seed + fixed actions => byte-identical obs/reward stream.
	Match match(rewardFn, terminalConds, obsBuilder, actionParser, stateSetter, 1, false);
	Gym gym(&match, 8, CAR_CONFIG_OCTANE, GameMode::SOCCAR);

	// Fixed deterministic action sequence (no RNG at step time)
	FList action = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	auto obs0 = gym.Reset();

	// Golden checksum: deterministic function of the whole obs/reward stream
	double obsSum = 0.0, rewSum = 0.0;
	size_t obsBytes = obs0[0].size();

	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++) {
		auto r = gym.Step(action);
		for (auto& o : r.obs) for (float x : o) obsSum += (double)x;
		for (float x : r.reward) rewSum += (double)x;
		if (r.done) break;
	}
	auto t1 = std::chrono::steady_clock::now();

	double secs = std::chrono::duration<double>(t1 - t0).count();
	printf("steps=%d obsWidth=%zu stepsPerSec=%.2f obsSum=%.10f rewSum=%.10f\n",
	       steps, obsBytes, steps / secs, obsSum, rewSum);

	// --- Component isolation (profiling instrumentation) ---
	// Physics-only: same tickSkip budget, no obs/reward plumbing.
	// Reset to the SAME kickoff state and zero controls so the car lands and sits still:
	// identical contact structure in both builds => steps/s difference is real tick-cost
	// difference, not a workload confound from diverged trajectories.
	gym.Reset();
	for (auto& car : gym.arena->_cars)
		car->controls = CarControls{};

	auto t2 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++)
		gym.arena->Step(8);
	auto t3 = std::chrono::steady_clock::now();
	printf("physicsOnlyPerSec=%.2f\n", steps / std::chrono::duration<double>(t3 - t2).count());

	// Obs-only: BuildObservations on a static state (shift + extract + flatten, no physics).
	auto t4 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++)
		match.BuildObservations(gym.prevState);
	auto t5 = std::chrono::steady_clock::now();
	printf("obsOnlyPerSec=%.2f\n", steps / std::chrono::duration<double>(t5 - t4).count());

	// Reward-only: GetRewards on a static state.
	auto t6 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++)
		match.GetRewards(gym.prevState, false);
	auto t7 = std::chrono::steady_clock::now();
	printf("rewardOnlyPerSec=%.2f\n", steps / std::chrono::duration<double>(t7 - t6).count());

	// Plumbing-only: everything in Step EXCEPT BuildObservations/GetRewards
	// (ParseActions + arena steps + eventTracker + GameState copies + StepResult return).
	auto t8 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++) {
		ActionSet actions = match.ParseActions(action, gym.prevState);
		match.prevActions = actions;
		GameState state;
		auto carItr = gym.arena->_cars.begin();
		for (int j = 0; j < (int)actions.size(); j++) {
			(*carItr)->controls = (CarControls)actions[j];
			carItr++;
		}
		gym.arena->Step(1);
		gym.eventTracker.Update(gym.arena);
		state = gym.prevState;          // copy 1
		state.UpdateFromArena(gym.arena);
		gym.arena->Step(7);
		gym.prevState = state;          // copy 2
		Gym::StepResult res{FList2{}, FList{}, false, state}; // copy 3 (return)
		(void)res;
	}
	auto t9 = std::chrono::steady_clock::now();
	printf("plumbingOnlyPerSec=%.2f\n", steps / std::chrono::duration<double>(t9 - t8).count());

	// Physics + obs (no reward, no StepResult plumbing): isolates obs cost IN-LOOP.
	auto t10 = std::chrono::steady_clock::now();
	for (int i = 0; i < steps; i++) {
		gym.arena->Step(1);
		gym.eventTracker.Update(gym.arena);
		GameState state = gym.prevState;
		state.UpdateFromArena(gym.arena);
		gym.arena->Step(7);
		auto obs = match.BuildObservations(state);
		gym.prevState = state;
		(void)obs;
	}
	auto t11 = std::chrono::steady_clock::now();
	printf("physPlusObsPerSec=%.2f\n", steps / std::chrono::duration<double>(t11 - t10).count());

#ifdef RS_PROF
	RocketSim::RSProf::Dump();
#endif

	return 0;
}
