// EnvThroughput: C++-side env-throughput benchmark (the C++ half of the Python-vs-C++ env speed test).
//
// Mirrors reference/scripts/bench/env_throughput.py EXACTLY so the ticks/sec numbers are comparable:
//   * soccar only, mutators OFF, tick_skip fixed (default 8).
//   * 1v1 (2 agents), fixed action stream read from a file dumped by the Python bench (so both sides
//     feed byte-identical actions). The stream is keyed on (seed, step) on the Python side.
//   * fixed number of decision steps (NOT early-termination on goal): we step a fixed count so both
//     sides do identical work regardless of when a goal would have ended the episode.
//   * warmup steps (excluded from timing); then a timed window; median over reps.
//
// The policy forward pass is a CONSTANT in the comparison (same ActorCritic on both learner sides) and
// cancels out, so this harness runs NO policy net -- it measures pure env throughput: obs-build +
// reward + engine step, in raw sim ticks per wall-second.
//
// Usage:
//   EnvThroughput <streamFile> [--tick-skip 8] [--reps 3]
//
// <streamFile> is the file written by:
//   python scripts/bench/env_throughput.py --dump-stream <file> --steps N --tick-skip 8 --seed S
// Format: comment line, then "steps n c" header, then one line of n*c floats per decision step.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstdlib>

#include "RLGymSim_CPP/Framework.h"
#include "RLGymSim_CPP/Gym.h"
#include "RLGymSim_CPP/Envs/Match.h"
#include "RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/RewardTerms.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h"
#include "RLGymSim_CPP/Utils/TerminalConditions/ScoreboardGoalCondition.h"
#include "RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h"
#include "RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h"

using namespace RLGSC;

struct Stream {
	int steps = 0;
	int n = 0;
	int c = 0;
	std::vector<std::vector<float>> actions; // [steps][n*c]
};

static Stream readStream(const std::string& path) {
	std::ifstream in(path);
	if (!in) { std::cerr << "cannot open stream " << path << std::endl; std::exit(2); }
	Stream s;
	std::string line;
	// skip leading comment lines
	while (std::getline(in, line)) {
		if (!line.empty() && line[0] == '#') continue;
		std::istringstream iss(line);
		iss >> s.steps >> s.n >> s.c;
		break;
	}
	s.actions.reserve(s.steps);
	int row = 0;
	while (row < s.steps && std::getline(in, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		std::vector<float> rowv;
		float v;
		while (iss >> v) rowv.push_back(v);
		if ((int)rowv.size() != s.n * s.c) {
			std::cerr << "stream row " << row << " has " << rowv.size() << " floats, expected "
			          << s.n * s.c << std::endl;
			std::exit(2);
		}
		s.actions.push_back(std::move(rowv));
		row++;
	}
	s.steps = (int)s.actions.size();
	return s;
}

struct Result {
	int nAgents;
	int tickSkip;
	int timedSteps;
	int warmupSteps;
	double wallSeconds;
	double ticksPerSec;
	double decisionsPerSec;
};

static Result runBench(const Stream& s, int tickSkip, int warmup) {
	auto obsBuilder = new TemporalObsBuilder();
	obsBuilder->activeMode = GameMode::SOCCAR;
	obsBuilder->tickSkip = tickSkip;

	auto rewardFn = new ComposedReward();
	rewardFn->addTerm("goal", new GoalTerm(), 5.0f);
	rewardFn->addTerm("win_prob", new WinProbTerm(), 1.0f);
	rewardFn->addTerm("proximity", new ProximityTerm(), 0.5f);
	rewardFn->addTerm("alignment", new AlignmentTerm(), 0.3f);

	auto terminalConds = std::vector<TerminalCondition*>{
		new ScoreboardGoalCondition(7),
		new ScoreboardTimeCondition(7200)
	};
	auto actionParser = new ContinuousActionParser(tickSkip);

	// Single fixed (1,1) matchup, single member (self-play): the active-set spawner degenerates to the
	// old fixed-topology path (every slot owned by member 0).
	auto stateSetter = new TeamConfigMutator(std::vector<std::pair<int, int>>{ {1, 1} }, std::vector<float>{ 1.0f }, 1, 0.0f, 0u);

	Match match(rewardFn, terminalConds, obsBuilder, actionParser, stateSetter, 1, true);
	Gym gym(&match, tickSkip, CAR_CONFIG_OCTANE, GameMode::SOCCAR);

	const int nAgents = 2;
	const int totalSteps = s.steps;

	// Prime the arena/prevState before stepping (Gym::Step reads prevState; without a Reset the
	// player vector is empty and the obs builder indexes out of bounds).
	gym.Reset();

	// Warmup (excluded from timing): JIT/allocate, prime the engine.
	int warm = std::min(warmup, totalSteps);
	for (int t = 0; t < warm; t++) {
		auto res = gym.Step(s.actions[t]);
		if (res.done) gym.Reset();
	}

	// Timed window: step the remaining decisions, count raw ticks.
	int timed = totalSteps - warm;
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int t = warm; t < totalSteps; t++) {
		auto res = gym.Step(s.actions[t]);
		if (res.done) gym.Reset();
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	double wall = std::chrono::duration<double>(t1 - t0).count();

	Result r;
	r.nAgents = nAgents;
	r.tickSkip = tickSkip;
	r.timedSteps = timed;
	r.warmupSteps = warm;
	r.wallSeconds = wall;
	r.decisionsPerSec = timed / wall;
	r.ticksPerSec = ((double)timed * tickSkip) / wall;
	return r;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: EnvThroughput <streamFile> [--tick-skip 8] [--reps 3]" << std::endl;
		return 1;
	}
	std::string streamPath = argv[1];
	int tickSkip = 8;
	int reps = 3;
	for (int i = 2; i < argc; i++) {
		std::string a = argv[i];
		if (a == "--tick-skip" && i + 1 < argc) tickSkip = std::atoi(argv[++i]);
		else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
	}

	RocketSim::Init("collision_meshes", true);

	Stream s = readStream(streamPath);
	std::cout << "=== C++ ENV-THROUGHPUT: stream " << s.steps << " steps x " << s.n << " agents x "
	          << s.c << " controls ===" << std::endl;

	std::vector<double> tps, dps;
	for (int r = 0; r < reps; r++) {
		Result res = runBench(s, tickSkip, 200);
		tps.push_back(res.ticksPerSec);
		dps.push_back(res.decisionsPerSec);
		std::cout << "[rep " << r << "] ticks/sec=" << res.ticksPerSec
		          << "  decisions/sec=" << res.decisionsPerSec
		          << "  sec/decision=" << (res.wallSeconds / res.timedSteps) * 1e3 << "ms"
		          << "  (n_agents=" << res.nAgents << " tick_skip=" << res.tickSkip << ")" << std::endl;
	}
	std::sort(tps.begin(), tps.end());
	std::sort(dps.begin(), dps.end());
	double medTps = tps[tps.size() / 2];
	double medDps = dps[dps.size() / 2];
	std::cout << "\n==================== C++ ENV-THROUGHPUT ====================" << std::endl;
	std::cout << "  mode=soccar mutators=off random_resets=off (deterministic-equivalent)" << std::endl;
	std::cout << "  tick_skip=" << tickSkip << "  agents=1v1  policy_net=EXCLUDED (env-only)" << std::endl;
	std::cout << "  median ticks/sec     : " << medTps << std::endl;
	std::cout << "  median decisions/sec : " << medDps << std::endl;
	std::cout << "  (raw sim ticks per wall-second; decision rate = ticks/sec / tick_skip)" << std::endl;
	std::cout << "============================================================" << std::endl;
	return 0;
}
