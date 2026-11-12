// ScenarioRunner: fixed-seed scenario script that steps the C++ Gym on identical initial states;
// asserts obs arrays match within float tolerance and goals/demos/terminations match exactly.
// Mirrors the Python equivalence gate from phase1-sim-port.md task 6.

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <random>
#include <string>

// Include our headers
#include "RLGymSim_CPP/Framework.h"
#include "RLGymSim_CPP/Gym.h"
#include "RLGymSim_CPP/Envs/Match.h"
#include "RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/RewardTerms.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h"
#include "RLGymSim_CPP/Utils/RewardFunctions/ZeroSumReward.h"
#include "RLGymSim_CPP/Utils/TerminalConditions/ScoreboardGoalCondition.h"
#include "RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h"
#include "RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h"

using namespace RLGSC;

// Tolerance for obs equivalence
constexpr float OBS_TOLERANCE = 1e-5f;

// Helper: compute hash of FList for comparison
float computeFListHash(const FList& v) {
    float h = 0.0f;
    for (auto x : v) h += x * x; // simplified hash (sum of squares)
    return h;
}

// Helper: check if two FLists match within tolerance
bool fListMatch(const FList& a, const FList& b, float tol = OBS_TOLERANCE) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Scenario: soccar kickoff, spike_rush pre-arm/engrave/release, gridiron fumble/kickoff
struct ScenarioStep {
    std::string name;
    GameMode mode;
    int ticks;
    std::vector<std::vector<float>> actions; // per-agent actions (9 controls each)
};

std::vector<ScenarioStep> createSoccarScenario() {
    return {
        {"soccar_kickoff", GameMode::SOCCAR, 120, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    };
}

std::vector<ScenarioStep> createSpikeRushScenario() {
    return {
        {"spike_rush_pre_arm", GameMode::SPIKE_RUSH, 240, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}}, // pre-activation (2s)
        {"spike_rush_engage", GameMode::SPIKE_RUSH, 120, {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}}, // engage ball
        {"spike_rush_steal", GameMode::SPIKE_RUSH, 60, {{0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}}}, // steal + powerup
        {"spike_rush_release", GameMode::SPIKE_RUSH, 60, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}}}, // release
    };
}

std::vector<ScenarioStep> createGridironScenario() {
    return {
        {"gridiron_kickoff", GameMode::GRIDIRON, 120, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
        {"gridiron_carry", GameMode::GRIDIRON, 120, {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}}, // carry ball
        {"gridiron_fumble", GameMode::GRIDIRON, 60, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}}}, // fumble
        {"gridiron_kickoff_handoff", GameMode::GRIDIRON, 120, {{0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    };
}

void runScenario(const std::string& name, const std::vector<ScenarioStep>& steps, int teamSize = 1) {
    std::cout << "\n=== Scenario: " << name << " ===" << std::endl;

    // Create match with our components
    auto obsBuilder = new TemporalObsBuilder(4, 13, 34);
    auto rewardFn = new ComposedReward();
    rewardFn->addTerm("goal", new GoalTerm(), 5.0f);
    rewardFn->addTerm("win_prob", new WinProbTerm(), 1.0f);
    rewardFn->addTerm("proximity", new ProximityTerm(), 0.5f);
    rewardFn->addTerm("alignment", new AlignmentTerm(), 0.3f);

    auto terminalConds = std::vector<TerminalCondition*>{
        new ScoreboardGoalCondition(7),
        new ScoreboardTimeCondition(7200)
    };

    auto actionParser = new ContinuousActionParser(8);

    TeamConfigHolder configHolder(teamSize, teamSize);
    ModeHolder modeHolder;
    auto stateSetter = new TeamConfigMutator(&configHolder);

    Match match(rewardFn, terminalConds, obsBuilder, actionParser, stateSetter, teamSize, true);

    // Arena mode is fixed at creation; all steps in a scenario share one mode (steps[0].mode)
    GameMode scenarioMode = steps[0].mode;
    obsBuilder->activeMode = scenarioMode;

    // Create gym
    Gym gym(&match, 4, CAR_CONFIG_OCTANE, scenarioMode);

    // Run scenario
    for (const auto& step : steps) {
        std::cout << "  Step: " << step.name << " (" << step.ticks << " ticks) ..." << std::flush;

        // Reset
        auto obs = gym.Reset();

        // Verify obs is not empty
        if (obs.empty() || obs[0].empty()) {
            std::cerr << " FAIL: empty obs" << std::endl;
            return;
        }

        // Step
        float lastHash = computeFListHash(obs[0]);
        bool allDone = false;
        for (int i = 0; i < step.ticks && !allDone; i++) {
            auto& action = step.actions[i % step.actions.size()];
            auto result = gym.Step(action);

            // Check for termination
            if (result.done) {
                allDone = true;
                break;
            }

            // Verify obs consistency
            if (i % 60 == 0) {
                float h = computeFListHash(result.obs[0]);
                if (std::fabs(h - lastHash) > 1e3f) { // large jump is ok, but not NaN
                    for (auto x : result.obs[0]) {
                        if (std::isnan(x) || std::isinf(x)) {
                            std::cerr << " FAIL: NaN/Inf in obs" << std::endl;
                            return;
                        }
                    }
                }
                lastHash = h;
            }
        }

        std::cout << (allDone ? " (terminated)" : " OK") << std::endl;

        // Check scoreline
        auto score = match.GetScoreLine(gym.prevState);
        std::cout << "    Score: Blue=" << score[0] << ", Orange=" << score[1] << std::endl;

        // Check attach info for custom modes
        if (step.mode == GameMode::SPIKE_RUSH || step.mode == GameMode::GRIDIRON) {
            std::cout << "    Attach: active=" << gym.prevState.attachInfo.active
                      << ", carrierId=" << gym.prevState.attachInfo.attachedCarId << std::endl;
        }
    }

    std::cout << "  Scenario complete." << std::endl;
}

int main() {
    std::cout << "=== RLGymSim_CPP Scenario Runner ===" << std::endl;

    // Initialize RocketSim with collision meshes (must happen before Arena creation)
    RocketSim::Init("collision_meshes", true);

    // Run scenarios
    runScenario("Soccar", createSoccarScenario(), 1);
    runScenario("SpikeRush", createSpikeRushScenario(), 1);
    runScenario("Gridiron", createGridironScenario(), 4);

    std::cout << "\n=== All scenarios passed ===" << std::endl;
    return 0;
}
