#pragma once
#include "ActionParser.h"
#include <vector>

// CategoricalActionParser: the discrete arm's counterpart of ContinuousActionParser.
// Mirrors custombot/custombot/agent.py DiscreteActionParser + bc_expand.decode_bins:
// the policy emits a (9,) vector of per-channel BIN INDICES; each channel's index selects
// one of its fixed bin centers (analog axes 9-way step-0.25, ternary axes {-1,0,1}, binary
// axes {0,1}); the chosen centers form the 9-control engine vector, repeated `repeats` times.
// This is the INDEPENDENT per-axis head (NOT the old 90-row joint lookup table in
// DiscreteAction.cpp), so the level counts never multiply.

namespace RLGSC {

	// Per-channel bin centers, SAME column order as the label vector:
	//   col0 throttle(analog) col1 steer(analog) col2 pitch(ternary) col3 yaw(analog)
	//   col4 roll(ternary) col5 jump col6 boost col7 handbrake col8 release (binary)
	inline const std::vector<std::vector<float>>& BinCenters() {
		static const std::vector<std::vector<float>> centers = {
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // throttle
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // steer
			{ -1.0f, 0.0f, 1.0f },                                            // pitch
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // yaw
			{ -1.0f, 0.0f, 1.0f },                                            // roll
			{ 0.0f, 1.0f },                                                   // jump
			{ 0.0f, 1.0f },                                                   // boost
			{ 0.0f, 1.0f },                                                   // handbrake
			{ 0.0f, 1.0f },                                                   // release
		};
		return centers;
	}

	// Decode a (9,) bin-index vector into a (9,) control vector (the chosen center per channel).
	static float* decode_bins(const float* idx) {
		static float out[N_CONTROLS];
		const auto& centers = BinCenters();
		for (int c = 0; c < N_CONTROLS; c++) {
			int i = (int)idx[c];
			if (i < 0) i = 0;
			if (i >= (int)centers[c].size()) i = (int)centers[c].size() - 1;
			out[c] = centers[c][i];
		}
		return out;
	}

	class CategoricalActionParser : public ActionParser {
	public:
		int repeats;

		CategoricalActionParser(int repeats = 8) : repeats(repeats) {}

		virtual int GetActionAmount() override {
			return N_CONTROLS; // 9 per-channel bin indices
		}

		virtual ActionSet ParseActions(const Input& actionsData, const GameState& gameState) override {
			ActionSet actions;
			int nAgents = (int)gameState.players.size();
			actions.resize(nAgents);

			// actionsData is a flat vector of [nAgents * N_CONTROLS] bin indices (as floats)
			for (int i = 0; i < nAgents; i++) {
				const float* idx = actionsData.data() + i * N_CONTROLS;
				float* decoded = decode_bins(idx);

				CarControls ctrl;
				ctrl.throttle = decoded[0];
				ctrl.steer = decoded[1];
				ctrl.pitch = decoded[2];
				ctrl.yaw = decoded[3];
				ctrl.roll = decoded[4];
				ctrl.jump = (decoded[5] > 0.5f);
				ctrl.boost = (decoded[6] > 0.5f);
				ctrl.handbrake = (decoded[7] > 0.5f);
				ctrl.powerup = (decoded[8] > 0.5f);

				actions[i] = Action(
					ctrl.throttle, ctrl.steer, ctrl.pitch, ctrl.yaw, ctrl.roll,
					ctrl.jump ? 1.0f : 0.0f, ctrl.boost ? 1.0f : 0.0f, ctrl.handbrake ? 1.0f : 0.0f,
					ctrl.powerup ? 1.0f : 0.0f
				);
			}

			return actions;
		}
	};

} // namespace RLGSC
