#pragma once
#include "ActionParser.h"
#include "ContinuousActionParser.h" // N_CONTROLS, CarControls via GameState.h
#include <vector>

// PerChannelDiscreteActionParser: the discrete arm's counterpart to ContinuousActionParser.
//
// The discrete policy emits a (9,) vector of per-channel BIN INDICES (one categorical sample per
// control channel — INDEPENDENT heads, NOT the old 90-row joint lookup table in DiscreteAction.h).
// This parser decodes each channel's bin index back to that channel's control value via the FIXED
// per-channel centers, mirroring the reference implementation's decode_bins EXACTLY:
//   ctrl[c] = BIN_CENTERS[c][idx[c]]
//
// Centers (must stay byte-identical to the reference BIN_CENTERS / N_BINS = [9,9,3,9,3,2,2,2,2]):
//   col0 throttle (analog 9) col1 steer (analog 9) col2 pitch (ternary 3) col3 yaw (analog 9)
//   col4 roll (ternary 3) col5 jump col6 boost col7 handbrake col8 release (all binary 2)
//
// Input is the SAME flat FList layout as the continuous parser: [nAgents * N_CONTROLS] floats, each
// float being that channel's bin index (0..nBins[c]-1). GetActionAmount returns N_CONTROLS (9) so
// the learner's action tensor width matches the policy head's channel count.

namespace RLGSC {

	// Fixed per-channel bin centers, indexed [channel][bin]. Mirrors reference BIN_CENTERS.
	inline const std::vector<std::vector<float>>& BinCenters() {
		static const std::vector<std::vector<float>> centers = {
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // throttle (analog)
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // steer   (analog)
			{ -1.0f, 0.0f, 1.0f },                                            // pitch   (ternary)
			{ -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, // yaw     (analog)
			{ -1.0f, 0.0f, 1.0f },                                            // roll    (ternary)
			{ 0.0f, 1.0f },                                                   // jump    (binary)
			{ 0.0f, 1.0f },                                                   // boost   (binary)
			{ 0.0f, 1.0f },                                                   // handbrake(binary)
			{ 0.0f, 1.0f },                                                   // release (binary)
		};
		return centers;
	}

	class PerChannelDiscreteActionParser : public ActionParser {
	public:
		int repeats;

		PerChannelDiscreteActionParser(int repeats = 8) : repeats(repeats) {}

		virtual int GetActionAmount() override {
			return N_CONTROLS; // 9 channels; each carries one bin index
		}

		virtual ActionSet ParseActions(const Input& actionsData, const GameState& gameState) override {
			const auto& centers = BinCenters();
			ActionSet actions;
			int nAgents = (int)gameState.players.size();
			actions.resize(nAgents);

			// actionsData is a flat vector of [nAgents * N_CONTROLS] floats, each a bin index.
			for (int i = 0; i < nAgents; i++) {
				const float* vec = actionsData.data() + i * N_CONTROLS;
				float decoded[N_CONTROLS];
				for (int c = 0; c < N_CONTROLS; c++) {
					int idx = (int)vec[c];
					const auto& col = centers[c];
					if (idx < 0) idx = 0;
					if (idx >= (int)col.size()) idx = (int)col.size() - 1;
					decoded[c] = col[idx];
				}

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
