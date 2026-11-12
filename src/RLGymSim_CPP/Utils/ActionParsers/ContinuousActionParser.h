#pragma once
#include "ActionParser.h"
#include <vector>

// ContinuousActionParser: passes the (9,) float control vector straight to the engine
// after decode_controls (clip continuous cols 0..4 to [-1,1]; threshold binary cols 5..8 at 0),
// repeated `repeats` times into the (repeats, 9) block the engine steps over.
// Mirrors custombot/custombot/agent.py ContinuousActionParser.

namespace RLGSC {

	constexpr int N_CONTROLS = 9;    // throttle, steer, pitch, yaw, roll, jump, boost, handbrake, release
	constexpr int N_CONTINUOUS = 5;  // cols 0..4: throttle, steer, pitch, yaw, roll
	constexpr int N_BINARY = 4;      // cols 5..8: jump, boost, handbrake, release

	// Decode a raw 9-dim policy vector into engine control values.
	// Continuous channels (0..4) are clipped to [-1,1]; binary channels (5..8) are thresholded at 0.
	static float* decode_controls(const float* vec) {
		static float out[N_CONTROLS];
		for (int i = 0; i < N_CONTINUOUS; i++) {
			out[i] = vec[i];
			if (out[i] < -1.0f) out[i] = -1.0f;
			if (out[i] > 1.0f) out[i] = 1.0f;
		}
		for (int i = N_CONTINUOUS; i < N_CONTROLS; i++) {
			out[i] = (vec[i] > 0.0f) ? 1.0f : 0.0f;
		}
		return out;
	}

	class ContinuousActionParser : public ActionParser {
	public:
		int repeats;

		ContinuousActionParser(int repeats = 8) : repeats(repeats) {}

		virtual int GetActionAmount() override {
			return N_CONTROLS; // 9 controls for continuous action space
		}

		virtual ActionSet ParseActions(const Input& actionsData, const GameState& gameState) override {
			ActionSet actions;
			int nAgents = (int)gameState.players.size();
			actions.resize(nAgents);

			// actionsData is a flat vector of [nAgents * N_CONTROLS] floats
			for (int i = 0; i < nAgents; i++) {
				const float* vec = actionsData.begin() + i * N_CONTROLS;
				float* decoded = decode_controls(vec);

				// Build CarControls from decoded values
				CarControls ctrl;
				ctrl.throttle = decoded[0];
				ctrl.steer = decoded[1];
				ctrl.pitch = decoded[2];
				ctrl.yaw = decoded[3];
				ctrl.roll = decoded[4];
				ctrl.jump = (decoded[5] > 0.5f);
				ctrl.boost = (decoded[6] > 0.5f);
				ctrl.handbrake = (decoded[7] > 0.5f);
				ctrl.powerup = (decoded[8] > 0.5f); // release button for SPIKE_RUSH

				// Store as Action for prevActions tracking
				actions[i] = Action(
					ctrl.throttle, ctrl.steer, ctrl.pitch, ctrl.yaw, ctrl.roll,
					ctrl.jump ? 1.0f : 0.0f, ctrl.boost ? 1.0f : 0.0f, ctrl.handbrake ? 1.0f : 0.0f
				);

				// Repeat this control block into the arena's input buffer
				for (int r = 0; r < repeats; r++) {
					// In the full implementation, these would be pushed to arena->carControls
					// For now, we return the ActionSet for the Match layer to apply
				}
			}

			return actions;
		}
	};

} // namespace RLGSC
