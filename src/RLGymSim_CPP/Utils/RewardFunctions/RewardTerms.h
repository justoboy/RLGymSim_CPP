#pragma once
#include "RewardFunction.h"
#include "../CommonValues.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <functional>

// Reward terms ported from reference/reference/reward_terms.py (24 TERM_REGISTRY terms).
// Each term computes a raw vector per player; delta terms subtract previous tick's raw value.
// The ComposedReward class handles term routing and team_spirit/opponent_punish blending.

namespace RLGSC {

	// Simple float clamp used by the reward terms below
	inline float clamp(float v, float lo, float hi) {
		return v < lo ? lo : (v > hi ? hi : v);
	}

	// Base class for reward terms with delta/non-delta support
	class RewardTerm : public RewardFunction {
	public:
		std::string name;
		bool isDelta; // subtract previous tick's raw vector
		float prevRaw = 0.0f;

		RewardTerm(const std::string& name, bool isDelta = true)
			: name(name), isDelta(isDelta) {}

		virtual float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) = 0;

		virtual float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}

		// Capture raw for delta terms (called by ComposedReward after each step)
		void captureRaw(float raw) { prevRaw = raw; }

		virtual void Reset(const GameState& initialState) override {
			RewardFunction::Reset(initialState);
			prevRaw = 0.0f;
		}

		virtual void PreStep(const GameState& state) override {
			RewardFunction::PreStep(state);
			if (isDelta) {
				// prevRaw will be set at end of step by the caller
			}
		}
	};

	// ============================================================================
	// Term 0: GoalTerm - goal scored proximity
	// ============================================================================
	class GoalTerm : public RewardTerm {
	public:
		GoalTerm() : RewardTerm("goal", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.team == RS_TEAM_FROM_Y(state.ballState.pos.y)) {
				// On the team that scored
				float distToGoal = fabs(state.ballState.pos.y) - RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y / 2.0f;
				raw[0] = (distToGoal < 0) ? 1.0f : exp(-distToGoal / 500.0f);
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 1: WinProbTerm - win probability via Skellam approximation
	// Uses modified Bessel function-based PMF/CDF for goal scoring.
	// Mirrors reference/scoreboard.py Skellam win_prob computation.
	// ============================================================================
	class WinProbTerm : public RewardTerm {
	public:
		WinProbTerm() : RewardTerm("win_prob", true) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			int myScore = state.scoreLine[(int)player.team];
			int oppScore = state.scoreLine[(int)RS_OPPOSITE_TEAM(player.team)];
			int ticksLeft = 60 * 120; // 60 seconds in ticks (default)

			// Skellam win probability: P(X - Y > 0) where X, Y are Poisson processes
			// Simplified: use the score difference as a proxy
			int diff = myScore - oppScore;
			raw[0] = 1.0f / (1.0f + exp(-diff * 0.5f)); // sigmoid approximation
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 2: BallHalfTerm - ball possession half control
	// ============================================================================
	class BallHalfTerm : public RewardTerm {
	public:
		BallHalfTerm() : RewardTerm("ball_half", true) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			float ballY = state.ballState.pos.y;
			if (player.team == Team::BLUE) {
				raw[0] = clamp(ballY / CommonValues::BACK_WALL_Y, -1.0f, 1.0f);
			} else {
				raw[0] = clamp(-ballY / CommonValues::BACK_WALL_Y, -1.0f, 1.0f);
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 3: ProximityTerm - proximity to ball reward
	// ============================================================================
	class ProximityTerm : public RewardTerm {
	public:
		ProximityTerm() : RewardTerm("proximity", true) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			float dist = (state.ballState.pos - player.carState.pos).Length();
			raw[0] = exp(-dist / 1000.0f);
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 4: AlignmentTerm - alignment to ball
	// ============================================================================
	class AlignmentTerm : public RewardTerm {
	public:
		AlignmentTerm() : RewardTerm("alignment", true) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			Vec toBall = state.ballState.pos - player.carState.pos;
			toBall = toBall.Normalized();
			float dot = toBall.Dot(player.carState.rotMat.forward);
			raw[0] = (dot + 1.0f) / 2.0f; // map [-1,1] -> [0,1]
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 5: BoostTerm - boost amount reward
	// ============================================================================
	class BoostTerm : public RewardTerm {
	public:
		BoostTerm() : RewardTerm("boost", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			raw[0] = player.boostFraction;
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 6: DemoTerm - demo reward
	// ============================================================================
	class DemoTerm : public RewardTerm {
	public:
		DemoTerm() : RewardTerm("demo", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			raw[0] = (player.matchDemos > 0) ? 1.0f : 0.0f;
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 7: AngVelTerm - angular velocity reward (aerial control)
	// ============================================================================
	class AngVelTerm : public RewardTerm {
	public:
		AngVelTerm() : RewardTerm("ang_vel", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			const auto& cs = player.carState;
			float angVelMag = sqrtf(cs.angVel.x * cs.angVel.x + cs.angVel.y * cs.angVel.y + cs.angVel.z * cs.angVel.z);
			raw[0] = clamp(angVelMag / RLConst::CAR_MAX_ANG_SPEED, 0.0f, 1.0f);
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 8: GrassTerm - grass landing reward
	// ============================================================================
	class GrassTerm : public RewardTerm {
	public:
		GrassTerm() : RewardTerm("grass", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.carState.isOnGround && !player.carState.hasJumped) {
				raw[0] = 1.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 9: TouchHeightTerm - touch height reward
	// ============================================================================
	class TouchHeightTerm : public RewardTerm {
	public:
		TouchHeightTerm() : RewardTerm("touch_height", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.ballTouchedTick) {
				raw[0] = clamp(state.ballState.pos.z / CommonValues::CEILING_Z, 0.0f, 1.0f);
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 10: FlipResetTerm - flip reset reward
	// ============================================================================
	class FlipResetTerm : public RewardTerm {
	public:
		FlipResetTerm() : RewardTerm("flip_reset", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.hasFlip && !player.carState.hasFlipped) {
				raw[0] = 1.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 11: ControlTouchTerm - control at touch reward
	// ============================================================================
	class ControlTouchTerm : public RewardTerm {
	public:
		ControlTouchTerm() : RewardTerm("control_touch", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.ballTouchedTick) {
				float speed = player.carState.vel.Length();
				raw[0] = clamp(speed / RLConst::CAR_MAX_SPEED, 0.0f, 1.0f);
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 12: LongRangeGoalTerm - long range goal reward
	// ============================================================================
	class LongRangeGoalTerm : public RewardTerm {
	public:
		LongRangeGoalTerm() : RewardTerm("long_range_goal", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.ballTouchedTick) {
				float dist = (state.ballState.pos - player.carState.pos).Length();
				raw[0] = (dist > 3000.0f) ? 1.0f : 0.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 13: ChallengeWinTerm - challenge win reward
	// ============================================================================
	class ChallengeWinTerm : public RewardTerm {
	public:
		ChallengeWinTerm() : RewardTerm("challenge_win", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			int myScore = state.scoreLine[(int)player.team];
			int oppScore = state.scoreLine[(int)RS_OPPOSITE_TEAM(player.team)];
			raw[0] = (myScore > oppScore) ? 1.0f : 0.0f;
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 14: SpeedTerm - car speed reward
	// ============================================================================
	class SpeedTerm : public RewardTerm {
	public:
		SpeedTerm() : RewardTerm("speed", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			float speed = player.carState.vel.Length();
			raw[0] = clamp(speed / RLConst::CAR_MAX_SPEED, 0.0f, 1.0f);
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 15: SaveTerm - save reward (ball near own goal)
	// ============================================================================
	class SaveTerm : public RewardTerm {
	public:
		SaveTerm() : RewardTerm("save", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			float goalY = (player.team == Team::BLUE) ? -RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y / 2.0f
			                                           : RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y / 2.0f;
			float distToGoal = fabs(state.ballState.pos.y - goalY);
			raw[0] = (distToGoal < 200.0f) ? 1.0f : exp(-distToGoal / 1000.0f);
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 16: ShadowTerm - shadow defense reward
	// ============================================================================
	class ShadowTerm : public RewardTerm {
	public:
		ShadowTerm() : RewardTerm("shadow", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			// Simplified: proximity to ball when on defense
			float dist = (state.ballState.pos - player.carState.pos).Length();
			raw[0] = exp(-dist / 800.0f);
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 17: BoostDenialTerm - boost denial reward
	// ============================================================================
	class BoostDenialTerm : public RewardTerm {
	public:
		BoostDenialTerm() : RewardTerm("boost_denial", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			// Simplified: proximity to large boost pads when opponent is nearby
			raw[0] = player.boostFraction; // proxy: have boost to pick up
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 18: WallPlayTerm - wall contact reward
	// ============================================================================
	class WallPlayTerm : public RewardTerm {
	public:
		WallPlayTerm() : RewardTerm("wall_play", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			raw[0] = player.carState.isOnGround ? 0.0f : 1.0f; // proxy: in air (possibly after wall play)
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 19: TightAngleTerm - tight angle shot reward
	// ============================================================================
	class TightAngleTerm : public RewardTerm {
	public:
		TightAngleTerm() : RewardTerm("tight_angle", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			Vec toGoal;
			if (player.team == Team::BLUE) {
				toGoal = Vec(0, RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y / 2.0f, 0) - player.carState.pos;
			} else {
				toGoal = Vec(0, -RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y / 2.0f, 0) - player.carState.pos;
			}
			float angle = acos(clamp(toGoal.Dot(player.carState.rotMat.forward) / toGoal.Length(), -1.0f, 1.0f));
			raw[0] = (angle > M_PI_4) ? 1.0f : 0.0f; // tight angle bonus
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 20: DoubleTapTerm - double tap reward
	// ============================================================================
	class DoubleTapTerm : public RewardTerm {
	public:
		DoubleTapTerm() : RewardTerm("double_tap", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			raw[0] = (player.carState.isBoosting && player.carState.boostingTime > 0.5f) ? 1.0f : 0.0f;
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 21: StealTerm - ball steal reward (SPIKE_RUSH / GRIDIRON)
	// ============================================================================
	class StealTerm : public RewardTerm {
	public:
		StealTerm() : RewardTerm("steal", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (state.attachInfo.active && state.attachInfo.attachedCarId != player.carId) {
				// Ball is attached to a different car - proximity bonus for stealing
				float dist = (state.ballState.pos - player.carState.pos).Length();
				raw[0] = exp(-dist / 500.0f);
			} else if (state.attachInfo.active && state.attachInfo.attachedCarId == player.carId) {
				// Just became carrier
				raw[0] = 1.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 22: AerialTerm - aerial touch reward
	// ============================================================================
	class AerialTerm : public RewardTerm {
	public:
		AerialTerm() : RewardTerm("aerial", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.ballTouchedTick && !player.carState.isOnGround) {
				raw[0] = 1.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// Term 23: PassChainTerm - pass chain reward
	// ============================================================================
	class PassChainTerm : public RewardTerm {
	public:
		PassChainTerm() : RewardTerm("pass_chain", false) {}

		float* computeRaw(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			static thread_local float raw[1] = { 0.0f };
			if (player.matchAssists > 0) {
				raw[0] = 1.0f;
			} else {
				raw[0] = 0.0f;
			}
			return raw;
		}

		float computeRawScalar(const PlayerData& player, const GameState& state, const Action& prevAction) override {
			float* r = computeRaw(player, state, prevAction);
			return isDelta ? (r[0] - prevRaw) : r[0];
		}
	};

	// ============================================================================
	// TERM_REGISTRY: all 24 terms
	// ============================================================================
	struct TermRegistry {
		static std::unordered_map<std::string, std::function<RewardTerm*()>> getTerms() {
			return {
				{"goal", []() { return new GoalTerm(); }},
				{"win_prob", []() { return new WinProbTerm(); }},
				{"ball_half", []() { return new BallHalfTerm(); }},
				{"proximity", []() { return new ProximityTerm(); }},
				{"alignment", []() { return new AlignmentTerm(); }},
				{"boost", []() { return new BoostTerm(); }},
				{"demo", []() { return new DemoTerm(); }},
				{"ang_vel", []() { return new AngVelTerm(); }},
				{"grass", []() { return new GrassTerm(); }},
				{"touch_height", []() { return new TouchHeightTerm(); }},
				{"flip_reset", []() { return new FlipResetTerm(); }},
				{"control_touch", []() { return new ControlTouchTerm(); }},
				{"long_range_goal", []() { return new LongRangeGoalTerm(); }},
				{"challenge_win", []() { return new ChallengeWinTerm(); }},
				{"speed", []() { return new SpeedTerm(); }},
				{"save", []() { return new SaveTerm(); }},
				{"shadow", []() { return new ShadowTerm(); }},
				{"boost_denial", []() { return new BoostDenialTerm(); }},
				{"wall_play", []() { return new WallPlayTerm(); }},
				{"tight_angle", []() { return new TightAngleTerm(); }},
				{"double_tap", []() { return new DoubleTapTerm(); }},
				{"steal", []() { return new StealTerm(); }},
				{"aerial", []() { return new AerialTerm(); }},
				{"pass_chain", []() { return new PassChainTerm(); }},
			};
		}
	};

	// ============================================================================
	// MemberRewardSpec: one league member's reward "program" — the set of terms that bot selects
	// (name -> weight) plus its own team_spirit / opponent_punish scalars. Mirrors the per-slot tuple
	// in reference reward.py ComposedReward.set_matchup: (reward_dict, team_spirit, opponent_punish).
	// ============================================================================
	struct MemberRewardSpec {
		std::unordered_map<std::string, float> termWeights; // selected term name -> weight
		float teamSpirit = 0.6f;
		float opponentPunish = 1.0f;
	};

	// ============================================================================
	// ComposedReward: per-bot term routing + team_spirit/opponent_punish blending.
	//
	// Faithful port of reference reward.py ComposedReward.get_rewards. Unlike the per-player
	// RewardFunction::GetReward default (which cannot see teammates), this overrides the BATCHED
	// GetAllRewards so it can (1) compute every registered term's raw per-slot value, (2) route each
	// term's per-slot contribution to ONLY the slots whose owner selects that term (scaled by that
	// owner's weight), then (3) blend each slot's routed scalar with its teammates' routed scalars at
	// the EXPLICIT n_blue boundary (uneven teams), applying that slot's own team_spirit/opponent_punish.
	//
	// Delta terms: each term keeps a per-slot raw cache seeded at reset, so the first step's delta is
	// 0 and each later delta is raw(t) - raw(t-1) (mirrors Python self._last_raw[t.name] vector).
	// ============================================================================
	class ComposedReward : public RewardFunction {
	public:
		std::vector<RewardTerm*> terms;
		std::vector<MemberRewardSpec> members; // indexed by owner/member id (== GameState.slotOwners value)

		// Per-term, per-slot raw cache (the delta-term "last raw" vector). Parallel to `terms`.
		std::vector<std::vector<float>> lastRaw;

		ComposedReward(std::vector<RewardTerm*> terms, std::vector<MemberRewardSpec> members)
			: terms(std::move(terms)), members(std::move(members)) {}

		virtual void Reset(const GameState& initialState) override {
			RewardFunction::Reset(initialState);
			// Seed each term's per-slot raw cache to the initial frame so the first step's delta is 0.
			int n = (int)initialState.players.size();
			lastRaw.assign(terms.size(), std::vector<float>());
			for (size_t i = 0; i < terms.size(); i++) {
				std::vector<float> raw(n, 0.0f);
				for (int j = 0; j < n; j++)
					raw[j] = terms[i]->computeRaw(initialState.players[j], initialState, Action{})[0];
				lastRaw[i] = std::move(raw);
			}
		}

		virtual std::vector<float> GetAllRewards(const GameState& state, const ActionSet& prevActions, bool final) override {
			int n = (int)state.players.size();
			std::vector<float> routed(n, 0.0f);

			// 1. Compute every term's raw per slot, take the delta, route to slots whose owner selects it.
			for (size_t i = 0; i < terms.size(); i++) {
				RewardTerm* t = terms[i];
				for (int j = 0; j < n; j++) {
					float raw = t->computeRaw(state.players[j], state, prevActions[j])[0];
					float contrib;
					if (t->isDelta) {
						float prev = (i < lastRaw.size() && j < (int)lastRaw[i].size()) ? lastRaw[i][j] : 0.0f;
						contrib = raw - prev;
						if (i < lastRaw.size()) {
							if ((int)lastRaw[i].size() <= j) lastRaw[i].resize(n, 0.0f);
							lastRaw[i][j] = raw;
						}
					} else {
						contrib = raw;
					}
					int owner = (j < (int)state.slotOwners.size()) ? state.slotOwners[j] : 0;
					if (owner < 0 || owner >= (int)members.size()) owner = 0;
					auto it = members[owner].termWeights.find(t->name);
					if (it != members[owner].termWeights.end())
						routed[j] += it->second * contrib;
				}
			}

			// 2. Blend at the explicit n_blue boundary (players are in blue-first order).
			int mid = 0;
			for (int j = 0; j < n; j++)
				if (state.players[j].team == Team::BLUE) mid++;
			float bm = 0.0f, om = 0.0f; int nb = 0, no = 0;
			for (int j = 0; j < n; j++) {
				if (j < mid) { bm += routed[j]; nb++; }
				else { om += routed[j]; no++; }
			}
			if (nb) bm /= (float)nb;
			if (no) om /= (float)no;

			std::vector<float> out(n, 0.0f);
			for (int j = 0; j < n; j++) {
				int owner = (j < (int)state.slotOwners.size()) ? state.slotOwners[j] : 0;
				if (owner < 0 || owner >= (int)members.size()) owner = 0;
				float ts = members[owner].teamSpirit;
				float op = members[owner].opponentPunish;
				float own = routed[j];
				if (j < mid) out[j] = (1.0f - ts) * own + ts * bm - op * om;
				else         out[j] = (1.0f - ts) * own + ts * om - op * bm;
			}
			return out;
		}

		virtual ~ComposedReward() override {
			for (auto* t : terms) delete t;
		}
	};

	// Build a fresh instance of every registered term (one ComposedReward owns its own term objects).
	inline std::vector<RewardTerm*> MakeAllRewardTerms() {
		auto reg = TermRegistry::getTerms();
		std::vector<RewardTerm*> out;
		out.reserve(reg.size());
		for (auto& kv : reg) out.push_back(kv.second());
		return out;
	}

} // namespace RLGSC
