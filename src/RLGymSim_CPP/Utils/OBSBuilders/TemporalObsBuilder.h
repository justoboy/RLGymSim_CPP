#pragma once
#include "OBSBuilder.h"
#include "../Gamestates/GameState.h"
#include <unordered_map>
#include <array>

// TemporalObsBuilder: per-agent temporal observation in the agent's self-relative,
// team-inverted frame. Each agent's flattened obs is the concatenation q ++ kv ++ m:
//   q:  query_features (208) = self entity row (39) + direct ball block (12) + tactical (3)
//       + mode token (1) + mode fields (11: ball_type + heatseeker(3) + mutator(6) + charge(1))
//       + tile block (140) + team-size (2)
//   kv: (F*n_bc + MAX_BOOST_PADS) tokens * kv_features (69); ball+cars replicated across the
//       F temporal frames (frame-major), boost pads emit ONE token at the latest frame only
//   m:  (F*n_bc + MAX_BOOST_PADS) attention mask (1 = ignore), constant per match
//
// Entity row layout (39 = 25 world + 14 profile):
//   [0] IS_SELF [1] IS_MATE [2] IS_OPP [3] IS_BALL [4] IS_BOOST
//   [5..7] pos [8..10] linVel [11..13] forward [14..16] up [17..19] angVel
//   [20] boost [21] demo [22] onGround [23] hasFlip [24] hasJump
//   [25..38] behavioral profile (14)
// Relative block (30): 6 groups {yawX, yawY, rawZ, fullX, fullY, fullZ} x 5 vector blocks
//   {pos, linVel, forward, up, angVel}; the pos block is self-relative (minus self pos).

namespace RLGSC {

	// World-frame feature column indices (mirror the reference layout).
	namespace ObsLayout {
		constexpr int IS_SELF = 0, IS_MATE = 1, IS_OPP = 2, IS_BALL = 3, IS_BOOST = 4;
		constexpr int POS = 5, LIN_VEL = 8, FW = 11, UP = 14, ANG_VEL = 17;
		constexpr int BOOST = 20, DEMO = 21, ON_GROUND = 22, HAS_FLIP = 23, HAS_JUMP = 24;
		constexpr int WORLD_FEATURES = 25;
		constexpr int N_PROFILE = 14;
		constexpr int PROFILE = 25;
		constexpr int ENTITY_FEATURES = 39;
		constexpr int RELATIVE_FEATURES = 30;
		constexpr int KV_FEATURES = 69;
		constexpr int N_BALL_QUERY = 12;
		constexpr int MAX_TEAM = 4;
		constexpr int MAX_BOOST_PADS = 34;
		constexpr int N_TEMPORAL_FRAMES = 13;
		constexpr int N_DROPSHOT_TILES = 140;
		constexpr float TILE_STATE_MAX = 2.0f;
		constexpr float POS_NORM = 2300.0f;
		constexpr float ANG_VEL_NORM = 5.5f;
	}

	// Python-reference mode token codes (the mode token + ball-type tag). These are the integer
	// codes the frozen BC obs encodes, NOT the RocketSim GameMode enum values (which differ:
	// RocketSim keeps THE_VOID and orders DROPSHOT before the custom modes).
	enum class ModeCode : int {
		SOCCAR = 0, HOOPS = 1, HEATSEEKER = 2, SNOWDAY = 3, GRIDIRON = 4, SPIKE_RUSH = 5, DROPSHOT = 6
	};

	// Maps the arena's GameMode to the Python-reference mode token code.
	inline ModeCode ModeCodeFromGameMode(GameMode gm) {
		switch (gm) {
			case GameMode::SOCCAR:     return ModeCode::SOCCAR;
			case GameMode::HOOPS:      return ModeCode::HOOPS;
			case GameMode::HEATSEEKER: return ModeCode::HEATSEEKER;
			case GameMode::SNOWDAY:    return ModeCode::SNOWDAY;
			case GameMode::DROPSHOT:   return ModeCode::DROPSHOT;
			case GameMode::SPIKE_RUSH: return ModeCode::SPIKE_RUSH;
			case GameMode::GRIDIRON:   return ModeCode::GRIDIRON;
			default:                   return ModeCode::SOCCAR;
		}
	}

	class TemporalObsBuilder : public OBSBuilder {
	public:
		// Fixed canonical 4v4-max superset layout.
		static constexpr int N_SLOTS = 2 * ObsLayout::MAX_TEAM;                 // 8 car slots
		static constexpr int N_FRAMES = ObsLayout::N_TEMPORAL_FRAMES;           // 13
		static constexpr int N_BC = 1 + N_SLOTS;                                // ball + cars = 9
		static constexpr int N_ENTITIES = N_BC + ObsLayout::MAX_BOOST_PADS;     // 43
		static constexpr int ENTITY_FEATURES = ObsLayout::ENTITY_FEATURES;      // 39
		static constexpr int KV_FEATURES = ObsLayout::KV_FEATURES;              // 69
		static constexpr int N_TOKENS = N_FRAMES * N_BC + ObsLayout::MAX_BOOST_PADS; // 151
		static constexpr int QUERY_FEATURES =
			ObsLayout::ENTITY_FEATURES + ObsLayout::N_BALL_QUERY + 157;         // 208

		// Mode + mutator state (set by the sampler mutator each reset). Stored as the arena's
		// GameMode; converted to the Python-reference ModeCode internally for the mode token.
		GameMode activeMode = GameMode::SOCCAR;
		float gravityMult = 1.0f, boostConsumptionMult = 1.0f, boostStrengthMult = 1.0f;
		float ballMaxSpeedMult = 1.0f, ballSizeMult = 1.0f, ballBouncinessMult = 1.0f;

		// Ticks per decision step. Drives the profile tracker's per-step EMA weight
		// (lam = 1 - 0.5**(dt/halfLife), dt = tickSkip/120). Defaults to the Python reference default.
		int tickSkip = 8;

		TemporalObsBuilder() = default;

		void Reset(const GameState& initialState) override;
		void PreStep(const GameState& state) override;

		// Returns flattened obs for this player: q ++ kv ++ m.
		FList BuildOBS(const PlayerData& player, const GameState& state, const Action& prevAction) override;

	private:
		// World-frame temporal ring buffers: F * E * 39 (normalized). worldBuf is the raw frame;
		// invBuf is the per-component X-mirrored (team-inverted) copy with IS_MATE/IS_OPP swapped.
		std::vector<float> worldBuf; // [f][e][39]
		std::vector<float> invBuf;   // [f][e][39]
		std::vector<int> slotCarId;  // [N_ENTITIES] car id per slot (-1 for ball/pads)

	// Constant per-match attention mask (1 = ignore), size N_TOKENS.
	std::vector<float> maskCache;

	int nActive = 0; // active car count (fixed per match)

	// --- Behavioral profile tracker (mirrors reference profile_tracker.py) ---
	// Per-car rolling estimator state. Frequent events -> time-EMA (30s half-life); rare events ->
	// log-count; ratios -> shrunk counts (N0 pseudo-counts toward neutral). Keyed by carId so a car
	// that respawns under a new id after a demo gets a fresh neutral profile.
	// Accumulators are double to match the Python tracker's Python-float (double) EMA math; the
	// export casts to float32 when written into the entity row.
	struct ProfileState {
		double pressure = 0.5;     // EMA of car_dist / max(ball_dist_to_own_goal, eps), ratio/2
		double depth = 0.0;        // EMA of signed car-behind-ball depth (own-goal frame)
		double boostLevel = 0.5;   // EMA of current boost
		double speedContact = 0.0; // EMA of speed at car-car contact
		double shotDist = 1.0;     // EMA of |ball - goal_I_shoot_at| / POS_NORM at my touches
		int nTouch = 0, nAir = 0, nWall = 0, nBump = 0, nDemo = 0;
		int nPoss = 0, nDuel = 0;
		double powerSum = 0.0, boostTouchSum = 0.0;
		bool wasDemoed = false, wasBumped = false;
	};
	std::unordered_map<uint32_t, ProfileState> profiles;
	double trackerBallSpeed = 0.0;
	int trackerLastToucherTeam = -1; // -1 = none
	bool trackerWasContested = false;

	// Advance the per-car profile estimators one step (called from PreStep before encodeFrame).
	void updateProfiles(const GameState& state);

	// Materialize a car's 14 committed profile columns (neutral defaults if not yet tracked).
	std::array<float, ObsLayout::N_PROFILE> exportProfile(uint32_t carId) const;

		// Encode the current frame into a (E,39) world-frame normalized row set, writing both the
		// world row and its team-inverted copy into the last frame of worldBuf/invBuf.
		void encodeFrame(const GameState& state);

		// Per-agent self-relative math: writes flattened (N_TOKENS,69) into out for this agent.
		void relativeAndFlatten(const std::vector<float>& buf, int selfSlot, bool isOrange,
		                        const GameState& state, const PlayerData& player, FList& out) const;

		static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
		// Mirrors np.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).
		static float sanitize(float v) {
			if (std::isnan(v)) return 0.0f;
			if (std::isinf(v)) return v > 0.0f ? 1.0f : -1.0f;
			return v;
		}
	};

} // namespace RLGSC
