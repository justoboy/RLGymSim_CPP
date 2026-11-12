#pragma once
#include "OBSBuilder.h"
#include "../Gamestates/GameState.h"
#include <tuple>

// TemporalObsBuilder: produces the (q, kv, mask) tuple used by custombot.
// - q: query tensor  (batch, seq, query_features)
// - kv: key/value tensor (batch, F*n_bc + MAX_BOOST_PADS, kv_features)
// - m: attention mask (batch, F*n_bc + MAX_BOOST_PADS)
//
// Entity feature layout (30 dims each):
//   [0..2]   pos (normalized)
//   [3..5]   vel (normalized)
//   [6..8]   angVel (normalized)
//   [9..11]  rotMat forward (normalized)
//   [12..14] rotMat right (normalized)
//   [15..17] rotMat up (normalized)
//   [18]     isOnGround (car only, else 0)
//   [19]     boostAmount / boostPadActive (car/pad specific)
//   [20..21] ball: angularVelocity; pad: cooldown; car: airTimeSinceJump
//   [22]     ball: chargeLevel (spike_rush); 0 otherwise
//   [23..29] zero-padded

namespace RLGSC {

	// Game mode enum (mirrors RocketSim/src/Sim/GameMode.h)
	enum class GameMode : int {
		SOCCAR = 0,
		HOOPS,
		HEATSEEKER,
		SNOWDAY,
		SPIKE_RUSH,
		GRIDIRON,
		DROPSHOT,
	};

	// Temporal entity type
	enum class EntityType : int {
		BALL = 0,
		CAR_BLUE_0 = 1, CAR_BLUE_1 = 2, CAR_BLUE_2 = 3, CAR_BLUE_3 = 4,
		CAR_ORANGE_0 = 5, CAR_ORANGE_1 = 6, CAR_ORANGE_2 = 7, CAR_ORANGE_3 = 8,
		BOOST_PAD = 9  // all boost pads share this type index
	};

	struct TemporalObsConfig {
		int maxPlayers = 4;           // max cars per team
		int nFrames = 13;             // temporal window size F
		int maxBoostPads = 34;        // MAX_BOOST_PADS
		int queryFeatures = 0;        // computed at runtime
		int kvFeatures = 30;          // entity feature dims
	};

	class TemporalObsBuilder : public OBSBuilder {
	public:
		TemporalObsConfig config;

		// Temporal ring buffer: F frames x E entities
		struct TemporalBuffer {
			int F, E;
			std::vector<float> data;  // flattened [F][E][kvFeatures]
			std::vector<int> slotType; // entity type per slot (EntityType)
			std::vector<int> slotCarId; // car ID per slot (-1 if non-car)

			void resize(int frames, int entities, int kvDims);
		};

		TemporalBuffer temporalBuf;

		// Game mode and mutator state (set by MutatorHolder)
		GameMode activeMode = GameMode::SOCCAR;
		float gravityMult = 1.0f;
		float boostConsumptionMult = 1.0f;
		float boostStrengthMult = 1.0f;
		float ballMaxSpeedMult = 1.0f;
		float ballSizeMult = 1.0f;
		float ballBouncinessMult = 1.0f;

		// Team size (set at reset)
		int nBlue = 1, nOrange = 1;

		TemporalObsBuilder(
			int maxPlayers = 4,
			int nFrames = 13,
			int maxBoostPads = 34
		);

		void Reset(const GameState& initialState) override;
		void PreStep(const GameState& state) override;
		std::tuple<FList, FList, FList> BuildOBS(
			const PlayerData& player,
			const GameState& state,
			const Action& prevAction
		);

		// Query features computation (called internally)
		int computeQueryFeatures(const GameState& state);

	private:
		// Entity feature extraction
		void extractEntityFeatures(const PlayerData& player, const GameState& state,
		                           int frameIdx, int slotIdx, float* out);
		void extractBallFeatures(const BallState& ballState, int frameIdx, int slotIdx, float* out);
		void extractPadFeatures(const GameState& state, int padIdx, int frameIdx, int slotIdx, float* out);

		// Self-relative frame math (arctan2 -> cos/sin rotation kernel)
		struct RotatedVec {
			float x, y, z;
			RotatedVec() : x(0), y(0), z(0) {}
			RotatedVec(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
		};

		RotatedVec rotateRelative(const Vec& relPos, const Vec& forward, const Vec& right, const Vec& up);
		float safeAtan2(float y, float x);

		// Query builder: self features + direct ball block + tactical/mode/tile/team-size tail
		FList buildQuery(const PlayerData& player, const GameState& state, int queryDims);

		// Key/value builder: temporal buffer flattened
		FList buildKeyValue(const GameState& state);

		// Mask builder: constant attention mask for zero-padded slots
		FList buildMask(const GameState& state);

		// Helper: normalize vector component
		static float clamp(float v, float lo, float hi) {
			return v < lo ? lo : (v > hi ? hi : v);
		}
	};

} // namespace RLGSC
