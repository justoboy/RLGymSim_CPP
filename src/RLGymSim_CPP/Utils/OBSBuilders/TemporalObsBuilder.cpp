#include "TemporalObsBuilder.h"
#include "../CommonValues.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace RLGSC {

// Per-component invert (orange) and normalization vectors, length 39 (superset).
// Profile columns (25:39) are self-normalized scalars: never invert, norm 1.
// DEMO column (21) is overloaded: cars carry is_demolished (0/1), pads carry the pad respawn
// timer pre-scaled by PAD_TIMER_COEF at the assignment site; norm 1 (no divide).
static const float kInvert[ObsLayout::ENTITY_FEATURES] = {
	1, 1, 1, 1, 1,
	-1, -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, -1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static const float kNorm[ObsLayout::ENTITY_FEATURES] = {
	1, 1, 1, 1, 1,
	2300, 2300, 2300,  2300, 2300, 2300,  1, 1, 1,  1, 1, 1,  5.5f, 5.5f, 5.5f,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

// Mutator conditioning maxima (raw multiplier / max -> ~[0,1]).
static constexpr float kMutGravityMax = 2.5f;
static constexpr float kMutBoostConsumptionMax = 1.0f;
static constexpr float kMutBoostStrengthMax = 2.75f;
static constexpr float kMutBallMaxSpeedMax = 2.75f;
static constexpr float kMutBallSizeMax = 1.5f;
static constexpr float kMutBallBouncinessMax = 1.5f;
static constexpr float kLinVelCoef = 1.0f / 2300.0f;

// --- Behavioral profile tracker constants (mirror reference profile_tracker.py) ---
static constexpr double kProfileN0 = 4.0;          // pseudo-counts for shrunk ratios
static constexpr double kAirTouchZ = 300.0;         // ball height above which a touch is an "air touch"
static constexpr double kWallMargin = 300.0;        // ball within this of side/back wall at touch -> wall touch
static constexpr double kSideWallX = 4096.0;
static constexpr double kBackWallY = 5120.0;
static constexpr double kEmaHalfLife = 30.0;        // seconds
static constexpr double kProfilePosNorm = 2300.0;   // uu normalization for distance/speed columns
static constexpr double kControlRadius = 2.0 * CommonValues::BALL_RADIUS; // contested 50/50 radius

// Euclidean norm of a Vec (mirrors np.linalg.norm on a 3-vector).
static inline double norm3(const Vec& v) {
	return std::sqrt((double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z);
}
static inline double clampd(double v, double lo, double hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================================
// encodeFrame: write one normalized world-frame (E,39) frame into worldBuf[frameIdx]
// and its team-inverted copy into invBuf[frameIdx].
// ============================================================================

void TemporalObsBuilder::encodeFrame(const GameState& state) {
	const int E = N_ENTITIES;
	const int F = N_FRAMES;
	const int lastFrame = F - 1;
	float* w = worldBuf.data() + (size_t)lastFrame * E * ObsLayout::ENTITY_FEATURES;
	float* v = invBuf.data() + (size_t)lastFrame * E * ObsLayout::ENTITY_FEATURES;

	// --- Ball (entity 0) ---
	{
		float* row = w;
		const BallState& bs = state.ballState;
		row[ObsLayout::IS_BALL] = 1.0f;
		row[ObsLayout::POS + 0] = bs.pos.x / ObsLayout::POS_NORM;
		row[ObsLayout::POS + 1] = bs.pos.y / ObsLayout::POS_NORM;
		row[ObsLayout::POS + 2] = bs.pos.z / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 0] = bs.vel.x / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 1] = bs.vel.y / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 2] = bs.vel.z / ObsLayout::POS_NORM;
		// Real ball spin (documented divergence from the frozen Python reference; BC retrained).
		row[ObsLayout::ANG_VEL + 0] = bs.angVel.x / ObsLayout::ANG_VEL_NORM;
		row[ObsLayout::ANG_VEL + 1] = bs.angVel.y / ObsLayout::ANG_VEL_NORM;
		row[ObsLayout::ANG_VEL + 2] = bs.angVel.z / ObsLayout::ANG_VEL_NORM;
	}

	// --- Cars (entities 1..n_slots) ---
	for (int i = 0; i < (int)state.players.size(); i++) {
		const PlayerData& pd = state.players[i];
		const CarState& cs = pd.carState;
		float* row = w + (size_t)(1 + i) * ObsLayout::ENTITY_FEATURES;
		int team = (int)pd.team;
		row[ObsLayout::IS_MATE] = 1.0f - team;
		row[ObsLayout::IS_OPP] = (float)team;
		row[ObsLayout::POS + 0] = cs.pos.x / ObsLayout::POS_NORM;
		row[ObsLayout::POS + 1] = cs.pos.y / ObsLayout::POS_NORM;
		row[ObsLayout::POS + 2] = cs.pos.z / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 0] = cs.vel.x / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 1] = cs.vel.y / ObsLayout::POS_NORM;
		row[ObsLayout::LIN_VEL + 2] = cs.vel.z / ObsLayout::POS_NORM;
		row[ObsLayout::ANG_VEL + 0] = cs.angVel.x / ObsLayout::ANG_VEL_NORM;
		row[ObsLayout::ANG_VEL + 1] = cs.angVel.y / ObsLayout::ANG_VEL_NORM;
		row[ObsLayout::ANG_VEL + 2] = cs.angVel.z / ObsLayout::ANG_VEL_NORM;
		row[ObsLayout::FW + 0] = cs.rotMat.forward.x;
		row[ObsLayout::FW + 1] = cs.rotMat.forward.y;
		row[ObsLayout::FW + 2] = cs.rotMat.forward.z;
		row[ObsLayout::UP + 0] = cs.rotMat.up.x;
		row[ObsLayout::UP + 1] = cs.rotMat.up.y;
		row[ObsLayout::UP + 2] = cs.rotMat.up.z;
		row[ObsLayout::BOOST] = clampf(cs.boost / 100.0f, 0.0f, 1.0f);
		row[ObsLayout::DEMO] = cs.isDemoed ? 1.0f : 0.0f;
		row[ObsLayout::ON_GROUND] = cs.isOnGround ? 1.0f : 0.0f;
		row[ObsLayout::HAS_FLIP] = cs.hasFlipped ? 1.0f : 0.0f;
		row[ObsLayout::HAS_JUMP] = cs.hasJumped ? 1.0f : 0.0f;
		std::array<float, ObsLayout::N_PROFILE> prof = exportProfile(pd.carId);
		for (int p = 0; p < ObsLayout::N_PROFILE; p++)
			row[ObsLayout::PROFILE + p] = prof[p];
	}

	// --- Boost pads (entities n_bc..E-1) ---
	for (int p = 0; p < ObsLayout::MAX_BOOST_PADS; p++) {
		float* row = w + (size_t)(N_BC + p) * ObsLayout::ENTITY_FEATURES;
		if (p < state.nActivePads) {
			const Vec& loc = CommonValues::BOOST_LOCATIONS[p];
			row[ObsLayout::IS_BOOST] = 1.0f;
			row[ObsLayout::POS + 0] = loc.x / ObsLayout::POS_NORM;
			row[ObsLayout::POS + 1] = loc.y / ObsLayout::POS_NORM;
			row[ObsLayout::POS + 2] = loc.z / ObsLayout::POS_NORM;
			row[ObsLayout::BOOST] = 1.0f;
			row[ObsLayout::DEMO] = clampf(state.boostPadTimers[p] * 0.1f, -1.0f, 1.0f);
		}
	}

	// Build the team-inverted copy: per-component invert, then swap IS_MATE/IS_OPP.
	for (int e = 0; e < E; e++) {
		float* src = w + (size_t)e * ObsLayout::ENTITY_FEATURES;
		float* dst = v + (size_t)e * ObsLayout::ENTITY_FEATURES;
		for (int c = 0; c < ObsLayout::ENTITY_FEATURES; c++)
			dst[c] = src[c] * kInvert[c];
		float mate = dst[ObsLayout::IS_MATE];
		dst[ObsLayout::IS_MATE] = dst[ObsLayout::IS_OPP];
		dst[ObsLayout::IS_OPP] = mate;
	}
}

// ============================================================================
// Behavioral profile tracker (mirrors reference profile_tracker.py)
// ============================================================================

std::array<float, ObsLayout::N_PROFILE> TemporalObsBuilder::exportProfile(uint32_t carId) const {
	auto it = profiles.find(carId);
	// Neutral defaults if not yet tracked (mirrors the set_state neutral priors).
	ProfileState p;
	if (it != profiles.end())
		p = it->second;
	const double n = (double)p.nTouch;
	const double N0 = kProfileN0;
	std::array<float, ObsLayout::N_PROFILE> out;
	out[0]  = (float)p.pressure;                                        // 1  pressure_ratio
	out[1]  = (float)p.depth;                                           // 2  depth_vs_ball
	out[2]  = (float)((p.nAir + N0 / 2.0) / (n + N0));                  // 3  air_touch_ratio
	out[3]  = (float)((p.nWall + N0 / 2.0) / (n + N0));                 // 4  wall_touch_ratio
	out[4]  = (float)((p.powerSum / (n + N0)) / kProfilePosNorm);       // 5  touch_power
	out[5]  = (float)((p.boostTouchSum + N0 * 0.5) / (n + N0));         // 6  boost_at_touch
	out[6]  = (float)p.boostLevel;                                      // 7  avg_boost_level
	out[7]  = (float)std::log1p((double)p.nBump);                       // 8  log_bump_count
	out[8]  = (float)std::log1p((double)p.nDemo);                        // 9  log_demo_count
	out[9]  = (float)p.speedContact;                                    // 10 avg_speed_at_contact
	out[10] = (float)std::log1p(n);                                     // 11 log_touch_count
	out[11] = (float)p.shotDist;                                        // 12 shot_dist (sniper)
	out[12] = (float)std::log1p((double)p.nPoss);                        // 13 log_possession_gain (thief)
	out[13] = (float)std::log1p((double)p.nDuel);                        // 14 log_duel_win (challenger)
	return out;
}

void TemporalObsBuilder::updateProfiles(const GameState& state) {
	const double lam = 1.0 - std::pow(0.5, ((double)tickSkip / CommonValues::kTicksPerSecond) / kEmaHalfLife);
	const Vec ballPos = state.ballState.pos;
	const Vec ballVel = state.ballState.vel;
	const double ballSpeed = norm3(ballVel);

	// Possession-change + duel detection need the PREVIOUS toucher team (before this frame's touches).
	const int prevToucherTeam = trackerLastToucherTeam;

	// Contested 50/50 = cars from BOTH teams within control radius of the ball at once.
	bool blueIn = false, orangeIn = false;
	for (const PlayerData& pd : state.players) {
		if (norm3(pd.carState.pos - ballPos) < kControlRadius) {
			if ((int)pd.team == (int)Team::BLUE) blueIn = true;
			else orangeIn = true;
		}
	}
	const bool contested = blueIn && orangeIn;
	const bool duelRising = contested && !trackerWasContested;
	// Winner = team the ball travels AWAY from. Ball moving +y (toward orange net) => blue won.
	const int duelWinner = ballVel.y > 0.0 ? (int)Team::BLUE : (int)Team::ORANGE;
	trackerWasContested = contested;

	for (const PlayerData& pd : state.players) {
		const uint32_t id = pd.carId;
		const int team = (int)pd.team;
		const Vec pos = pd.carState.pos;

		ProfileState& p = profiles[id]; // default-constructs to neutral priors on first sight

		// 1. pressure_ratio: EMA of car_dist / max(ball_dist_to_own_goal, eps), ratio/2
		const double ownGoalY = team == (int)Team::BLUE ? -kBackWallY : kBackWallY;
		const double ballToOwn = std::hypot((double)ballPos.x, (double)ballPos.y - ownGoalY);
		const double carDist = norm3(pos - ballPos);
		const double ratio = clampd(carDist / std::max(ballToOwn, 1.0), 0.0, 2.0) / 2.0;
		p.pressure += lam * (ratio - p.pressure);

		// 2. depth_vs_ball: EMA of signed car-behind-ball depth (own-goal frame)
		const double depth = team == (int)Team::BLUE
			? ((double)ballPos.y - (double)pos.y)
			: ((double)pos.y - (double)ballPos.y);
		p.depth += lam * (depth / kProfilePosNorm - p.depth);

		// 7. avg_boost_level: EMA of current boost
		p.boostLevel += lam * ((double)pd.carState.boost / 100.0 - p.boostLevel);

		// touch-conditioned columns (ballTouchedStep mirrors Python ball_touches > 0)
		if (pd.ballTouchedStep) {
			p.nTouch += 1;
			if (ballPos.z > kAirTouchZ) p.nAir += 1;
			if (std::abs((double)ballPos.x) > kSideWallX - kWallMargin
				|| std::abs((double)ballPos.y) > kBackWallY - kWallMargin)
				p.nWall += 1;
			// 5. touch_power: delta |v_ball| at their touches
			p.powerSum += ballSpeed - trackerBallSpeed;
			// 6. boost_at_touch: own boost at own touches
			p.boostTouchSum += (double)pd.carState.boost / 100.0;
			// 12. shot_dist: EMA of distance from ball at MY touch to the goal I shoot at
			const Vec goal = team == (int)Team::BLUE ? CommonValues::ORANGE_GOAL_BACK
			                                        : CommonValues::BLUE_GOAL_BACK;
			const double d = norm3(ballPos - goal) / kProfilePosNorm;
			p.shotDist += lam * (d - p.shotDist);
			// 13. possession_gain: I took the ball FROM the opponent
			if (prevToucherTeam != -1 && prevToucherTeam != team) p.nPoss += 1;
			trackerLastToucherTeam = team;
		}

		// 14. duel_win: my team won a contested 50/50 (ball direction decided the winner)
		if (duelRising && team == duelWinner) p.nDuel += 1;

		// bump events (rising edge so multi-tick contact cooldown doesn't double-count)
		const bool bumped = pd.carState.carContact.cooldownTimer > 0.0f;
		if (bumped && !p.wasBumped) {
			p.nBump += 1;
			const double speed = norm3(pd.carState.vel) / kProfilePosNorm;
			p.speedContact += lam * (speed - p.speedContact);
		}
		p.wasBumped = bumped;

		// demo events (rising edge)
		const bool demoed = pd.carState.isDemoed;
		if (demoed && !p.wasDemoed) p.nDemo += 1;
		p.wasDemoed = demoed;
	}

	trackerBallSpeed = ballSpeed;
}

// ============================================================================
// Reset / PreStep
// ============================================================================

void TemporalObsBuilder::Reset(const GameState& initialState) {
	nActive = (int)initialState.players.size();

	worldBuf.assign((size_t)N_FRAMES * N_ENTITIES * ObsLayout::ENTITY_FEATURES, 0.0f);
	invBuf.assign((size_t)N_FRAMES * N_ENTITIES * ObsLayout::ENTITY_FEATURES, 0.0f);

	slotCarId.assign(N_ENTITIES, -1);
	for (int i = 0; i < nActive; i++)
		slotCarId[1 + i] = (int)initialState.players[i].carId;

	// Seed the behavioral profile tracker (mirrors Python set_state): fresh neutral profiles per
	// live car, no cross-match persistence. encodeFrame below reads these via exportProfile, so the
	// reset obs carries neutral profile columns (the first updateProfiles advances them one step).
	profiles.clear();
	for (const PlayerData& pd : initialState.players) {
		ProfileState p;
		p.wasDemoed = pd.carState.isDemoed;
		p.wasBumped = pd.carState.carContact.cooldownTimer > 0.0f;
		profiles[pd.carId] = p;
	}
	trackerBallSpeed = norm3(initialState.ballState.vel);
	trackerLastToucherTeam = -1;
	trackerWasContested = false;

	// Encode the initial frame once, then broadcast it across all F frames.
	encodeFrame(initialState);
	const float* last = worldBuf.data() + (size_t)(N_FRAMES - 1) * N_ENTITIES * ObsLayout::ENTITY_FEATURES;
	float* lastInv = invBuf.data() + (size_t)(N_FRAMES - 1) * N_ENTITIES * ObsLayout::ENTITY_FEATURES;
	for (int f = 0; f < N_FRAMES - 1; f++) {
		memcpy(worldBuf.data() + (size_t)f * N_ENTITIES * ObsLayout::ENTITY_FEATURES,
		       last, (size_t)N_ENTITIES * ObsLayout::ENTITY_FEATURES * sizeof(float));
		memcpy(invBuf.data() + (size_t)f * N_ENTITIES * ObsLayout::ENTITY_FEATURES,
		       lastInv, (size_t)N_ENTITIES * ObsLayout::ENTITY_FEATURES * sizeof(float));
	}

	// Constant per-match attention mask (1 = ignore). A masked CAR slot is masked at F flat
	// indices (one per frame); a masked PAD slot is masked at exactly one flat index.
	maskCache.assign(N_TOKENS, 0.0f);
	for (int e = 1 + nActive; e < N_BC; e++)
		for (int f = 0; f < N_FRAMES; f++)
			maskCache[(size_t)f * N_BC + e] = 1.0f;
	for (int p = initialState.nActivePads; p < ObsLayout::MAX_BOOST_PADS; p++)
		maskCache[(size_t)N_FRAMES * N_BC + p] = 1.0f;
}

void TemporalObsBuilder::PreStep(const GameState& state) {
	const int E = N_ENTITIES;
	const int stride = (size_t)E * ObsLayout::ENTITY_FEATURES;

	// Shift both buffers forward by one frame (newest frame last).
	if (N_FRAMES > 1) {
		memmove(worldBuf.data(), worldBuf.data() + stride, (N_FRAMES - 1) * stride * sizeof(float));
		memmove(invBuf.data(), invBuf.data() + stride, (N_FRAMES - 1) * stride * sizeof(float));
	}

	// Re-encode the (now) last frame and refresh the car-slot id map.
	for (int i = 0; i < (int)state.players.size(); i++)
		slotCarId[1 + i] = (int)state.players[i].carId;
	nActive = (int)state.players.size();
	// Advance the per-car profile estimators one step, then encode (encodeFrame reads exportProfile).
	updateProfiles(state);
	encodeFrame(state);
}

// ============================================================================
// BuildOBS: per-agent self-relative math + query assembly + flatten -> q ++ kv ++ m
// ============================================================================

FList TemporalObsBuilder::BuildOBS(
	const PlayerData& player,
	const GameState& state,
	const Action& prevAction
) {
	(void)prevAction;

	// Locate this player's fixed car slot (slot 1+i where slotCarId[slot]==carId).
	int selfSlot = -1;
	for (int e = 1; e < N_BC; e++)
		if (slotCarId[e] == (int)player.carId) { selfSlot = e; break; }
	RG_ASSERT(selfSlot >= 0);

	const bool isOrange = player.team == Team::ORANGE;
	const std::vector<float>& buf = isOrange ? invBuf : worldBuf;

	// --- Self-relative math for this agent over the whole temporal window ---
	// self_feats = the agent's own entity row at the latest frame (team-inverted).
	const int EF = ObsLayout::ENTITY_FEATURES;
	const float* selfRow = buf.data() + (size_t)(N_FRAMES - 1) * N_ENTITIES * EF
	                     + (size_t)selfSlot * EF;

	const float f0 = selfRow[ObsLayout::FW + 0];
	const float f1 = selfRow[ObsLayout::FW + 1];
	const float f2 = selfRow[ObsLayout::FW + 2];
	const float u0 = selfRow[ObsLayout::UP + 0];
	const float u1 = selfRow[ObsLayout::UP + 1];
	const float u2 = selfRow[ObsLayout::UP + 2];
	// left = cross(up, forward)
	const float l0 = u1 * f2 - u2 * f1;
	const float l1 = u2 * f0 - u0 * f2;
	const float l2 = u0 * f1 - u1 * f0;
	const float px = selfRow[ObsLayout::POS + 0];
	const float py = selfRow[ObsLayout::POS + 1];
	const float pz = selfRow[ObsLayout::POS + 2];
	const float pitch = atan2f(f2, sqrtf(f0 * f0 + f1 * f1));
	const float yaw = atan2f(f1, f0);
	const float roll = atan2f(l2, u2);
	const float cr = cosf(roll), sr = sinf(roll);
	const float cp = cosf(pitch), sp = sinf(pitch);
	const float cy = cosf(yaw), sy = sinf(yaw);

	// Build this agent's flattened kv token set (N_TOKENS, 69).
	FList kv((size_t)N_TOKENS * ObsLayout::KV_FEATURES, 0.0f);

	for (int f = 0; f < N_FRAMES; f++) {
		for (int e = 0; e < N_BC; e++) {
			const float* src = buf.data() + (size_t)f * N_ENTITIES * EF + (size_t)e * EF;
			// 5 world-vector blocks at cols {5,8,11,14,17}; pos block (k=0) is self-relative.
			float xs[5], ys[5], zs[5];
			for (int k = 0; k < 5; k++) {
				int cc = ObsLayout::POS + 3 * k;
				float v0 = src[cc], v1 = src[cc + 1], v2 = src[cc + 2];
				if (k == 0) { v0 -= px; v1 -= py; v2 -= pz; }
				xs[k] = v0; ys[k] = v1; zs[k] = v2;
			}
			// token index = f*n_bc + e (frame-major)
			float* dst = kv.data() + ((size_t)f * N_BC + e) * ObsLayout::KV_FEATURES;
			for (int c = 0; c < EF; c++) dst[c] = src[c];
			const int rel0 = EF;
			for (int k = 0; k < 5; k++) {
				float v0 = xs[k], v1 = ys[k], v2 = zs[k];
				float frx = cy * v0 - sy * v1;
				float fry = sy * v0 + cy * v1;
				float crx = cp * cy * v0 + (sr * sp * cy - cr * sy) * v1 - (cr * sp * cy + sr * sy) * v2;
				float cry = cp * sy * v0 + (sr * sp * sy + cr * cy) * v1 - (cr * sp * sy - sr * cy) * v2;
				float crz = sp * v0 - cp * sr * v1 + cp * cr * v2;
				dst[rel0 + k] = sanitize(frx);
				dst[rel0 + 5 + k] = sanitize(fry);
				dst[rel0 + 10 + k] = sanitize(v2);
				dst[rel0 + 15 + k] = sanitize(crx);
				dst[rel0 + 20 + k] = sanitize(cry);
				dst[rel0 + 25 + k] = sanitize(crz);
			}
			// self-row IS_* override (all frames of this agent's entity row)
			if (e == selfSlot) {
				for (int ff = 0; ff < N_FRAMES; ff++) {
					float* dr = kv.data() + ((size_t)ff * N_BC + e) * ObsLayout::KV_FEATURES;
					dr[ObsLayout::IS_SELF] = 1.0f;
					dr[ObsLayout::IS_MATE] = 0.0f;
					dr[ObsLayout::IS_OPP] = 0.0f;
				}
			}
		}
	}

	// Pads: one token each, latest frame only (buffer slots N_BC..N_BC+P-1). The pad's 5 vector
	// blocks get the SAME self-relative rotation as ball/cars (the pos block is made self-relative,
	// the fw/up/angVel blocks are zero for a static pad so they rotate to zero); Python's
	// _relative_core rotates every entity in the buffer, pads included, so the 30-col relative block
	// must be filled here too (not left zero).
	for (int p = 0; p < ObsLayout::MAX_BOOST_PADS; p++) {
		const float* src = buf.data() + (size_t)(N_FRAMES - 1) * N_ENTITIES * EF + (size_t)(N_BC + p) * EF;
		float* dst = kv.data() + ((size_t)N_FRAMES * N_BC + p) * ObsLayout::KV_FEATURES;
		float xs[5], ys[5], zs[5];
		for (int k = 0; k < 5; k++) {
			int cc = ObsLayout::POS + 3 * k;
			float v0 = src[cc], v1 = src[cc + 1], v2 = src[cc + 2];
			if (k == 0) { v0 -= px; v1 -= py; v2 -= pz; }
			xs[k] = v0; ys[k] = v1; zs[k] = v2;
		}
		for (int c = 0; c < EF; c++) dst[c] = src[c];
		const int rel0 = EF;
		for (int k = 0; k < 5; k++) {
			float v0 = xs[k], v1 = ys[k], v2 = zs[k];
			float frx = cy * v0 - sy * v1;
			float fry = sy * v0 + cy * v1;
			float crx = cp * cy * v0 + (sr * sp * cy - cr * sy) * v1 - (cr * sp * cy + sr * sy) * v2;
			float cry = cp * sy * v0 + (sr * sp * sy + cr * cy) * v1 - (cr * sp * sy - sr * cy) * v2;
			float crz = sp * v0 - cp * sr * v1 + cp * cr * v2;
			dst[rel0 + k] = sanitize(frx);
			dst[rel0 + 5 + k] = sanitize(fry);
			dst[rel0 + 10 + k] = sanitize(v2);
			dst[rel0 + 15 + k] = sanitize(crx);
			dst[rel0 + 20 + k] = sanitize(cry);
			dst[rel0 + 25 + k] = sanitize(crz);
		}
	}

	// --- Query assembly ---
	FList q;
	q.reserve(QUERY_FEATURES);

	// Self entity row (39) = latest-frame self features (entity block only).
	const float* qSelf = kv.data() + (size_t)((N_FRAMES - 1) * N_BC + selfSlot) * ObsLayout::KV_FEATURES;
	for (int c = 0; c < EF; c++) q.push_back(qSelf[c]);

	// Direct ball block (12): absolute ball pos/vel + ball-relative-to-self pos/vel.
	const float* ballRow = buf.data() + (size_t)(N_FRAMES - 1) * N_ENTITIES * EF; // entity 0
	float bpx = ballRow[ObsLayout::POS + 0], bpy = ballRow[ObsLayout::POS + 1], bpz = ballRow[ObsLayout::POS + 2];
	float bvx = ballRow[ObsLayout::LIN_VEL + 0], bvy = ballRow[ObsLayout::LIN_VEL + 1], bvz = ballRow[ObsLayout::LIN_VEL + 2];
	float qspx = qSelf[ObsLayout::POS + 0], qspy = qSelf[ObsLayout::POS + 1], qspz = qSelf[ObsLayout::POS + 2];
	float qsvx = qSelf[ObsLayout::LIN_VEL + 0], qsvy = qSelf[ObsLayout::LIN_VEL + 1], qsvz = qSelf[ObsLayout::LIN_VEL + 2];
	q += { bpx, bpy, bpz };                 // absolute ball pos (3)
	q += { bvx, bvy, bvz };                 // absolute ball vel (3)
	q += { bpx - qspx, bpy - qspy, bpz - qspz }; // ball-relative pos (3)
	q += { bvx - qsvx, bvy - qsvy, bvz - qsvz }; // ball-relative vel (3)

	// Tactical (3): goal_diff (team-signed), time_left, is_overtime.
	int blueScore = state.scoreLine[0];
	int orangeScore = state.scoreLine[1];
	float goalDiff = clampf((float)(blueScore - orangeScore), -5.0f, 5.0f) / 5.0f;
	float gd = isOrange ? -goalDiff : goalDiff;
	bool isOvertime = state.isOvertime;
	float timeLeft = isOvertime ? 0.0f : clampf(state.ticksLeft, 0.0f, 300.0f * 120.0f) / (120.0f * 300.0f);
	q += { gd, timeLeft, isOvertime ? 1.0f : 0.0f };

	// Mode token (1) + ball_type (1).
	const ModeCode mc = ModeCodeFromGameMode(activeMode);
	int modeCode = (int)mc;
	q += (float)modeCode;
	float ballType = (mc == ModeCode::SPIKE_RUSH) ? 1.0f : 0.0f;
	q += ballType;

	// Heatseeker (3): zero unless heatseeker mode.
	if (mc == ModeCode::HEATSEEKER) {
		float hsd = state.ballState.hsInfo.yTargetDir * (isOrange ? -1.0f : 1.0f);
		float hss = state.ballState.hsInfo.curTargetSpeed * kLinVelCoef;
		float hst = state.ballState.hsInfo.timeSinceHit;
		q += { sanitize(hsd), sanitize(hss), sanitize(hst) };
	} else {
		q += { 0.0f, 0.0f, 0.0f };
	}

	// Mutator (6).
	q += clampf(gravityMult / kMutGravityMax, 0.0f, 1.0f);
	q += clampf(boostConsumptionMult / kMutBoostConsumptionMax, 0.0f, 1.0f);
	q += clampf(boostStrengthMult / kMutBoostStrengthMax, 0.0f, 1.0f);
	q += clampf(ballMaxSpeedMult / kMutBallMaxSpeedMax, 0.0f, 1.0f);
	q += clampf(ballSizeMult / kMutBallSizeMax, 0.0f, 1.0f);
	q += clampf(ballBouncinessMult / kMutBallBouncinessMax, 0.0f, 1.0f);

	// Dropshot ball-charge tier (1): raw 0/1/2 / TILE_STATE_MAX; zero for non-dropshot.
	float charge = (mc == ModeCode::DROPSHOT)
		? (float)state.ballState.dsInfo.chargeLevel / ObsLayout::TILE_STATE_MAX : 0.0f;
	q += charge;

	// Tile block (140): normalized tile damage; zeroed for non-dropshot.
	if (mc == ModeCode::DROPSHOT) {
		for (int t = 0; t < ObsLayout::N_DROPSHOT_TILES; t++)
			q += state.tileStates[t] / ObsLayout::TILE_STATE_MAX;
	} else {
		for (int t = 0; t < ObsLayout::N_DROPSHOT_TILES; t++) q += 0.0f;
	}

	// Team-size (2): my team count / MAX_TEAM, enemy count / MAX_TEAM.
	int nBlue = 0, nOrange = 0;
	for (int i = 0; i < nActive; i++)
		(state.players[i].team == Team::ORANGE ? nOrange : nBlue)++;
	int my = isOrange ? nOrange : nBlue;
	int opp = nActive - my;
	q += (float)my / ObsLayout::MAX_TEAM;
	q += (float)opp / ObsLayout::MAX_TEAM;

	// --- Concatenate q ++ kv ++ m ---
	FList obs;
	obs.reserve(q.size() + kv.size() + maskCache.size());
	obs.insert(obs.end(), q.begin(), q.end());
	obs.insert(obs.end(), kv.begin(), kv.end());
	obs.insert(obs.end(), maskCache.begin(), maskCache.end());
	return obs;
}

} // namespace RLGSC
