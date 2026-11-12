#include "TemporalObsBuilder.h"
#include "../CommonValues.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace RLGSC {

// ============================================================================
// TemporalBuffer
// ============================================================================

void TemporalObsBuilder::TemporalBuffer::resize(int frames, int entities, int kvDims) {
	F = frames;
	E = entities;
	data.assign(frames * entities * kvDims, 0.0f);
	slotType.resize(entities, (int)EntityType::BALL);
	slotCarId.resize(entities, -1);
}

// ============================================================================
// Constructor
// ============================================================================

TemporalObsBuilder::TemporalObsBuilder(int maxPlayers, int nFrames, int maxBoostPads) {
	config.maxPlayers = maxPlayers;
	config.nFrames = nFrames;
	config.maxBoostPads = maxBoostPads;
	config.queryFeatures = 0; // computed at runtime in computeQueryFeatures()
	config.kvFeatures = 30;

	// Total entities: ball + maxCars*2 + boostPads
	int totalEntities = 1 + (maxPlayers * 2) + maxBoostPads;
	temporalBuf.resize(nFrames, totalEntities, config.kvFeatures);
}

// ============================================================================
// Reset / PreStep
// ============================================================================

void TemporalObsBuilder::Reset(const GameState& initialState) {
	// Zero out temporal buffer on reset
	for (auto& v : temporalBuf.data) v = 0.0f;
}

void TemporalObsBuilder::PreStep(const GameState& state) {
	// Shift temporal buffer forward by one frame and push current frame
	int F = temporalBuf.F;
	int E = temporalBuf.E;
	int kvDims = config.kvFeatures;

	// Shift: [1..F-1] -> [0..F-2]
	if (F > 1) {
		int shiftBytes = (F - 1) * E * kvDims * sizeof(float);
		memmove(temporalBuf.data.data(),
		        temporalBuf.data.data() + E * kvDims,
		        shiftBytes);
	}

	// Fill last frame [F-1] with current state features
	int lastFrame = F - 1;

	// Entity 0: ball
	extractBallFeatures(state.ballState, lastFrame, 0, temporalBuf.data.data(), E, kvDims);

	// Entities 1..nBlue+nOrange: cars (players)
	for (int i = 0; i < (int)state.players.size(); i++) {
		int slotIdx = 1 + i;
		if (slotIdx < E) {
			extractEntityFeatures(state.players[i], state, lastFrame, slotIdx, temporalBuf.data.data(), E, kvDims);
		}
	}

	// Remaining slots: boost pads
	for (int p = 0; p < config.maxBoostPads && ((1 + (int)state.players.size()) + p) < E; p++) {
		int slotIdx = 1 + (int)state.players.size() + p;
		extractPadFeatures(state, p, lastFrame, slotIdx, temporalBuf.data.data(), E, kvDims);
	}
}

// ============================================================================
// BuildOBS: produces flattened obs for this player: q ++ kv ++ m
// ============================================================================

FList TemporalObsBuilder::BuildOBS(
	const PlayerData& player,
	const GameState& state,
	const Action& prevAction
) {
	int queryDims = computeQueryFeatures(state);
	FList q = buildQuery(player, state, queryDims);
	FList kv = buildKeyValue(state);
	FList m = buildMask(state);

	FList obs;
	obs.reserve(q.size() + kv.size() + m.size());
	obs.insert(obs.end(), q.begin(), q.end());
	obs.insert(obs.end(), kv.begin(), kv.end());
	obs.insert(obs.end(), m.begin(), m.end());
	return obs;
}

// ============================================================================
// computeQueryFeatures: counts dims for query tensor tail
// ============================================================================

int TemporalObsBuilder::computeQueryFeatures(const GameState& state) {
	// Self features: 25 (world) + 14 (profile) = 39
	// Direct ball block: 12
	// Tactical/mode/mutator/tile/team-size tail: computed dynamically
	int baseDims = 39 + 12; // self + ball

	// Mode-specific query features
	int modeTail = 0;
	switch (activeMode) {
	case GameMode::SPIKE_RUSH:
		modeTail = 4; // activation_delay, attach_radius, release_cooldown, charge_level
		break;
	case GameMode::GRIDIRON:
		modeTail = 3; // roof_offset_z, reacquire_cooldown, football_semi_major
		break;
	case GameMode::DROPSHOT:
		modeTail = 2; // tiles_total, tiles_blue_damaged
		break;
	default:
		modeTail = 0;
		break;
	}

	// Mutator tail (6 dims)
	modeTail += 6;

	// Team-size tail (2 dims: nBlue, nOrange as fractions)
	modeTail += 2;

	return baseDims + modeTail;
}

// ============================================================================
// rotateRelative: arctan2 -> cos/sin rotation kernel
// ============================================================================

	TemporalObsBuilder::RotatedVec TemporalObsBuilder::rotateRelative(
	const Vec& relPos, const Vec& forward, const Vec& right, const Vec& up) {
	// Project relative position onto car's local axes
	float fx = relPos.Dot(forward);
	float rx = relPos.Dot(right);
	float ux = relPos.Dot(up);
	return RotatedVec(fx, rx, ux);
}

float TemporalObsBuilder::safeAtan2(float y, float x) {
	// Safe atan2 with small epsilon to avoid NaN
	const float EPS = 1e-8f;
	if (fabs(x) < EPS && fabs(y) < EPS) return 0.0f;
	return atan2f(y, x);
}

// ============================================================================
// extractEntityFeatures: car entity feature extraction (30 dims)
// ============================================================================

void TemporalObsBuilder::extractEntityFeatures(
	const PlayerData& player, const GameState& state,
	int frameIdx, int slotIdx, float* out, int E, int kvDims) {

	// Zero out this entity's features for current frame
	float* base = out + frameIdx * E * kvDims + slotIdx * kvDims;
	memset(base, 0, kvDims * sizeof(float));

	const auto& cs = player.carState;
	const Vec pos = cs.pos;
	const Vec vel = cs.vel;
	const Vec angVel = cs.angVel;

	// [0..2] pos (normalized by arena bounds)
	base[0] = pos.x / CommonValues::SIDE_WALL_X;
	base[1] = pos.y / CommonValues::BACK_WALL_Y;
	base[2] = pos.z / CommonValues::CEILING_Z;

	// [3..5] vel (normalized)
	base[3] = vel.x / CommonValues::CAR_MAX_SPEED;
	base[4] = vel.y / CommonValues::CAR_MAX_SPEED;
	base[5] = vel.z / CommonValues::CAR_MAX_SPEED;

	// [6..8] angVel (normalized)
	base[6] = angVel.x / CommonValues::CAR_MAX_ANG_VEL;
	base[7] = angVel.y / CommonValues::CAR_MAX_ANG_VEL;
	base[8] = angVel.z / CommonValues::CAR_MAX_ANG_VEL;

	// [9..11] rotMat forward
	const auto& rm = cs.rotMat;
	base[9] = rm.forward.x; base[10] = rm.forward.y; base[11] = rm.forward.z;

	// [12..14] rotMat right
	base[12] = rm.right.x; base[13] = rm.right.y; base[14] = rm.right.z;

	// [15..17] rotMat up
	base[15] = rm.up.x; base[16] = rm.up.y; base[17] = rm.up.z;

	// [18] isOnGround
	base[18] = cs.isOnGround ? 1.0f : 0.0f;

	// [19] boostAmount (fraction)
	base[19] = player.boostFraction;

	// [20..21] airTimeSinceJump (scaled), reserved
	base[20] = clamp(cs.airTimeSinceJump / 5.0f, 0.0f, 1.0f);
	base[21] = 0.0f;

	// [22] chargeLevel (always 0 for cars)
	base[22] = 0.0f;

	// [23..29] zero-padded (already zeroed)
}

// ============================================================================
// extractBallFeatures: ball entity feature extraction (30 dims)
// ============================================================================

void TemporalObsBuilder::extractBallFeatures(
	const BallState& ballState, int frameIdx, int slotIdx, float* out, int E, int kvDims) {

	float* base = out + frameIdx * E * kvDims + slotIdx * kvDims;
	memset(base, 0, kvDims * sizeof(float));

	const Vec pos = ballState.pos;
	const Vec vel = ballState.vel;
	const Vec angVel = ballState.angVel;

	// [0..2] pos (normalized)
	base[0] = pos.x / CommonValues::SIDE_WALL_X;
	base[1] = pos.y / CommonValues::BACK_WALL_Y;
	base[2] = pos.z / CommonValues::CEILING_Z;

	// [3..5] vel (normalized by ball max speed)
	float maxSpeed = CommonValues::BALL_MAX_SPEED * ballBouncinessMult;
	base[3] = vel.x / maxSpeed;
	base[4] = vel.y / maxSpeed;
	base[5] = vel.z / maxSpeed;

	// [6..8] angVel (normalized)
	base[6] = angVel.x / CommonValues::BALL_MAX_ANG_VEL;
	base[7] = angVel.y / CommonValues::BALL_MAX_ANG_VEL;
	base[8] = angVel.z / CommonValues::BALL_MAX_ANG_VEL;

	// [9..11] rotMat forward (ball is symmetric, use identity)
	base[9] = 1.0f; base[10] = 0.0f; base[11] = 0.0f;

	// [12..14] rotMat right
	base[12] = 0.0f; base[13] = 1.0f; base[14] = 0.0f;

	// [15..17] rotMat up
	base[15] = 0.0f; base[16] = 0.0f; base[17] = 1.0f;

	// [18] isOnGround (ball: always false)
	base[18] = 0.0f;

	// [19] boostAmount (ball: always 0)
	base[19] = 0.0f;

	// [20..21] angularVelocity magnitude (scaled), reserved
	float angVelMag = sqrtf(angVel.x * angVel.x + angVel.y * angVel.y + angVel.z * angVel.z);
	base[20] = clamp(angVelMag / CommonValues::BALL_MAX_ANG_VEL, 0.0f, 1.0f);
	base[21] = 0.0f;

	// [22] chargeLevel (for SPIKE_RUSH)
	if (activeMode == GameMode::SPIKE_RUSH) {
		base[22] = (float)ballState.dsInfo.chargeLevel;
	} else if (activeMode == GameMode::GRIDIRON) {
		// GRIDIRON: roof_offset_z as a feature
		base[22] = RLConst::Gridiron::ROOF_LOCAL_OFFSET.z / 200.0f;
	}

	// [23..29] zero-padded (already zeroed)
}

// ============================================================================
// extractPadFeatures: boost pad entity feature extraction (30 dims)
// ============================================================================

void TemporalObsBuilder::extractPadFeatures(
	const GameState& state, int padIdx, int frameIdx, int slotIdx, float* out, int E, int kvDims) {

	float* base = out + frameIdx * E * kvDims + slotIdx * kvDims;
	memset(base, 0, kvDims * sizeof(float));

	// Boost pad positions are from CommonValues::BOOST_LOCATIONS
	if (padIdx >= CommonValues::BOOST_LOCATIONS_AMOUNT) return;

	const Vec& pos = CommonValues::BOOST_LOCATIONS[padIdx];

	// [0..2] pos (normalized)
	base[0] = pos.x / CommonValues::SIDE_WALL_X;
	base[1] = pos.y / CommonValues::BACK_WALL_Y;
	base[2] = pos.z / CommonValues::CEILING_Z;

	// [3..5] vel (pad: always 0)
	base[3] = base[4] = base[5] = 0.0f;

	// [6..8] angVel (pad: always 0)
	base[6] = base[7] = base[8] = 0.0f;

	// [9..17] rotMat identity (pad is a static object)
	base[9] = 1.0f; base[12] = 0.0f; base[15] = 0.0f;
	base[10] = 0.0f; base[13] = 1.0f; base[16] = 0.0f;
	base[11] = 0.0f; base[14] = 0.0f; base[17] = 1.0f;

	// [18] isOnGround (pad: always true)
	base[18] = 1.0f;

	// [19] boostPadActive
	if (padIdx < (int)state.boostPads.size()) {
		base[19] = state.boostPads[padIdx] ? 1.0f : 0.0f;
	}

	// [20..21] cooldown (scaled)
	if (padIdx < (int)state.boostPadTimers.size()) {
		base[20] = clamp(state.boostPadTimers[padIdx] / 10.0f, 0.0f, 1.0f);
	}

	// [22..29] zero-padded (already zeroed)
}

// ============================================================================
// buildQuery: self features + direct ball block + tactical/mode/tile/team-size tail
// ============================================================================

FList TemporalObsBuilder::buildQuery(const PlayerData& player, const GameState& state, int queryDims) {
	FList q;
	q.reserve(queryDims);

	const auto& cs = player.carState;

	// === Self features (25 world-space) ===
	// [0..2] pos (normalized)
	q += clamp(cs.pos.x / CommonValues::SIDE_WALL_X, -1.0f, 1.0f);
	q += clamp(cs.pos.y / CommonValues::BACK_WALL_Y, -1.0f, 1.0f);
	q += clamp(cs.pos.z / CommonValues::CEILING_Z, -1.0f, 1.0f);

	// [3..5] vel (normalized)
	q += clamp(cs.vel.x / CommonValues::CAR_MAX_SPEED, -1.0f, 1.0f);
	q += clamp(cs.vel.y / CommonValues::CAR_MAX_SPEED, -1.0f, 1.0f);
	q += clamp(cs.vel.z / CommonValues::CAR_MAX_SPEED, -1.0f, 1.0f);

	// [6..8] angVel (normalized)
	q += clamp(cs.angVel.x / CommonValues::CAR_MAX_ANG_VEL, -1.0f, 1.0f);
	q += clamp(cs.angVel.y / CommonValues::CAR_MAX_ANG_VEL, -1.0f, 1.0f);
	q += clamp(cs.angVel.z / CommonValues::CAR_MAX_ANG_VEL, -1.0f, 1.0f);

	// [9..11] forward vector (rotMat)
	q += cs.rotMat.forward.x; q += cs.rotMat.forward.y; q += cs.rotMat.forward.z;

	// [12..14] right vector (rotMat)
	q += cs.rotMat.right.x; q += cs.rotMat.right.y; q += cs.rotMat.right.z;

	// [15..17] up vector (rotMat)
	q += cs.rotMat.up.x; q += cs.rotMat.up.y; q += cs.rotMat.up.z;

	// [18] isOnGround
	q += cs.isOnGround ? 1.0f : 0.0f;

	// [19] boostAmount (fraction)
	q += player.boostFraction;

	// [20] hasJump
	q += player.hasJump ? 1.0f : 0.0f;

	// [21] hasFlip
	q += player.hasFlip ? 1.0f : 0.0f;

	// [22] airTimeSinceJump (scaled)
	q += clamp(cs.airTimeSinceJump / 5.0f, 0.0f, 1.0f);

	// [23] isDemoed
	q += cs.isDemoed ? 1.0f : 0.0f;

	// [24] demoRespawnTimer (scaled)
	q += clamp(cs.demoRespawnTimer / 5.0f, 0.0f, 1.0f);

	// === Direct ball block (12 dims) ===
	// Ball position relative to player, rotated to player's local space
	Vec relPos = state.ballState.pos - cs.pos;
	auto rotRel = rotateRelative(relPos, cs.rotMat.forward, cs.rotMat.right, cs.rotMat.up);

	q += clamp(rotRel.x / CommonValues::SIDE_WALL_X, -1.0f, 1.0f);
	q += clamp(rotRel.y / CommonValues::BACK_WALL_Y, -1.0f, 1.0f);
	q += clamp(rotRel.z / CommonValues::CEILING_Z, -1.0f, 1.0f);

	// Ball distance
	float ballDist = relPos.Length();
	q += clamp(ballDist / 6000.0f, 0.0f, 1.0f);

	// Ball velocity (normalized)
	q += clamp(state.ballState.vel.x / CommonValues::BALL_MAX_SPEED, -1.0f, 1.0f);
	q += clamp(state.ballState.vel.y / CommonValues::BALL_MAX_SPEED, -1.0f, 1.0f);
	q += clamp(state.ballState.vel.z / CommonValues::BALL_MAX_SPEED, -1.0f, 1.0f);

	// Ball angular velocity magnitude (scaled)
	float ballAngVelMag = sqrtf(
		state.ballState.angVel.x * state.ballState.angVel.x +
		state.ballState.angVel.y * state.ballState.angVel.y +
		state.ballState.angVel.z * state.ballState.angVel.z
	);
	q += clamp(ballAngVelMag / CommonValues::BALL_MAX_ANG_VEL, 0.0f, 1.0f);

	// Ball isOnGround (ball: always false)
	q += 0.0f;

	// === Tactical/mode/mutator/tile/team-size tail ===
	switch (activeMode) {
	case GameMode::SPIKE_RUSH:
		// [activation_delay_remaining, attach_radius_normalized, release_cooldown_remaining, charge_level]
		q += clamp(state.attachInfo.engageTimer / RLConst::SpikeRush::ACTIVATION_DELAY, 0.0f, 1.0f);
		q += clamp(RLConst::SpikeRush::ATTACH_RADIUS / 200.0f, 0.0f, 1.0f);
		q += clamp(state.attachInfo.releaseCooldown / RLConst::SpikeRush::RELEASE_COOLDOWN, 0.0f, 1.0f);
		q += (float)state.ballState.dsInfo.chargeLevel;
		break;

	case GameMode::GRIDIRON:
		// [roof_offset_z_normalized, reacquire_cooldown_remaining, football_semi_major_normalized]
		q += clamp(RLConst::Gridiron::ROOF_LOCAL_OFFSET.z / 200.0f, 0.0f, 1.0f);
		q += clamp(state.attachInfo.releaseCooldown / RLConst::Gridiron::REACQUIRE_COOLDOWN, 0.0f, 1.0f);
		q += clamp(RLConst::Gridiron::FOOTBALL_SEMI_MAJOR / 100.0f, 0.0f, 1.0f);
		break;

	case GameMode::DROPSHOT:
		// [tiles_total, tiles_blue_damaged] - simplified (no tile state in GameState yet)
		q += (float)RLConst::Dropshot::NUM_TILE_ROWS * 13.0f; // approx total
		q += 0.0f; // would need tile state from arena
		break;

	default:
		break;
	}

	// Mutator tail (6 dims): gravity, boostConsumption, boostStrength, ballMaxSpeed, ballSize, ballBounciness
	q += clamp(gravityMult / 2.0f, 0.0f, 1.0f);
	q += clamp(boostConsumptionMult / 2.0f, 0.0f, 1.0f);
	q += clamp(boostStrengthMult / 2.0f, 0.0f, 1.0f);
	q += clamp(ballMaxSpeedMult / 2.0f, 0.0f, 1.0f);
	q += clamp(ballSizeMult / 2.0f, 0.0f, 1.0f);
	q += clamp(ballBouncinessMult / 2.0f, 0.0f, 1.0f);

	// Team-size tail (2 dims): nBlue/maxPlayers, nOrange/maxPlayers as fractions
	q += (float)nBlue / config.maxPlayers;
	q += (float)nOrange / config.maxPlayers;

	// Pad to queryDims if needed
	while ((int)q.size() < queryDims) q.push_back(0.0f);

	return q;
}

// ============================================================================
// buildKeyValue: temporal buffer flattened [F][E][kvFeatures] -> [F*E*kvFeatures]
// ============================================================================

FList TemporalObsBuilder::buildKeyValue(const GameState& state) {
	FList kv;
	int totalSlots = config.nFrames * config.maxBoostPads + (int)state.players.size() + 1; // ball + cars + pads
	kv.resize(totalSlots * config.kvFeatures, 0.0f);

	// Copy from temporal buffer
	for (int f = 0; f < config.nFrames; f++) {
		int entityOffset = 0;

		// Ball at slot 0
		if (f * totalSlots + 0 < (int)temporalBuf.data.size()) {
			memcpy(kv.data() + f * totalSlots * config.kvFeatures,
			       temporalBuf.data.data() + f * temporalBuf.E * config.kvFeatures,
			       config.kvFeatures * sizeof(float));
		}

		// Cars at slots 1..nPlayers
		for (int i = 0; i < (int)state.players.size(); i++) {
			int slot = 1 + i;
			if (f * totalSlots + slot < (int)temporalBuf.data.size()) {
				memcpy(kv.data() + f * totalSlots * config.kvFeatures + slot * config.kvFeatures,
				       temporalBuf.data.data() + f * temporalBuf.E * config.kvFeatures + slot * config.kvFeatures,
				       config.kvFeatures * sizeof(float));
			}
		}

		// Boost pads at remaining slots
		for (int p = 0; p < config.maxBoostPads; p++) {
			int slot = 1 + (int)state.players.size() + p;
			if (slot < totalSlots && f * totalSlots + slot < (int)temporalBuf.data.size()) {
				memcpy(kv.data() + f * totalSlots * config.kvFeatures + slot * config.kvFeatures,
				       temporalBuf.data.data() + f * temporalBuf.E * config.kvFeatures + slot * config.kvFeatures,
				       config.kvFeatures * sizeof(float));
			}
		}
	}

	return kv;
}

// ============================================================================
// buildMask: constant attention mask for zero-padded slots
// ============================================================================

FList TemporalObsBuilder::buildMask(const GameState& state) {
	FList m;
	int totalSlots = config.nFrames * config.maxBoostPads + (int)state.players.size() + 1; // ball + cars + pads

	// All slots are visible (mask = 1.0) for entities that exist
	// Zero-padded slots get mask = 0.0
	for (int f = 0; f < config.nFrames; f++) {
		for (int s = 0; s < totalSlots; s++) {
			m.push_back(1.0f); // default: visible
		}
	}

	return m;
}

} // namespace RLGSC
