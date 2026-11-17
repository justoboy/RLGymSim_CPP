// FixtureGate: pure obs-builder test harness.
//
// Unlike ScenarioRunner (which drives the RNG'd Gym rollout), this harness feeds HAND-AUTHORED,
// RNG-free GameStates straight into TemporalObsBuilder and (a) dumps canonical obs to fixtures/
// and (b) asserts self-consistency invariants that need no second implementation to compare against:
//   - determinism: identical input -> byte-identical obs
//   - width: profile-on obs is exactly 10778 floats
//   - mask: inactive car slots / pads are masked (mask=1)
//   - soccar gating: heatseeker + tile tail fields are zero in soccar
//   - spin: ball angVel reaches exactly the ball ANG_VEL columns (not zeroed)
//   - score/clock: goal_diff/time_left read from real GameState fields
//   - inversion symmetry: a pose-mirrored solo state yields obs_blue == obs_orange EXACTLY
//
// The golden dumps in fixtures/ are the canonical obs a reference implementation must reproduce.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>

#include "RLGymSim_CPP/Framework.h"
#include "RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h"

using namespace RLGSC;

static int g_failures = 0;

static void check(bool cond, const std::string& what) {
	if (!cond) {
		std::cerr << "  [FAIL] " << what << std::endl;
		g_failures++;
	} else {
		std::cout << "  [ ok ] " << what << std::endl;
	}
}

// Build a car PlayerData with an explicit pose. forward/up are the car's local axes.
static PlayerData makeCar(uint32_t id, Team team, Vec pos, Vec forward, Vec up, float boost) {
	PlayerData pd{};
	pd.carId = id;
	pd.team = team;
	CarState cs{};
	cs.pos = pos;
	cs.rotMat = RotMat(forward, up.Cross(forward), up);
	cs.vel = Vec{0, 0, 0};
	cs.angVel = Vec{0, 0, 0};
	cs.boost = boost;
	cs.isOnGround = true;
	pd.carState = cs;
	pd.hasJump = true;
	pd.hasFlip = true;
	return pd;
}

// A canonical 1v1 soccar state with the ball on the x=0 plane (so it is inversion-invariant).
static GameState makeSymmetric1v1() {
	GameState s{};
	s.ballState.pos = Vec{0, 0, CommonValues::BALL_RADIUS};
	s.ballState.vel = Vec{0, 0, 0};
	s.ballState.angVel = Vec{0, 0, 0};
	s.players.push_back(makeCar(1, Team::BLUE,   Vec{0, -2000, 18}, Vec{0,  1, 0}, Vec{0, 0, 1}, 34));
	s.players.push_back(makeCar(2, Team::ORANGE, Vec{0,  2000, 18}, Vec{0, -1, 0}, Vec{0, 0, 1}, 34));
	s.nActivePads = 0; // no pads active -> all pad slots masked
	s.scoreLine[0] = 0;
	s.scoreLine[1] = 0;
	s.ticksLeft = 0;
	s.isOvertime = false;
	return s;
}

// A single-car state on the x=0 plane. Used for the inversion-symmetry check: with exactly one
// car, selfSlot==1 for BOTH teams, so a pose-mirrored pair (blue at -y facing +y, orange at +y
// facing -y) must yield elementwise-identical obs.
static GameState makeSymmetricSolo(Team team, Vec pos, Vec forward) {
	GameState s{};
	s.ballState.pos = Vec{0, 0, CommonValues::BALL_RADIUS};
	s.ballState.vel = Vec{0, 0, 0};
	s.ballState.angVel = Vec{0, 0, 0};
	s.players.push_back(makeCar(1, team, pos, forward, Vec{0, 0, 1}, 34));
	s.nActivePads = 0;
	s.scoreLine[0] = 0;
	s.scoreLine[1] = 0;
	s.ticksLeft = 0;
	s.isOvertime = false;
	return s;
}

static TemporalObsBuilder makeBuilder(GameMode mode) {
	TemporalObsBuilder b;
	b.activeMode = mode;
	b.tickSkip = 8;
	return b;
}

// ---- Invariant 1: determinism ----
static void testDeterminism() {
	std::cout << "\n[determinism]" << std::endl;
	GameState s = makeSymmetric1v1();
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(s);
	Action a{};
	FList o1 = b.BuildOBS(s.players[0], s, a);
	FList o2 = b.BuildOBS(s.players[0], s, a);
	check(o1.size() == o2.size(), "determinism: same size");
	bool identical = true;
	for (size_t i = 0; i < o1.size(); i++) if (o1[i] != o2[i]) { identical = false; break; }
	check(identical, "determinism: byte-identical obs on identical input");
}

// ---- Invariant 2: width ----
static void testWidth() {
	std::cout << "\n[width]" << std::endl;
	GameState s = makeSymmetric1v1();
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(s);
	FList o = b.BuildOBS(s.players[0], s, Action{});
	check(o.size() == 10778, "width: profile-on obs == 10778");
}

// ---- Invariant 3: mask (inactive slots masked) ----
static void testMask() {
	std::cout << "\n[mask]" << std::endl;
	GameState s = makeSymmetric1v1(); // 2 active cars, 0 pads
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(s);
	FList o = b.BuildOBS(s.players[0], s, Action{});
	const int Q = TemporalObsBuilder::QUERY_FEATURES;
	const int KV = TemporalObsBuilder::KV_FEATURES;
	const int NBC = TemporalObsBuilder::N_BC;
	const int F = TemporalObsBuilder::N_FRAMES;
	const int NTOK = TemporalObsBuilder::N_TOKENS;
	// mask block starts at Q + N_TOKENS*KV
	size_t maskStart = (size_t)Q + (size_t)NTOK * KV;
	// car slots 2..8 (entities 2..NBC-1) are inactive -> masked at all frames
	bool inactiveCarsMasked = true;
	for (int e = 1 + 2; e < NBC; e++)
		for (int f = 0; f < F; f++)
			if (o[maskStart + (size_t)f * NBC + e] != 1.0f) inactiveCarsMasked = false;
	check(inactiveCarsMasked, "mask: inactive car slots -> mask=1");
	// active car slot (self, e=1) must NOT be masked
	check(o[maskStart + 1] == 0.0f, "mask: active car slot -> mask=0");
	// all pads inactive (nActivePads=0) -> every pad token masked
	bool padsMasked = true;
	for (int p = 0; p < ObsLayout::MAX_BOOST_PADS; p++)
		if (o[maskStart + (size_t)F * NBC + p] != 1.0f) padsMasked = false;
	check(padsMasked, "mask: inactive pad slots -> mask=1");
}

// ---- Invariant 4: soccar gating (heatseeker + tile tail zeroed) ----
static void testSoccarGating() {
	std::cout << "\n[soccar-gating]" << std::endl;
	GameState s = makeSymmetric1v1();
	// plant nonzero heatseeker + tile data; soccar must ignore them.
	s.ballState.hsInfo.yTargetDir = 1.0f;
	s.ballState.hsInfo.curTargetSpeed = 999.0f;
	for (auto& t : s.tileStates) t = 2.0f;
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(s);
	FList o = b.BuildOBS(s.players[0], s, Action{});
	const int Q = TemporalObsBuilder::QUERY_FEATURES;
	// query tail: heatseeker at q[56..58], charge at q[65], tile block q[66..205]
	bool hsZero = true; for (int c = 56; c <= 58; c++) if (o[c] != 0.0f) hsZero = false;
	check(hsZero, "gating: heatseeker tail fields zero in soccar");
	check(o[65] == 0.0f, "gating: dropshot charge tier zero in soccar");
	bool tilesZero = true; for (int c = 66; c <= 205; c++) if (o[c] != 0.0f) tilesZero = false;
	check(tilesZero, "gating: tile block zero in soccar");
}

// ---- Invariant 5: real ball spin reaches exactly the ball token's spin-derived columns ----
static void testSpinWired() {
	std::cout << "\n[spin-wired]" << std::endl;
	GameState a = makeSymmetric1v1();
	GameState b = a;
	a.ballState.angVel = Vec{0, 0, 0};
	b.ballState.angVel = Vec{5.5f, 0, 0}; // normalized -> (1,0,0)
	TemporalObsBuilder ba = makeBuilder(GameMode::SOCCAR); ba.Reset(a);
	TemporalObsBuilder bb = makeBuilder(GameMode::SOCCAR); bb.Reset(b);
	FList oa = ba.BuildOBS(a.players[0], a, Action{});
	FList ob = bb.BuildOBS(b.players[0], b, Action{});
	const int Q = TemporalObsBuilder::QUERY_FEATURES;
	const int KV = TemporalObsBuilder::KV_FEATURES;
	const int NBC = TemporalObsBuilder::N_BC;
	const int F = TemporalObsBuilder::N_FRAMES;
	// A ball spin change must reach the obs (spin is NOT zeroed), and ONLY the ball token's
	// spin-derived columns (entity ANG_VEL cols 17-19 + relative angVel cols 43,48,53,58,63,68).
	std::vector<size_t> diff;
	for (size_t i = 0; i < oa.size(); i++) if (oa[i] != ob[i]) diff.push_back(i);
	check(!diff.empty(), "spin: changing ball angVel changes obs (spin wired, not zeroed)");
	auto isBallSpinCol = [&](int col) {
		return col == 17 || col == 18 || col == 19 ||
		       col == 43 || col == 48 || col == 53 || col == 58 || col == 63 || col == 68;
	};
	bool localized = true;
	for (size_t idx : diff) {
		size_t rel = (size_t)idx - Q;
		size_t token = rel / KV;
		int col = (int)(rel % KV);
		int f = (int)(token / NBC);
		if ((int)token != f * NBC || !isBallSpinCol(col) || f < 0 || f >= F) { localized = false; break; }
	}
	check(localized, "spin: diffs localized to the ball token's spin-derived columns");
}

// ---- Invariant 6: score/clock read from real fields ----
static void testScoreClock() {
	std::cout << "\n[score-clock]" << std::endl;
	GameState s = makeSymmetric1v1();
	s.players[0].team = Team::BLUE;
	s.scoreLine[0] = 3; s.scoreLine[1] = 1; // blue leads by 2
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(s);
	FList o = b.BuildOBS(s.players[0], s, Action{});
	// tactical goal_diff at q[51] for a blue agent = clamp(3-1,-5,5)/5 = 0.4
	check(std::fabs(o[51] - 0.4f) < 1e-5f, "score-clock: blue goal_diff from real scoreLine");
}

// ---- Invariant 7 (crown jewel): inversion symmetry ----
static void testInversionSymmetry() {
	std::cout << "\n[inversion-symmetry]" << std::endl;
	// single car per team on the x=0 plane -> selfSlot==1 for both. A pose-mirrored pair must
	// produce elementwise-identical obs (the team-inversion + IS_MATE/IS_OPP swap + self-relative
	// math must be mutually consistent).
	GameState sb = makeSymmetricSolo(Team::BLUE,   Vec{0, -2000, 18}, Vec{0,  1, 0});
	GameState so = makeSymmetricSolo(Team::ORANGE, Vec{0,  2000, 18}, Vec{0, -1, 0});
	TemporalObsBuilder b = makeBuilder(GameMode::SOCCAR);
	b.Reset(sb);
	FList blue = b.BuildOBS(sb.players[0], sb, Action{});
	TemporalObsBuilder o = makeBuilder(GameMode::SOCCAR);
	o.Reset(so);
	FList orange = o.BuildOBS(so.players[0], so, Action{});
	check(blue.size() == orange.size(), "symmetry: blue/orange obs same size");
	bool identical = true;
	for (size_t i = 0; i < blue.size(); i++) if (blue[i] != orange[i]) { identical = false; break; }
	check(identical, "symmetry: obs_blue == obs_orange on a pose-mirrored solo state (exact)");
}

// ---- Golden dump: write canonical obs for each mode to fixtures/ ----
static void dumpObs(const std::string& name, GameMode mode, const GameState& s, const PlayerData& pd) {
	TemporalObsBuilder b = makeBuilder(mode);
	b.Reset(s);
	FList o = b.BuildOBS(pd, s, Action{});
	std::string path = "build/fixtures/" + name + ".obs.txt";
	std::ofstream f(path);
	if (!f) { std::cerr << "  [warn] could not open " << path << " for writing" << std::endl; return; }
	for (size_t i = 0; i < o.size(); i++) {
		if (i) f << ' ';
		char buf[32]; std::snprintf(buf, sizeof(buf), "%.9g", o[i]);
		f << buf;
	}
	f << std::endl;
	std::cout << "  [dump] " << path << " (" << o.size() << " floats)" << std::endl;
}

int main() {
	std::cout << "=== RLGymSim_CPP Fixture Gate (pure obs-builder invariants) ===" << std::endl;

	testDeterminism();
	testWidth();
	testMask();
	testSoccarGating();
	testSpinWired();
	testScoreClock();
	testInversionSymmetry();

	// Golden dumps (the canonical obs a reference implementation must reproduce elementwise).
	std::cout << "\n[golden dumps]" << std::endl;
	GameState soccar = makeSymmetric1v1();
	soccar.scoreLine[0] = 2; soccar.scoreLine[1] = 1; soccar.ticksLeft = 1200;
	dumpObs("soccar_1v1", GameMode::SOCCAR, soccar, soccar.players[0]);

	GameState spike = makeSymmetric1v1();
	spike.ballState.attachInfo.active = true;
	spike.ballState.attachInfo.attachedCarId = 1;
	spike.ballState.attachInfo.lastCarrierId = 1;
	dumpObs("spike_rush_1v1", GameMode::SPIKE_RUSH, spike, spike.players[0]);

	GameState grid = makeSymmetric1v1();
	grid.ballState.attachInfo.active = true;
	dumpObs("gridiron_1v1", GameMode::GRIDIRON, grid, grid.players[0]);

	GameState heat = makeSymmetric1v1();
	heat.ballState.hsInfo.yTargetDir = 1.0f;
	heat.ballState.hsInfo.curTargetSpeed = 1500.0f;
	heat.ballState.hsInfo.timeSinceHit = 0.5f;
	dumpObs("heatseeker_1v1", GameMode::HEATSEEKER, heat, heat.players[0]);

	GameState drop = makeSymmetric1v1();
	drop.ballState.dsInfo.chargeLevel = 2;
	for (size_t t = 0; t < drop.tileStates.size(); t++) drop.tileStates[t] = (float)(t % 3);
	dumpObs("dropshot_1v1", GameMode::DROPSHOT, drop, drop.players[0]);

	if (g_failures == 0) {
		std::cout << "\n=== All fixture invariants passed ===" << std::endl;
		return 0;
	}
	std::cerr << "\n=== " << g_failures << " fixture invariant(s) FAILED ===" << std::endl;
	return 1;
}
