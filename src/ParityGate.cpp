// ParityGate: cross-language obs-equivalence harness (C++ half).
//
// Reads hand-authored, RNG-free fixture files (one per scenario) describing a deterministic
// sequence of game states, drives the C++ TemporalObsBuilder over each sequence, and dumps the
// per-(frame,agent) flattened obs to <outDir>/<name>.cpp.txt (one obs per line, line k = f*nActive+i).
//
// The Python harness (scripts/parity/run_parity.py) parses the SAME fixtures with the real
// reference obs.py + profile_tracker.py and diffs elementwise against these dumps. The two builders
// must agree to <1e-5 elementwise; that is the gate.
//
// Fixture grammar (line-based; '#' comments, blank lines ignored):
//   MODE <int>                 mode code 0..6 (see TemporalObsBuilder ModeCodeFromGameMode)
//   TICK_SKIP <int>
//   N_ACTIVE_PADS <int>
//   BLUE_SCORE <int>
//   ORANGE_SCORE <int>
//   TICKS_LEFT <float>
//   IS_OVERTIME <0|1>
//   FINISHED <0|1>
//   MUT <g> <bc> <bs> <bms> <bsz> <bb>
//   HS <dir> <speed> <time>
//   CHARGE <int>
//   PAD_TIMERS <t0 ... tN>     (N_ACTIVE_PADS floats)
//   TILES <f0 ... f139>        (optional; 140 floats, dropshot only)
//   FRAME
//   BALL <px py pz vx vy vz ax ay az>
//   CAR <team> <px py pz> <vx vy vz> <ax ay az> <fwx fwy fwz> <upx upy upz> \
//       <boost> <demoed> <onground> <hasflip> <hasjump> <balltouch> <bumped>
//   ... (one CAR line per active agent, same count every FRAME)
//   FRAME
//   ...

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <filesystem>

#include "RLGymSim_CPP/Framework.h"
#include "RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h"

using namespace RLGSC;

struct CarRow {
	int team;
	float pos[3], vel[3], ang[3], fw[3], up[3];
	float boost, demoed, onground, hasflip, hasjump, balltouch, bumped;
};
struct Frame {
	float ball[9];
	std::vector<CarRow> cars;
};
struct Fixture {
	int mode = 0, tickSkip = 8, nActivePads = 0;
	int blueScore = 0, orangeScore = 0;
	float ticksLeft = 0;
	int isOvertime = 0, finished = 0;
	float mut[6] = {1, 1, 1, 1, 1, 1};
	float hs[3] = {0, 0, 0};
	int charge = 0;
	std::vector<float> padTimers;
	std::array<float, 140> tiles{};
	bool hasTiles = false;
	std::vector<Frame> frames;
};

static std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> out;
	std::istringstream iss(s);
	std::string tok;
	while (iss >> tok) out.push_back(tok);
	return out;
}

static Fixture parseFixture(const std::string& path) {
	std::ifstream in(path);
	if (!in) { std::cerr << "cannot open fixture " << path << std::endl; std::exit(2); }
	Fixture fx;
	std::string line;
	while (std::getline(in, line)) {
		// strip comments
		size_t h = line.find('#');
		if (h != std::string::npos) line = line.substr(0, h);
		auto t = split(line);
		if (t.empty()) continue;
		const std::string& k = t[0];
		auto num = [&](size_t i) { return std::stof(t[i]); };
		if (k == "MODE") fx.mode = std::stoi(t[1]);
		else if (k == "TICK_SKIP") fx.tickSkip = std::stoi(t[1]);
		else if (k == "N_ACTIVE_PADS") fx.nActivePads = std::stoi(t[1]);
		else if (k == "BLUE_SCORE") fx.blueScore = std::stoi(t[1]);
		else if (k == "ORANGE_SCORE") fx.orangeScore = std::stoi(t[1]);
		else if (k == "TICKS_LEFT") fx.ticksLeft = std::stof(t[1]);
		else if (k == "IS_OVERTIME") fx.isOvertime = std::stoi(t[1]);
		else if (k == "FINISHED") fx.finished = std::stoi(t[1]);
		else if (k == "MUT") { for (int i = 0; i < 6; i++) fx.mut[i] = num(1 + i); }
		else if (k == "HS") { for (int i = 0; i < 3; i++) fx.hs[i] = num(1 + i); }
		else if (k == "CHARGE") fx.charge = std::stoi(t[1]);
		else if (k == "PAD_TIMERS") { for (size_t i = 1; i < t.size(); i++) fx.padTimers.push_back(std::stof(t[i])); }
		else if (k == "TILES") { fx.hasTiles = true; for (int i = 0; i < 140; i++) fx.tiles[i] = std::stof(t[1 + i]); }
		else if (k == "FRAME") fx.frames.push_back(Frame{});
		else if (k == "BALL") { for (int i = 0; i < 9; i++) fx.frames.back().ball[i] = num(1 + i); }
		else if (k == "CAR") {
			CarRow c{};
			c.team = std::stoi(t[1]);
			for (int i = 0; i < 3; i++) c.pos[i] = num(2 + i);
			for (int i = 0; i < 3; i++) c.vel[i] = num(5 + i);
			for (int i = 0; i < 3; i++) c.ang[i] = num(8 + i);
			for (int i = 0; i < 3; i++) c.fw[i] = num(11 + i);
			for (int i = 0; i < 3; i++) c.up[i] = num(14 + i);
			c.boost = num(17); c.demoed = num(18); c.onground = num(19);
			c.hasflip = num(20); c.hasjump = num(21); c.balltouch = num(22); c.bumped = num(23);
			fx.frames.back().cars.push_back(c);
		}
	}
	return fx;
}

// Build a C++ GameState from a fixture frame, with cars in agent order (carId = i+1).
static GameState buildState(const Fixture& fx, const Frame& fr) {
	GameState s{};
	s.ballState.pos = Vec{fr.ball[0], fr.ball[1], fr.ball[2]};
	s.ballState.vel = Vec{fr.ball[3], fr.ball[4], fr.ball[5]};
	s.ballState.angVel = Vec{fr.ball[6], fr.ball[7], fr.ball[8]};
	s.ballState.hsInfo.yTargetDir = fx.hs[0];
	s.ballState.hsInfo.curTargetSpeed = fx.hs[1];
	s.ballState.hsInfo.timeSinceHit = fx.hs[2];
	s.ballState.dsInfo.chargeLevel = fx.charge;
	for (int p = 0; p < (int)fx.padTimers.size() && p < CommonValues::BOOST_LOCATIONS_AMOUNT; p++)
		s.boostPadTimers[p] = fx.padTimers[p];
	if (fx.hasTiles) s.tileStates = fx.tiles;
	s.nActivePads = fx.nActivePads;
	s.scoreLine[0] = fx.blueScore;
	s.scoreLine[1] = fx.orangeScore;
	s.ticksLeft = fx.ticksLeft;
	s.isOvertime = fx.isOvertime != 0;
	s.finished = fx.finished != 0;
	for (size_t i = 0; i < fr.cars.size(); i++) {
		const CarRow& c = fr.cars[i];
		PlayerData pd{};
		pd.carId = (uint32_t)(i + 1);
		pd.team = c.team == 1 ? Team::ORANGE : Team::BLUE;
		CarState cs{};
		cs.pos = Vec{c.pos[0], c.pos[1], c.pos[2]};
		cs.vel = Vec{c.vel[0], c.vel[1], c.vel[2]};
		cs.angVel = Vec{c.ang[0], c.ang[1], c.ang[2]};
		Vec fwd{c.fw[0], c.fw[1], c.fw[2]};
		Vec up{c.up[0], c.up[1], c.up[2]};
		cs.rotMat = RotMat(fwd, up.Cross(fwd), up);
		cs.boost = c.boost;
		cs.isDemoed = c.demoed != 0;
		cs.isOnGround = c.onground != 0;
		cs.hasFlipped = c.hasflip != 0;
		cs.hasJumped = c.hasjump != 0;
		cs.carContact.cooldownTimer = c.bumped != 0 ? 1.0f : 0.0f;
		pd.carState = cs;
		pd.ballTouchedStep = c.balltouch != 0;
		s.players.push_back(pd);
	}
	return s;
}

static GameMode modeFromCode(int code) {
	switch (code) {
		case 0: return GameMode::SOCCAR;
		case 1: return GameMode::HOOPS;
		case 2: return GameMode::HEATSEEKER;
		case 3: return GameMode::SNOWDAY;
		case 6: return GameMode::DROPSHOT;
		case 5: return GameMode::SPIKE_RUSH;
		case 4: return GameMode::GRIDIRON;
		default: return GameMode::SOCCAR;
	}
}

static void dumpObs(std::ofstream& out, const FList& o) {
	for (size_t i = 0; i < o.size(); i++) {
		if (i) out << ' ';
		char buf[32]; std::snprintf(buf, sizeof(buf), "%.9g", o[i]);
		out << buf;
	}
	out << '\n';
}

static void runFixture(const std::string& inPath, const std::string& outPath) {
	Fixture fx = parseFixture(inPath);
	GameMode gm = modeFromCode(fx.mode);

	TemporalObsBuilder b;
	b.activeMode = gm;
	b.tickSkip = fx.tickSkip;
	b.gravityMult = fx.mut[0];
	b.boostConsumptionMult = fx.mut[1];
	b.boostStrengthMult = fx.mut[2];
	b.ballMaxSpeedMult = fx.mut[3];
	b.ballSizeMult = fx.mut[4];
	b.ballBouncinessMult = fx.mut[5];

	GameState s0 = buildState(fx, fx.frames[0]);
	int nActive = (int)fx.frames[0].cars.size();
	b.Reset(s0);

	std::ofstream out(outPath);
	if (!out) { std::cerr << "cannot open out " << outPath << std::endl; std::exit(3); }
	Action act{};
	// frame 0: read the broadcast buffer directly (neutral profiles)
	for (int i = 0; i < nActive; i++) dumpObs(out, b.BuildOBS(s0.players[i], s0, act));
	// frames 1..N-1: PreStep (shift + updateProfiles + encode) then read
	for (size_t f = 1; f < fx.frames.size(); f++) {
		GameState sf = buildState(fx, fx.frames[f]);
		b.PreStep(sf);
		for (int i = 0; i < nActive; i++) dumpObs(out, b.BuildOBS(sf.players[i], sf, act));
	}
	out.close();
	std::cout << "  [dump] " << outPath << " (" << fx.frames.size() << " frames x " << nActive << " agents)" << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "usage: ParityGate <fixturesDir> <outDir>" << std::endl;
		return 1;
	}
	std::string inDir = argv[1];
	std::string outDir = argv[2];
	std::filesystem::create_directories(outDir);

	std::vector<std::string> files;
	for (auto& e : std::filesystem::directory_iterator(inDir))
		if (e.path().extension() == ".txt") files.push_back(e.path().string());
	std::sort(files.begin(), files.end());

	std::cout << "=== RLGymSim_CPP ParityGate: " << files.size() << " fixture(s) ===" << std::endl;
	for (auto& f : files) {
		std::string name = std::filesystem::path(f).stem().string();
		runFixture(f, outDir + "/" + name + ".cpp.txt");
	}
	std::cout << "=== ParityGate done ===" << std::endl;
	return 0;
}
