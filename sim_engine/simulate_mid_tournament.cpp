// Compile: g++ -std=c++20 simulate_mid_tournament.cpp -o simulate_mid_tournament
// Run: ./simulate_mid_tournament

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <algorithm>
#include <random>
#include <iomanip>
#include <cctype>
#include <tuple>

using namespace std;

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------
struct MatchPrediction {
    string group;
    string homeTeam;
    string awayTeam;
    double homeWinProb = 0.0;
    double drawProb = 0.0;
    double awayWinProb = 0.0;
    double homeXg;
    double awayXg;
    int isPlayed = 0; // 1 = Fixed real result, 0 = Simulate with Poisson
    string likelyScore; // added for JSON compactness
};

struct TeamStats {
    string name;
    string group;
    double expectedPoints = 0.0;
    int finishPositions[4] = {0, 0, 0, 0};
};

struct ScoreProb {
    string score;
    double prob;
};

struct SimTeam {
    string name;
    int points = 0;
    int goalDiff = 0;
    int goalsScored = 0;
};

// Global storage for JSON output
vector<MatchPrediction> allMatches;                // all 72 matches
map<string, vector<TeamStats>> allStandings;       // group -> standings

// ---------------------------------------------------------
// PARSING HELPERS
// ---------------------------------------------------------
double safe_stod(const string& s) {
    if (s.empty()) return 0.0;
    string trimmed = s;
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start == string::npos) return 0.0;
    trimmed = trimmed.substr(start);
    size_t end = trimmed.find_last_not_of(" \t\n\r");
    if (end != string::npos) trimmed = trimmed.substr(0, end + 1);
    for (char& c : trimmed) c = tolower(c);
    if (trimmed == "nan" || trimmed == "-nan" || trimmed == "inf" || trimmed == "-inf")
        return 0.0;
    try { return stod(trimmed); } catch (...) { return 0.0; }
}

int safe_stoi(const string& s) {
    try { return stoi(s); } catch (...) { return 0; }
}

// ---------------------------------------------------------
// MATHEMATICAL HELPERS
// ---------------------------------------------------------
double factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

double poisson(int k, double lambda) {
    if (lambda <= 0) lambda = 0.1;
    return (pow(lambda, k) * exp(-lambda)) / factorial(k);
}

void calculateMatchProbabilities(MatchPrediction& match) {
    if (match.isPlayed == 1) {
        if (match.homeXg > match.awayXg)      { match.homeWinProb = 1.0; match.drawProb = 0.0; match.awayWinProb = 0.0; }
        else if (match.homeXg < match.awayXg) { match.homeWinProb = 0.0; match.drawProb = 0.0; match.awayWinProb = 1.0; }
        else                                  { match.homeWinProb = 0.0; match.drawProb = 1.0; match.awayWinProb = 0.0; }
        return;
    }

    match.homeWinProb = 0.0; match.drawProb = 0.0; match.awayWinProb = 0.0;
    const double rho = -0.12;

    for (int h = 0; h <= 15; ++h) {
        for (int a = 0; a <= 15; ++a) {
            double p_home = poisson(h, match.homeXg);
            double p_away = poisson(a, match.awayXg);
            double prob = p_home * p_away;

            // Dixon-Coles correction
            if (h == 0 && a == 0)      prob *= (1.0 - match.homeXg * match.awayXg * rho);
            else if (h == 1 && a == 0) prob *= (1.0 + match.homeXg * rho);
            else if (h == 0 && a == 1) prob *= (1.0 + match.awayXg * rho);
            else if (h == 1 && a == 1) prob *= (1.0 - rho);

            if (h > a)       match.homeWinProb += prob;
            else if (h < a)  match.awayWinProb += prob;
            else             match.drawProb += prob;
        }
    }

    double totalProb = match.homeWinProb + match.drawProb + match.awayWinProb;
    match.homeWinProb /= totalProb;
    match.drawProb   /= totalProb;
    match.awayWinProb /= totalProb;
}

vector<ScoreProb> getTopScorelines(const MatchPrediction& match, int topN = 3) {
    if (match.isPlayed == 1) {
        string exactScore = to_string((int)match.homeXg) + "-" + to_string((int)match.awayXg);
        return { {exactScore, 1.0} };
    }
    vector<tuple<double, int, int>> scoreProbs;
    for (int h = 0; h <= 10; ++h) {
        for (int a = 0; a <= 10; ++a) {
            double prob = poisson(h, match.homeXg) * poisson(a, match.awayXg);
            scoreProbs.emplace_back(prob, h, a);
        }
    }
    sort(scoreProbs.begin(), scoreProbs.end(), [](const auto& a, const auto& b) {
        return get<0>(a) > get<0>(b);
    });
    vector<ScoreProb> result;
    for (int i = 0; i < min(topN, (int)scoreProbs.size()); ++i) {
        string scoreStr = to_string(get<1>(scoreProbs[i])) + "-" + to_string(get<2>(scoreProbs[i]));
        result.push_back({scoreStr, get<0>(scoreProbs[i])});
    }
    return result;
}

// ---------------------------------------------------------
// MID-TOURNAMENT HYBRID SIMULATION ENGINE (unchanged logic)
// ---------------------------------------------------------
vector<TeamStats> simulateGroup(const string& groupName, vector<MatchPrediction>& matches) {
    map<string, TeamStats> groupTeams;
    for (auto& m : matches) {
        calculateMatchProbabilities(m);
        // Store the most likely scoreline for JSON output
        auto top = getTopScorelines(m, 1);
        if (!top.empty()) m.likelyScore = top[0].score;

        groupTeams[m.homeTeam].name = m.homeTeam;
        groupTeams[m.homeTeam].group = groupName;
        groupTeams[m.homeTeam].expectedPoints += (3.0 * m.homeWinProb) + (1.0 * m.drawProb);
        groupTeams[m.awayTeam].name = m.awayTeam;
        groupTeams[m.awayTeam].group = groupName;
        groupTeams[m.awayTeam].expectedPoints += (3.0 * m.awayWinProb) + (1.0 * m.drawProb);
    }

    const int SIMULATIONS = 1000000;
    random_device rd;
    mt19937 gen(rd());

    vector<poisson_distribution<int>> homeDists, awayDists;
    for (const auto& m : matches) {
        homeDists.push_back(poisson_distribution<int>(max(0.01, m.homeXg)));
        awayDists.push_back(poisson_distribution<int>(max(0.01, m.awayXg)));
    }

    for (int i = 0; i < SIMULATIONS; ++i) {
        map<string, SimTeam> simTable;
        for (const auto& pair : groupTeams) {
            simTable[pair.first] = {pair.first, 0, 0, 0};
        }

        for (size_t mIdx = 0; mIdx < matches.size(); ++mIdx) {
            const auto& m = matches[mIdx];
            int hGoals = 0, aGoals = 0;

            if (m.isPlayed == 1) {
                hGoals = (int)m.homeXg;
                aGoals = (int)m.awayXg;
            } else {
                hGoals = homeDists[mIdx](gen);
                aGoals = awayDists[mIdx](gen);
            }

            simTable[m.homeTeam].goalsScored += hGoals;
            simTable[m.homeTeam].goalDiff += (hGoals - aGoals);
            simTable[m.awayTeam].goalsScored += aGoals;
            simTable[m.awayTeam].goalDiff += (aGoals - hGoals);

            if (hGoals > aGoals)       simTable[m.homeTeam].points += 3;
            else if (hGoals < aGoals)  simTable[m.awayTeam].points += 3;
            else {
                simTable[m.homeTeam].points += 1;
                simTable[m.awayTeam].points += 1;
            }
        }

        vector<SimTeam> iterationTable;
        for (const auto& pair : simTable) iterationTable.push_back(pair.second);

        sort(iterationTable.begin(), iterationTable.end(), [](const SimTeam& a, const SimTeam& b) {
            if (a.points != b.points) return a.points > b.points;
            if (a.goalDiff != b.goalDiff) return a.goalDiff > b.goalDiff;
            if (a.goalsScored != b.goalsScored) return a.goalsScored > b.goalsScored;
            return a.name < b.name;
        });

        for (int pos = 0; pos < 4 && pos < (int)iterationTable.size(); ++pos) {
            groupTeams[iterationTable[pos].name].finishPositions[pos]++;
        }
    }

    vector<TeamStats> displayTable;
    for (const auto& pair : groupTeams) displayTable.push_back(pair.second);
    sort(displayTable.begin(), displayTable.end(), [](const TeamStats& a, const TeamStats& b) {
        if (a.expectedPoints != b.expectedPoints) return a.expectedPoints > b.expectedPoints;
        return (a.finishPositions[0] > b.finishPositions[0]);
    });

    // Console output (unchanged)
    cout << "\n========================================\n GROUP " << groupName << " MID-TOURNAMENT STATUS\n========================================\n";
    cout << left << setw(22) << "Team" << setw(12) << "Exp. Pts" << setw(10) << "1st %" << setw(10) << "2nd %" << setw(10) << "3rd %" << setw(10) << "4th %" << endl;
    cout << string(74, '-') << endl;
    for (const auto& t : displayTable) {
        cout << left << setw(22) << t.name << setw(12) << fixed << setprecision(1) << t.expectedPoints;
        for (int pos = 0; pos < 4; ++pos)
            cout << setw(10) << (t.finishPositions[pos] / (double)SIMULATIONS) * 100.0;
        cout << endl;
    }

    return displayTable;
}

// ---------------------------------------------------------
// JSON WRITING (compact, knockout‑style)
// ---------------------------------------------------------
void writeCompactJSON(const string& filepath) {
    ofstream out(filepath);
    if (!out.is_open()) { cerr << "Error writing JSON\n"; return; }

    out << "{\n  \"matches\": [\n";
    for (size_t i = 0; i < allMatches.size(); ++i) {
        const auto& m = allMatches[i];
        out << "    {"
            << "\"group\":\"" << m.group << "\","
            << "\"home\":\"" << m.homeTeam << "\","
            << "\"away\":\"" << m.awayTeam << "\","
            << "\"home_xg\":" << m.homeXg << ","
            << "\"away_xg\":" << m.awayXg << ","
            << "\"played\":" << (m.isPlayed ? "true" : "false") << ","
            << "\"p_home\":" << m.homeWinProb << ","
            << "\"p_draw\":" << m.drawProb << ","
            << "\"p_away\":" << m.awayWinProb << ","
            << "\"score\":\"" << m.likelyScore << "\"}";
        if (i < allMatches.size() - 1) out << ",";
        out << "\n";
    }
    out << "  ],\n  \"standings\": {\n";
    size_t gIdx = 0;
    for (const auto& [grp, teams] : allStandings) {
        out << "    \"" << grp << "\": [";
        for (size_t j = 0; j < teams.size(); ++j) {
            const auto& t = teams[j];
            out << "{\"team\":\"" << t.name
                << "\",\"pts\":" << fixed << setprecision(2) << t.expectedPoints
                << ",\"pos\":["
                << (t.finishPositions[0] / 1000000.0) << ","
                << (t.finishPositions[1] / 1000000.0) << ","
                << (t.finishPositions[2] / 1000000.0) << ","
                << (t.finishPositions[3] / 1000000.0) << "]}";
            if (j < teams.size() - 1) out << ",";
        }
        out << "]";
        if (++gIdx < allStandings.size()) out << ",";
        out << "\n";
    }
    out << "  }\n}\n";
    out.close();
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    string filepath = "../data/mid_tournament_predictions.csv";
    string jsonOutPath = "../data/simulation_matrices_mid_tournament.json";

    cout << "========================================================\n";
    cout << "FIFA WORLD WC 2026 HYBRID SIMULATION ENGINE\n";
    cout << "========================================================\n";

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "\nERROR: Failed to open hybrid file path! " << filepath << endl;
        return 1;
    }

    map<string, vector<MatchPrediction>> tournamentGroups;
    string line;
    getline(file, line); // skip header

    int rowCount = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        MatchPrediction match;

        getline(ss, match.group, ',');
        getline(ss, match.homeTeam, ',');
        getline(ss, match.awayTeam, ',');
        getline(ss, cell, ','); match.homeXg = safe_stod(cell);
        getline(ss, cell, ','); match.awayXg = safe_stod(cell);
        getline(ss, cell, ','); match.isPlayed = safe_stoi(cell);

        tournamentGroups[match.group].push_back(match);
        rowCount++;
    }
    file.close();
    cout << "Successfully ingested " << rowCount << " matches into memory architecture.\n";

    // Simulate groups and store results for JSON
    for (auto& groupPair : tournamentGroups) {
        string grp = groupPair.first;
        auto& matches = groupPair.second;
        auto standings = simulateGroup(grp, matches);
        // Copy matches (now containing probabilities and likelyScore) to global list
        for (auto& m : matches) allMatches.push_back(m);
        allStandings[grp] = standings;
    }

    // Write compact JSON
    writeCompactJSON(jsonOutPath);

    return 0;
}