// Compile: g++ -std=c++20 main.cpp -o main
// Run: ./main

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

// ---------------------------------------------------------
// SAFE STRING TO DOUBLE (handles "nan", empty, etc.)
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
    try {
        return stod(trimmed);
    } catch (...) {
        return 0.0;
    }
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

void calculatePoissonMatchProbabilities(MatchPrediction& match) {
    match.homeWinProb = 0.0;
    match.drawProb = 0.0;
    match.awayWinProb = 0.0;
    for (int h = 0; h <= 15; ++h) {
        for (int a = 0; a <= 15; ++a) {
            double prob = poisson(h, match.homeXg) * poisson(a, match.awayXg);
            if (h > a) match.homeWinProb += prob;
            else if (h < a) match.awayWinProb += prob;
            else match.drawProb += prob;
        }
    }
}

vector<ScoreProb> getTopScorelines(const MatchPrediction& match, int topN = 3) {
    vector<tuple<double, int, int>> scoreProbs;
    for (int h = 0; h <= 10; ++h) {
        for (int a = 0; a <= 10; ++a) {
            double prob = poisson(h, match.homeXg) * poisson(a, match.awayXg);
            scoreProbs.emplace_back(prob, h, a);
        }
    }
    sort(scoreProbs.begin(), scoreProbs.end(),
         [](const auto& a, const auto& b) { return get<0>(a) > get<0>(b); });
    vector<ScoreProb> result;
    for (int i = 0; i < min(topN, (int)scoreProbs.size()); ++i) {
        string scoreStr = to_string(get<1>(scoreProbs[i])) + "-" + to_string(get<2>(scoreProbs[i]));
        result.push_back({scoreStr, get<0>(scoreProbs[i])});
    }
    return result;
}

double probTotalGoalsAtLeast3(const MatchPrediction& match) {
    double prob = 0.0;
    for (int h = 0; h <= 15; ++h) {
        for (int a = 0; a <= 15; ++a) {
            if (h + a >= 3) {
                prob += poisson(h, match.homeXg) * poisson(a, match.awayXg);
            }
        }
    }
    return prob;
}

// ---------------------------------------------------------
// SIMULATION ENGINE
// ---------------------------------------------------------
vector<TeamStats> simulateGroup(const string& groupName, vector<MatchPrediction>& matches, stringstream& jsonBuffer) {
    map<string, TeamStats> groupTeams;
    for (auto& m : matches) {
        calculatePoissonMatchProbabilities(m);
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
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < SIMULATIONS; ++i) {
        map<string, int> simPoints;
        for (const auto& pair : groupTeams) simPoints[pair.first] = 0;
        for (const auto& m : matches) {
            double r = dis(gen);
            if (r < m.homeWinProb) simPoints[m.homeTeam] += 3;
            else if (r < m.homeWinProb + m.drawProb) {
                simPoints[m.homeTeam] += 1;
                simPoints[m.awayTeam] += 1;
            } else simPoints[m.awayTeam] += 3;
        }
        vector<pair<string, int>> table(simPoints.begin(), simPoints.end());
        sort(table.begin(), table.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        for (int pos = 0; pos < 4 && pos < (int)table.size(); ++pos)
            groupTeams[table[pos].first].finishPositions[pos]++;
    }

    vector<TeamStats> displayTable;
    for (const auto& pair : groupTeams) displayTable.push_back(pair.second);
    sort(displayTable.begin(), displayTable.end(), [](const TeamStats& a, const TeamStats& b) {
        if (a.expectedPoints != b.expectedPoints) return a.expectedPoints > b.expectedPoints;
        double aFirst = a.finishPositions[0] / (double)SIMULATIONS;
        double bFirst = b.finishPositions[0] / (double)SIMULATIONS;
        return aFirst > bFirst;
    });

    // Terminal output
    cout << "\n========================================\n GROUP " << groupName << " PREDICTIONS\n========================================\n";
    cout << "--- Most Likely Exact Scorelines (Top 3) ---\n";
    for (const auto& m : matches) {
        auto topScores = getTopScorelines(m, 3);
        double probHigh = probTotalGoalsAtLeast3(m);
        cout << "   " << m.homeTeam << " vs " << m.awayTeam << "\n";
        for (size_t i = 0; i < topScores.size(); ++i) {
            cout << "      " << i+1 << ". " << topScores[i].score
                 << " (" << fixed << setprecision(2) << topScores[i].prob * 100.0 << "%)\n";
        }
        cout << "      Prob of 3+ total goals: " << fixed << setprecision(1) << probHigh * 100.0 << "%\n";
        cout << "      W/D/L: " << fixed << setprecision(1) << m.homeWinProb * 100.0
             << "% / " << m.drawProb * 100.0 << "% / " << m.awayWinProb * 100.0 << "%\n\n";
    }

    cout << "--- Group Table Probabilities (1,000,000 Simulations) ---\n";
    cout << left << setw(22) << "Team" << setw(12) << "Exp. Pts" << setw(10) << "1st %"
         << setw(10) << "2nd %" << setw(10) << "3rd %" << setw(10) << "4th %" << endl;
    cout << string(74, '-') << endl;
    for (const auto& t : displayTable) {
        cout << left << setw(22) << t.name << setw(12) << fixed << setprecision(1) << t.expectedPoints;
        for (int pos = 0; pos < 4; ++pos)
            cout << setw(10) << (t.finishPositions[pos] / (double)SIMULATIONS) * 100.0;
        cout << endl;
    }

    // JSON output
    jsonBuffer << "    \"" << groupName << "\": {\n";
    jsonBuffer << "      \"matches\": [\n";
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& m = matches[i];
        auto topScores = getTopScorelines(m, 3);
        jsonBuffer << "        {\n"
                   << "          \"home\": \"" << m.homeTeam << "\",\n"
                   << "          \"away\": \"" << m.awayTeam << "\",\n"
                   << "          \"home_win_prob\": " << fixed << setprecision(6) << m.homeWinProb << ",\n"
                   << "          \"draw_prob\": " << m.drawProb << ",\n"
                   << "          \"away_win_prob\": " << m.awayWinProb << ",\n"
                   << "          \"home_xg\": " << m.homeXg << ",\n"
                   << "          \"away_xg\": " << m.awayXg << ",\n"
                   << "          \"likely_scores\": [\n";
        for (size_t j = 0; j < topScores.size(); ++j) {
            jsonBuffer << "            { \"score\": \"" << topScores[j].score
                       << "\", \"prob\": " << topScores[j].prob << " }"
                       << (j + 1 < topScores.size() ? "," : "") << "\n";
        }
        jsonBuffer << "          ]\n"
                   << "        }" << (i + 1 < matches.size() ? "," : "") << "\n";
    }
    jsonBuffer << "      ],\n      \"table_probabilities\": [\n";
    for (size_t i = 0; i < displayTable.size(); ++i) {
        const auto& t = displayTable[i];
        jsonBuffer << "        {\n"
                   << "          \"team\": \"" << t.name << "\",\n"
                   << "          \"expected_points\": " << fixed << setprecision(2) << t.expectedPoints << ",\n"
                   << "          \"probabilities\": {\n"
                   << "            \"1st\": " << (t.finishPositions[0] / (double)SIMULATIONS) << ",\n"
                   << "            \"2nd\": " << (t.finishPositions[1] / (double)SIMULATIONS) << ",\n"
                   << "            \"3rd\": " << (t.finishPositions[2] / (double)SIMULATIONS) << ",\n"
                   << "            \"4th\": " << (t.finishPositions[3] / (double)SIMULATIONS) << "\n"
                   << "          }\n"
                   << "        }" << (i + 1 < displayTable.size() ? "," : "") << "\n";
    }
    jsonBuffer << "      ]\n    }";
    return displayTable;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    // Relative paths looking back one folder, then into 'data'
    string filepath = "../data/group_stage_predictions.csv";
    string jsonOutPath = "../data/simulation_matrices.json";

    cout << "========================================================\n";
    cout << "FOOTBALL WORLD CUP SIMULATION ENGINE\n";
    cout << "========================================================\n";


    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "\nERROR: Failed to open file! " << filepath << endl;
        return 1;
    }

    map<string, vector<MatchPrediction>> tournamentGroups;
    string line;
    getline(file, line); // header
    cout << "CSV header: " << line << "\n";

    int rowCount = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        MatchPrediction match;

        getline(ss, match.group, ',');
        getline(ss, match.homeTeam, ',');
        getline(ss, match.awayTeam, ',');
        getline(ss, cell, ','); match.homeXg = safe_stod(cell);
        getline(ss, cell, ','); match.awayXg = safe_stod(cell);

        // These will be recalculated in simulateGroup
        match.homeWinProb = 0.0;
        match.drawProb = 0.0;
        match.awayWinProb = 0.0;

        tournamentGroups[match.group].push_back(match);
        rowCount++;
    }
    file.close();

    cout << "Loaded " << rowCount << " matches across " << tournamentGroups.size() << " groups.\n";
    cout << "Booting simulation engine...\n";

    stringstream jsonBuffer;
    jsonBuffer << "{\n  \"simulation_date\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    jsonBuffer << "  \"groups\": {\n";

    size_t groupCount = 0;
    size_t totalGroups = tournamentGroups.size();
    for (auto& groupPair : tournamentGroups) {
        simulateGroup(groupPair.first, groupPair.second, jsonBuffer);
        groupCount++;
        if (groupCount < totalGroups) jsonBuffer << ",\n";
        else jsonBuffer << "\n";
    }
    jsonBuffer << "  }\n}\n";

    ofstream jsonFile(jsonOutPath);
    if (jsonFile.is_open()) {
        jsonFile << jsonBuffer.str();
        jsonFile.close();
        cout << "\n========================================================\n";
        cout << "✅ Success! JSON exported to:\n   " << jsonOutPath << "\n";
        cout << "📊 Includes: W/D/L probabilities, top 3 scorelines, table probabilities.\n";
    } else {
        cerr << "\nERROR: Couldn't write JSON to " << jsonOutPath << endl;
        return 1;
    }

    cout << "========================================================\n";
    cout << "Simulation complete!\n";
    return 0;
}