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

// Internal struct used inside the Monte Carlo execution loop for accurate tracking
struct SimTeam {
    string name;
    int points = 0;
    int goalDiff = 0;
    int goalsScored = 0;
};

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
    try {
        return stod(trimmed);
    } catch (...) {
        return 0.0;
    }
}

int safe_stoi(const string& s) {
    try {
        return stoi(s);
    } catch (...) {
        return 0;
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


void calculateMatchProbabilities(MatchPrediction& match) {
    if (match.isPlayed == 1) {
        if (match.homeXg > match.awayXg) { match.homeWinProb = 1.0; match.drawProb = 0.0; match.awayWinProb = 0.0; }
        else if (match.homeXg < match.awayXg) { match.homeWinProb = 0.0; match.drawProb = 0.0; match.awayWinProb = 1.0; }
        else { match.homeWinProb = 0.0; match.drawProb = 1.0; match.awayWinProb = 0.0; }
        return;
    }

    match.homeWinProb = 0.0; match.drawProb = 0.0; match.awayWinProb = 0.0;
    
    // Dixon-Coles parameter capturing low score dependence optimization
    const double rho = -0.12; 

    for (int h = 0; h <= 15; ++h) {
        for (int a = 0; a <= 15; ++a) {
            double p_home = poisson(h, match.homeXg);
            double p_away = poisson(a, match.awayXg);
            double prob = p_home * p_away;

            // Apply Dixon-Coles correction matrix to low scores
            if (h == 0 && a == 0)      prob *= (1.0 - match.homeXg * match.awayXg * rho);
            else if (h == 1 && a == 0) prob *= (1.0 + match.homeXg * rho);
            else if (h == 0 && a == 1) prob *= (1.0 + match.awayXg * rho);
            else if (h == 1 && a == 1) prob *= (1.0 - rho);

            if (h > a)       match.homeWinProb += prob;
            else if (h < a)  match.awayWinProb += prob;
            else             match.drawProb += prob;
        }
    }
    
    // Normalize probabilities to perfectly sum to 1.0 due to structural adjustment
    double totalProb = match.homeWinProb + match.drawProb + match.awayWinProb;
    match.homeWinProb /= totalProb;
    match.drawProb /= totalProb;
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
    sort(scoreProbs.begin(), scoreProbs.end(), [](const auto& a, const auto& b) { return get<0>(a) > get<0>(b); });
    vector<ScoreProb> result;
    for (int i = 0; i < min(topN, (int)scoreProbs.size()); ++i) {
        string scoreStr = to_string(get<1>(scoreProbs[i])) + "-" + to_string(get<2>(scoreProbs[i]));
        result.push_back({scoreStr, get<0>(scoreProbs[i])});
    }
    return result;
}

// ---------------------------------------------------------
// MID-TOURNAMENT HYBRID SIMULATION ENGINE
// ---------------------------------------------------------
vector<TeamStats> simulateGroup(const string& groupName, vector<MatchPrediction>& matches, stringstream& jsonBuffer) {
    map<string, TeamStats> groupTeams;
    for (auto& m : matches) {
        calculateMatchProbabilities(m);
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

    // Pre-calculate distribution frameworks outside the main simulation loop for top performance optimization
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
                // Generates dynamic goal metrics on every single execution step
                hGoals = homeDists[mIdx](gen);
                aGoals = awayDists[mIdx](gen);
            }
            
            // Log core tie-breaker data arrays
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
        for (const auto& pair : simTable) {
            iterationTable.push_back(pair.second);
        }
        
        // Comprehensive FIFA Standings Sort: Points -> Goal Difference -> Goals Scored -> Fallback Name
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

    // Console output log
    cout << "\n========================================\n GROUP " << groupName << " MID-TOURNAMENT STATUS\n========================================\n";
    cout << left << setw(22) << "Team" << setw(12) << "Exp. Pts" << setw(10) << "1st %" << setw(10) << "2nd %" << setw(10) << "3rd %" << setw(10) << "4th %" << endl;
    cout << string(74, '-') << endl;
    for (const auto& t : displayTable) {
        cout << left << setw(22) << t.name << setw(12) << fixed << setprecision(1) << t.expectedPoints;
        for (int pos = 0; pos < 4; ++pos)
            cout << setw(10) << (t.finishPositions[pos] / (double)SIMULATIONS) * 100.0;
        cout << endl;
    }

    // Versioned JSON serialization string structure
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
                   << "          \"is_played\": " << m.isPlayed << ",\n"
                   << "          \"likely_scores\": [\n";
        for (size_t j = 0; j < topScores.size(); ++j) {
            jsonBuffer << "            { \"score\": \"" << topScores[j].score << "\", \"prob\": " << topScores[j].prob << " }" << (j + 1 < topScores.size() ? "," : "") << "\n";
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
    getline(file, line); // Skip CSV header line

    int rowCount = 0;
    while (getline(file, line)) {
        if(line.empty()) continue;
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
        cout << "✅ Success! Hybrid simulation data matrix saved to:\n   " << jsonOutPath << "\n";
    } else {
        cerr << "\nERROR: Target write stream failed for: " << jsonOutPath << endl;
        return 1;
    }

    return 0;
}