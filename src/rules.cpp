// ─────────────────────────────────────────────────────────────────────
// Rule Engine — Implementation
//
// Hill function rule engine: maps signals → behaviors.
// Port of PhysiCell's Hypothesis_Rule / rule_phenotype_function to SoA.
//
// Hill function:
//   increasing: base + (max - base) * s^n / (s^n + K^n)
//   decreasing: base + (min - base) * s^n / (s^n + K^n)
//
// CSV format (PhysiCell v3):
//   cell_type, signal, direction, behavior, base_value,
//   max_response_value, half_max, hill_power, applies_to_dead
//
// Reference: PhysiCell core/PhysiCell_rules.cpp
// ─────────────────────────────────────────────────────────────────────

#include "rules.h"
#include "signals.h"
#include "behaviors.h"
#include "cell_data.h"
#include "../shaders/types.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

// pugixml forward — only needed for XML parsing
#include "../lib/pugixml/pugixml.hpp"

namespace PhysiCellMetal {

// ─── CSV utilities ──────────────────────────────────────────────────

void splitCSV(const std::string& input, std::vector<std::string>& output,
              char delim)
{
    output.clear();
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, delim)) {
        output.push_back(token);
    }
}

std::string trimString(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ─── Hill function ──────────────────────────────────────────────────

double RuleEngine::evaluateHill(const HypothesisRule& rule, double s)
{
    double K = rule.half_max;
    double n = rule.hill_power;

    // Guard against degenerate cases
    if (K <= 0.0) {
        // If half_max is 0, Hill function is essentially a step function
        return rule.increasing ? rule.max_value : rule.min_value;
    }
    if (s < 0.0) s = 0.0;

    double s_n = std::pow(s, n);
    double K_n = std::pow(K, n);
    double denom = s_n + K_n;

    if (denom <= 0.0) {
        return rule.base_value;
    }

    if (rule.increasing) {
        // base + (max - base) * s^n / (s^n + K^n)
        return rule.base_value +
               (rule.max_value - rule.base_value) * s_n / denom;
    } else {
        // base + (min - base) * s^n / (s^n + K^n)
        return rule.base_value +
               (rule.min_value - rule.base_value) * s_n / denom;
    }
}

// ─── Rule management ────────────────────────────────────────────────

void RuleEngine::addRule(const HypothesisRule& rule) {
    rules_.push_back(rule);
}

void RuleEngine::displayRules() const {
    std::cout << "Rules (" << rules_.size() << " total):" << std::endl;
    std::cout << "======" << std::endl;
    for (size_t i = 0; i < rules_.size(); i++) {
        const auto& r = rules_[i];
        std::string dir = r.increasing ? "increases" : "decreases";
        std::cout << "  [" << i << "] "
                  << r.cell_type_name << ": "
                  << signalName(r.signal_index) << " " << dir << " "
                  << behaviorName(r.behavior_index)
                  << " | base=" << r.base_value
                  << " " << (r.increasing ? "max=" : "min=")
                  << (r.increasing ? r.max_value : r.min_value)
                  << " K=" << r.half_max
                  << " n=" << r.hill_power
                  << (r.applies_to_dead ? " [applies to dead]" : "")
                  << std::endl;
    }
    std::cout << std::endl;
}

// ─── CSV loading ────────────────────────────────────────────────────

void RuleEngine::loadRulesCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open rules CSV file: "
                  << filepath << std::endl;
        return;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;

        // Skip empty lines and comments
        std::string trimmed = trimString(line);
        if (trimmed.empty() || trimmed[0] == '#' || trimmed[0] == '/') {
            continue;
        }

        // Also skip header lines
        if (trimmed.find("cell_type") != std::string::npos &&
            trimmed.find("signal") != std::string::npos) {
            continue;
        }

        std::vector<std::string> tokens;
        splitCSV(trimmed, tokens);

        // Need at least 8 fields:
        // cell_type, signal, direction, behavior, base_value,
        // max_response_value, half_max, hill_power
        if (tokens.size() < 8) {
            std::cerr << "Warning: Skipping malformed rule at line "
                      << line_num << " (need >= 8 fields, got "
                      << tokens.size() << ")" << std::endl;
            continue;
        }

        HypothesisRule rule;
        rule.cell_type_name = trimString(tokens[0]);
        // cell_type: -1 means "all" or "default"
        if (rule.cell_type_name == "all" || rule.cell_type_name == "default") {
            rule.cell_type = -1;
        } else {
            // Try to parse as integer, otherwise leave as -1 (will match all)
            try {
                rule.cell_type = std::stoi(rule.cell_type_name);
            } catch (...) {
                rule.cell_type = -1;
            }
        }

        std::string signal_name = trimString(tokens[1]);
        rule.signal_index = findSignalIndex(signal_name);
        if (rule.signal_index < 0) {
            std::cerr << "Warning: Unknown signal '" << signal_name
                      << "' at line " << line_num << ", skipping rule."
                      << std::endl;
            continue;
        }

        std::string direction = trimString(tokens[2]);
        rule.increasing = (direction == "increases" || direction == "increase"
                        || direction == "up" || direction == "promotes");

        std::string behavior_name = trimString(tokens[3]);
        rule.behavior_index = findBehaviorIndex(behavior_name);
        if (rule.behavior_index < 0) {
            std::cerr << "Warning: Unknown behavior '" << behavior_name
                      << "' at line " << line_num << ", skipping rule."
                      << std::endl;
            continue;
        }

        rule.base_value = std::stod(trimString(tokens[4]));

        // tokens[5] is max_response_value — for increasing it's max_value,
        // for decreasing it's min_value
        double response_val = std::stod(trimString(tokens[5]));
        if (rule.increasing) {
            rule.max_value = response_val;
            rule.min_value = rule.base_value;  // not used for increasing
        } else {
            rule.min_value = response_val;
            rule.max_value = rule.base_value;  // not used for decreasing
        }

        rule.half_max = std::stod(trimString(tokens[6]));
        rule.hill_power = std::stod(trimString(tokens[7]));

        // Optional 9th field: applies_to_dead
        if (tokens.size() >= 9) {
            std::string dead_str = trimString(tokens[8]);
            rule.applies_to_dead = (dead_str == "1" || dead_str == "true"
                                 || dead_str == "yes");
        }

        addRule(rule);
    }

    std::cout << "Loaded " << rules_.size() << " rules from " << filepath
              << std::endl;
}

// ─── XML loading ────────────────────────────────────────────────────

void RuleEngine::loadRulesFromConfig(const pugi::xml_node& config) {
    // Look for <cell_rules> element
    pugi::xml_node cell_rules = config.child("cell_rules");
    if (!cell_rules) {
        // No rules in config — that's fine
        return;
    }

    // Check for CSV-based rules
    pugi::xml_node rulesets = cell_rules.child("rulesets");
    if (rulesets) {
        for (pugi::xml_node ruleset = rulesets.child("ruleset");
             ruleset; ruleset = ruleset.next_sibling("ruleset")) {

            bool enabled = ruleset.attribute("enabled").as_bool(true);
            if (!enabled) continue;

            std::string protocol = ruleset.attribute("protocol").as_string("");

            // Check for CSV filename
            pugi::xml_node folder_node = ruleset.child("folder");
            pugi::xml_node filename_node = ruleset.child("filename");

            if (filename_node) {
                std::string folder = folder_node ?
                    folder_node.text().as_string(".") : ".";
                std::string filename = filename_node.text().as_string("");

                if (!filename.empty()) {
                    std::string filepath = folder + "/" + filename;
                    std::cout << "Loading rules from CSV: " << filepath
                              << " (protocol: " << protocol << ")"
                              << std::endl;
                    loadRulesCSV(filepath);
                }
            }
        }
    }
}

// ─── Rule application ───────────────────────────────────────────────

void RuleEngine::apply(CellData& cells, uint32_t cell_idx,
                       const float* density, const GridParams& grid,
                       double current_time)
{
    bool cell_is_dead = (cells.is_alive[cell_idx] == 0);
    int cell_type = static_cast<int>(cells.cell_type[cell_idx]);

    for (const auto& rule : rules_) {
        // Check if rule applies to this cell type
        if (rule.cell_type >= 0 && rule.cell_type != cell_type) {
            continue;
        }

        // Check if rule applies to dead cells
        if (cell_is_dead && !rule.applies_to_dead) {
            continue;
        }

        // Get the signal value
        double signal_val = getSignal(cells, cell_idx, rule.signal_index,
                                      density, grid, current_time);

        // Evaluate the Hill function
        double new_behavior = evaluateHill(rule, signal_val);

        // Set the behavior
        setBehavior(cells, cell_idx, rule.behavior_index, new_behavior);
    }
}

void RuleEngine::applyAll(CellData& cells, const float* density,
                          const GridParams& grid, double current_time)
{
    if (rules_.empty()) return;

    uint32_t n = cells.num_cells;
    for (uint32_t i = 0; i < n; i++) {
        apply(cells, i, density, grid, current_time);
    }
}

} // namespace PhysiCellMetal
