#ifndef PHYSICELL_METAL_RULES_H
#define PHYSICELL_METAL_RULES_H

// ─────────────────────────────────────────────────────────────────────
// Rule Engine — Tier 4 (Signal/Behavior/Rules)
//
// Hill function rule engine: maps signals → behaviors.
// Port of PhysiCell's Hypothesis_Rule / rule_phenotype_function.
//
// Each rule: "signal X {increases|decreases} behavior Y in cell type Z"
//   - increasing: base + (max - base) * s^n / (s^n + K^n)
//   - decreasing: base + (min - base) * s^n / (s^n + K^n)
//
// Rules can be loaded from CSV or XML config.
// ─────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>

// Forward declarations
class CellData;
class CellTypeRegistry;
struct GridParams;

namespace pugi { class xml_node; }

namespace PhysiCellMetal {

struct HypothesisRule {
    std::string cell_type_name;     // cell type name (for display/debug)
    int cell_type = -1;             // which cell type this applies to (-1 = all)
    int signal_index = -1;          // input signal (from signal dictionary)
    int behavior_index = -1;        // output behavior (from behavior dictionary)
    bool increasing = true;         // true = up-regulation, false = down-regulation
    double half_max = 0.5;          // Hill function half-max (K)
    double hill_power = 2.0;        // Hill exponent (n), default 2
    double min_value = 0.0;         // minimum behavior value (for decreasing)
    double max_value = 1.0;         // maximum behavior value (for increasing)
    double base_value = 0.0;        // base behavior value (from cell defaults)
    bool applies_to_dead = false;   // should this rule apply to dead cells?
};

class RuleEngine {
public:
    RuleEngine() = default;

    /// Add a single rule to the engine.
    void addRule(const HypothesisRule& rule);

    /// Load rules from a CSV file (PhysiCell v3 format).
    /// Format: cell_type, signal, direction, behavior, base_value,
    ///         max_response_value, half_max, hill_power, applies_to_dead
    void loadRulesCSV(const std::string& filepath);

    /// Load rules from an XML config node.
    /// Looks for <cell_rules> / <rulesets> / <ruleset> elements.
    void loadRulesFromConfig(const pugi::xml_node& config);

    /// Evaluate all rules for a single cell, modifying its behaviors.
    void apply(CellData& cells, uint32_t cell_idx,
               const float* density, const GridParams& grid,
               double current_time);

    /// Apply all rules to all alive cells.
    void applyAll(CellData& cells, const float* density,
                  const GridParams& grid, double current_time);

    /// Number of rules loaded.
    int numRules() const { return static_cast<int>(rules_.size()); }

    /// Access rules (read-only, for display/debug).
    const std::vector<HypothesisRule>& rules() const { return rules_; }

    /// Display all rules to stdout.
    void displayRules() const;

private:
    std::vector<HypothesisRule> rules_;

    /// Hill function evaluation:
    ///   increasing: base + (max - base) * s^n / (s^n + K^n)
    ///   decreasing: base + (min - base) * s^n / (s^n + K^n)
    static double evaluateHill(const HypothesisRule& rule, double signal_value);
};

// ─── CSV parsing utilities ───

/// Split a CSV line into tokens by delimiter.
void splitCSV(const std::string& input, std::vector<std::string>& output,
              char delim = ',');

/// Trim leading/trailing whitespace from a string.
std::string trimString(const std::string& s);

} // namespace PhysiCellMetal

#endif // PHYSICELL_METAL_RULES_H
