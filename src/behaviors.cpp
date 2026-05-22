// ─────────────────────────────────────────────────────────────────────
// Behavior System — Implementation
//
// Modifies cell phenotype parameters through dictionary-indexed access.
// Port of PhysiCell's set_single_behavior / get_single_behavior to SoA.
//
// Reference: PhysiCell core/PhysiCell_signal_behavior.cpp
// ─────────────────────────────────────────────────────────────────────

#include "behaviors.h"
#include "cell_data.h"
#include "cell_definitions.h"

#include <iostream>
#include <cmath>

namespace PhysiCellMetal {

// ─── Module-level state ──────────────────────────────────────────────

static std::map<std::string, int> behavior_to_int;
static std::map<int, std::string> int_to_behavior;
static int n_substrates_cached = 0;

// Offsets into the behavior index space (computed at init)
static int off_secretion = 0;
static int off_uptake = 0;
static int off_cycle_entry = 0;
static int off_apoptosis = 0;
static int off_necrosis = 0;
static int off_migration_speed = 0;
static int off_migration_bias = 0;
static int off_adhesion = 0;
static int off_repulsion = 0;
static int off_max_adh_dist = 0;

// ─── Dictionary construction ─────────────────────────────────────────

void initBehaviorDictionary(int n_substrates,
                            const std::vector<std::string>& substrate_names)
{
    behavior_to_int.clear();
    int_to_behavior.clear();
    n_substrates_cached = n_substrates;

    int idx = 0;

    // [0 .. m-1] secretion rates
    off_secretion = idx;
    for (int i = 0; i < n_substrates; i++) {
        std::string name = substrate_names[i] + " secretion";
        behavior_to_int[name] = idx;
        int_to_behavior[idx] = name;
        idx++;
    }

    // [m .. 2m-1] uptake rates
    off_uptake = idx;
    for (int i = 0; i < n_substrates; i++) {
        std::string name = substrate_names[i] + " uptake";
        behavior_to_int[name] = idx;
        int_to_behavior[idx] = name;
        idx++;
    }

    // cycle entry rate
    off_cycle_entry = idx;
    behavior_to_int["cycle entry"] = idx;
    int_to_behavior[idx] = "cycle entry";
    behavior_to_int["exit from cycle phase 0"] = idx;
    behavior_to_int["cycle rate"] = idx;
    idx++;

    // apoptosis rate
    off_apoptosis = idx;
    behavior_to_int["apoptosis"] = idx;
    int_to_behavior[idx] = "apoptosis";
    behavior_to_int["apoptosis rate"] = idx;
    behavior_to_int["death rate"] = idx;
    idx++;

    // necrosis rate
    off_necrosis = idx;
    behavior_to_int["necrosis"] = idx;
    int_to_behavior[idx] = "necrosis";
    behavior_to_int["necrosis rate"] = idx;
    idx++;

    // migration speed
    off_migration_speed = idx;
    behavior_to_int["migration speed"] = idx;
    int_to_behavior[idx] = "migration speed";
    idx++;

    // migration bias
    off_migration_bias = idx;
    behavior_to_int["migration bias"] = idx;
    int_to_behavior[idx] = "migration bias";
    idx++;

    // cell-cell adhesion
    off_adhesion = idx;
    behavior_to_int["cell-cell adhesion"] = idx;
    int_to_behavior[idx] = "cell-cell adhesion";
    behavior_to_int["adhesion"] = idx;
    idx++;

    // cell-cell repulsion
    off_repulsion = idx;
    behavior_to_int["cell-cell repulsion"] = idx;
    int_to_behavior[idx] = "cell-cell repulsion";
    behavior_to_int["repulsion"] = idx;
    idx++;

    // relative maximum adhesion distance
    off_max_adh_dist = idx;
    behavior_to_int["relative maximum adhesion distance"] = idx;
    int_to_behavior[idx] = "relative maximum adhesion distance";
    idx++;
}

int findBehaviorIndex(const std::string& name) {
    auto it = behavior_to_int.find(name);
    if (it != behavior_to_int.end()) {
        return it->second;
    }
    std::cerr << "Warning: behavior '" << name << "' not found in dictionary."
              << std::endl;
    return -1;
}

std::string behaviorName(int index) {
    auto it = int_to_behavior.find(index);
    if (it != int_to_behavior.end()) {
        return it->second;
    }
    return "unknown";
}

int numBehaviors() {
    return static_cast<int>(int_to_behavior.size());
}

void displayBehaviorDictionary() {
    std::cout << "Behaviors:" << std::endl;
    std::cout << "==========" << std::endl;
    for (auto& [idx, name] : int_to_behavior) {
        std::cout << "  " << idx << " : " << name << std::endl;
    }
    std::cout << std::endl;
}

// ─── Behavior setters ────────────────────────────────────────────────

void setBehavior(CellData& cells, uint32_t cell_idx,
                 int behavior_id, double value)
{
    if (behavior_id < 0) {
        std::cerr << "Warning: setBehavior called with negative behavior_id"
                  << std::endl;
        return;
    }

    int m = n_substrates_cached;

    // Secretion rate [0 .. m-1]
    if (behavior_id >= off_secretion && behavior_id < off_secretion + m) {
        int s = behavior_id - off_secretion;
        if (cells.secretion_rate) {
            size_t offset = static_cast<size_t>(cell_idx) * cells.n_substrates + s;
            cells.secretion_rate[offset] = value;
        }
        return;
    }

    // Uptake rate [m .. 2m-1]
    if (behavior_id >= off_uptake && behavior_id < off_uptake + m) {
        int s = behavior_id - off_uptake;
        if (cells.uptake_rate) {
            size_t offset = static_cast<size_t>(cell_idx) * cells.n_substrates + s;
            cells.uptake_rate[offset] = value;
        }
        return;
    }

    // Cycle entry rate
    if (behavior_id == off_cycle_entry) {
        cells.transition_rate_01[cell_idx] = value;
        return;
    }

    // Apoptosis rate
    if (behavior_id == off_apoptosis) {
        cells.death_rate[cell_idx] = value;
        return;
    }

    // Necrosis rate
    if (behavior_id == off_necrosis) {
        cells.necrosis_rate[cell_idx] = value;
        return;
    }

    // Migration speed
    if (behavior_id == off_migration_speed) {
        cells.motility_speed[cell_idx] = static_cast<float>(value);
        return;
    }

    // Migration bias
    if (behavior_id == off_migration_bias) {
        cells.motility_bias[cell_idx] = static_cast<float>(value);
        return;
    }

    // Cell-cell adhesion
    if (behavior_id == off_adhesion) {
        cells.cell_cell_adhesion[cell_idx] = static_cast<float>(value);
        return;
    }

    // Cell-cell repulsion
    if (behavior_id == off_repulsion) {
        cells.cell_cell_repulsion[cell_idx] = static_cast<float>(value);
        return;
    }

    // Relative max adhesion distance
    if (behavior_id == off_max_adh_dist) {
        cells.relative_max_adhesion_distance[cell_idx] = static_cast<float>(value);
        return;
    }

    std::cerr << "Warning: setBehavior called with unknown behavior_id "
              << behavior_id << std::endl;
}

// ─── Behavior getters ────────────────────────────────────────────────

double getBehavior(const CellData& cells, uint32_t cell_idx,
                   int behavior_id)
{
    if (behavior_id < 0) return 0.0;

    int m = n_substrates_cached;

    // Secretion rate
    if (behavior_id >= off_secretion && behavior_id < off_secretion + m) {
        int s = behavior_id - off_secretion;
        if (cells.secretion_rate) {
            size_t offset = static_cast<size_t>(cell_idx) * cells.n_substrates + s;
            return cells.secretion_rate[offset];
        }
        return 0.0;
    }

    // Uptake rate
    if (behavior_id >= off_uptake && behavior_id < off_uptake + m) {
        int s = behavior_id - off_uptake;
        if (cells.uptake_rate) {
            size_t offset = static_cast<size_t>(cell_idx) * cells.n_substrates + s;
            return cells.uptake_rate[offset];
        }
        return 0.0;
    }

    // Cycle entry rate
    if (behavior_id == off_cycle_entry) {
        return cells.transition_rate_01[cell_idx];
    }

    // Apoptosis rate
    if (behavior_id == off_apoptosis) {
        return cells.death_rate[cell_idx];
    }

    // Necrosis rate
    if (behavior_id == off_necrosis) {
        return cells.necrosis_rate[cell_idx];
    }

    // Migration speed
    if (behavior_id == off_migration_speed) {
        return static_cast<double>(cells.motility_speed[cell_idx]);
    }

    // Migration bias
    if (behavior_id == off_migration_bias) {
        return static_cast<double>(cells.motility_bias[cell_idx]);
    }

    // Cell-cell adhesion
    if (behavior_id == off_adhesion) {
        return static_cast<double>(cells.cell_cell_adhesion[cell_idx]);
    }

    // Cell-cell repulsion
    if (behavior_id == off_repulsion) {
        return static_cast<double>(cells.cell_cell_repulsion[cell_idx]);
    }

    // Relative max adhesion distance
    if (behavior_id == off_max_adh_dist) {
        return static_cast<double>(cells.relative_max_adhesion_distance[cell_idx]);
    }

    return 0.0;
}

// ─── Base behavior (from cell type defaults) ─────────────────────────

double getBaseBehavior(const CellTypeRegistry& registry,
                       const CellData& cells, uint32_t cell_idx,
                       int behavior_id)
{
    if (behavior_id < 0) return 0.0;

    int type_id = static_cast<int>(cells.cell_type[cell_idx]);
    const CellTypeDefaults& def = registry.getType(type_id);

    int m = n_substrates_cached;

    // Secretion rate (base from cell type defaults — only first substrate)
    if (behavior_id >= off_secretion && behavior_id < off_secretion + m) {
        int s = behavior_id - off_secretion;
        return (s == 0) ? static_cast<double>(def.secretion_rate) : 0.0;
    }

    // Uptake rate (base from cell type defaults — only first substrate)
    if (behavior_id >= off_uptake && behavior_id < off_uptake + m) {
        int s = behavior_id - off_uptake;
        return (s == 0) ? static_cast<double>(def.uptake_rate) : 0.0;
    }

    if (behavior_id == off_cycle_entry) {
        return static_cast<double>(def.cycle_rate);
    }

    if (behavior_id == off_apoptosis) {
        return static_cast<double>(def.apoptosis_rate);
    }

    if (behavior_id == off_necrosis) {
        return static_cast<double>(def.necrosis_rate);
    }

    if (behavior_id == off_migration_speed) {
        return static_cast<double>(def.motility_speed);
    }

    if (behavior_id == off_migration_bias) {
        return static_cast<double>(def.motility_bias);
    }

    if (behavior_id == off_adhesion) {
        return static_cast<double>(def.cell_cell_adhesion);
    }

    if (behavior_id == off_repulsion) {
        return static_cast<double>(def.cell_cell_repulsion);
    }

    if (behavior_id == off_max_adh_dist) {
        return static_cast<double>(def.max_adhesion_distance);
    }

    return 0.0;
}

// ─── Convenience name-based access ───────────────────────────────────

void setBehaviorByName(CellData& cells, uint32_t cell_idx,
                       const std::string& name, double value)
{
    int idx = findBehaviorIndex(name);
    setBehavior(cells, cell_idx, idx, value);
}

double getBehaviorByName(const CellData& cells, uint32_t cell_idx,
                         const std::string& name)
{
    int idx = findBehaviorIndex(name);
    return getBehavior(cells, cell_idx, idx);
}

} // namespace PhysiCellMetal
