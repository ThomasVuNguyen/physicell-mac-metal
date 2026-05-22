#ifndef PHYSICELL_METAL_BEHAVIORS_H
#define PHYSICELL_METAL_BEHAVIORS_H

// ─────────────────────────────────────────────────────────────────────
// Behavior System — Tier 4 (Signal/Behavior/Rules)
//
// Modifies cell parameters (phenotype) through a dictionary-indexed
// interface. Port of PhysiCell's set_single_behavior to SoA.
//
// Behavior index layout (dynamic, built at init):
//   [0 .. m-1]         secretion rates (per substrate)
//   [m .. 2m-1]        uptake rates (per substrate)
//   [2m]               cycle entry rate (transition_rate_01)
//   [2m+1]             apoptosis rate (death_rate)
//   [2m+2]             necrosis rate
//   [2m+3]             migration speed
//   [2m+4]             migration bias
//   [2m+5]             cell-cell adhesion
//   [2m+6]             cell-cell repulsion
//   [2m+7]             relative max adhesion distance
//
// Where m = number of substrates.
// ─────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>
#include <map>

class CellData;
class CellTypeRegistry;

namespace PhysiCellMetal {

// ─── Behavior dictionary ───

/// Initialize the behavior name → index mapping.
void initBehaviorDictionary(int n_substrates,
                            const std::vector<std::string>& substrate_names);

/// Find behavior index by name. Returns -1 if not found.
int findBehaviorIndex(const std::string& name);

/// Get behavior name from index. Returns "unknown" if out of range.
std::string behaviorName(int index);

/// Total number of registered behaviors.
int numBehaviors();

/// Display the behavior dictionary to stdout.
void displayBehaviorDictionary();

// ─── Behavior get/set ───

/// Set a single behavior (phenotype parameter) for a cell.
void setBehavior(CellData& cells, uint32_t cell_idx,
                 int behavior_id, double value);

/// Get a single current behavior value from a cell.
double getBehavior(const CellData& cells, uint32_t cell_idx,
                   int behavior_id);

/// Get the base (default) behavior value from the cell type definition.
double getBaseBehavior(const CellTypeRegistry& registry,
                       const CellData& cells, uint32_t cell_idx,
                       int behavior_id);

/// Convenience: set behavior by name.
void setBehaviorByName(CellData& cells, uint32_t cell_idx,
                       const std::string& name, double value);

/// Convenience: get behavior by name.
double getBehaviorByName(const CellData& cells, uint32_t cell_idx,
                         const std::string& name);

} // namespace PhysiCellMetal

#endif // PHYSICELL_METAL_BEHAVIORS_H
