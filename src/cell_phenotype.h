#ifndef PHYSICELL_CELL_PHENOTYPE_H
#define PHYSICELL_CELL_PHENOTYPE_H

// ─────────────────────────────────────────────────────────────────────
// CPU-side phenotype updates — Tier 1 Feature Parity
//
// Implements the full PhysiCell phenotype models:
//   - Multi-compartment volume ODE (fluid/solid/nuclear/cytoplasmic)
//   - Phase-based cell cycle (live model + Ki67 basic)
//   - Full death models (apoptosis + necrosis with lysis)
//   - Substrate secretion/uptake with implicit volume-scaled solve
//   - Cell division with proper compartment halving
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"
#include "../shaders/types.h"

class CellMechanics;

// ─── Death model codes ───
static constexpr uint32_t DEATH_NONE      = 0;
static constexpr uint32_t DEATH_APOPTOSIS = 1;
static constexpr uint32_t DEATH_NECROSIS  = 2;

// ─── Cycle model codes (matching PhysiCell constants) ───
static constexpr uint32_t CYCLE_KI67_ADVANCED     = 0;  // Ki67 advanced (3 phases)
static constexpr uint32_t CYCLE_KI67_BASIC        = 1;  // Ki67 basic (2 phases)
static constexpr uint32_t CYCLE_FLOW_CYTO         = 2;  // flow cytometry basic (4 phases)
static constexpr uint32_t CYCLE_LIVE              = 5;  // live cells cycle model (1 phase)
static constexpr uint32_t CYCLE_FLOW_CYTO_SEP     = 6;  // flow cytometry separated (alias)
static constexpr uint32_t CYCLE_CYCLING_QUIESCENT = 7;  // cycling-quiescent (2 phases)

// ─── Phase indices ───
// For live model:  phase 0 = Live (division_at_phase_exit)
// For Ki67 basic:  phase 0 = Ki67-, phase 1 = Ki67+
// For Ki67 advanced: phase 0 = Ki67- (Q), phase 1 = Ki67+ pre-M, phase 2 = Ki67+ post-M
// For flow cytometry: phase 0 = G0/G1, phase 1 = S, phase 2 = G2, phase 3 = M
// For cycling-quiescent: phase 0 = Quiescent, phase 1 = Cycling
static constexpr uint32_t PHASE_KI67_NEG = 0;
static constexpr uint32_t PHASE_KI67_POS = 1;

/// Update phenotype for all alive cells.
/// Called once per dt_phenotype timestep.
/// If `mech` is provided, uses GPU spatial hash for efficient interaction lookups.
void updatePhenotype(CellData& cells, double dt, double current_time,
                     const float* density, const GridParams& grid,
                     const CellMechanics* mech = nullptr);

/// Advance the cell cycle for cell i using the full phase-based model.
/// Supports live, Ki67 basic, Ki67 advanced, flow cytometry, and
/// cycling-quiescent models.  Returns true if cell should divide.
bool updateCellCycle(CellData& cells, uint32_t i, double dt,
                     const float* density, const GridParams& grid);

/// Full PhysiCell volume ODE for cell i:
/// Updates fluid, solid_cytoplasmic, solid_nuclear, calcified_fraction,
/// then reconstructs total volume and updates geometry.
void updateVolume(CellData& cells, uint32_t i, double dt);

/// Check stochastic death (apoptosis) and O₂-threshold necrosis for cell i.
/// If death is triggered, sets up the appropriate death model parameters.
void updateDeath(CellData& cells, uint32_t i, double dt,
                 const float* density, const GridParams& grid);

/// Advance the death model for a dead cell (apoptosis shrinkage,
/// necrosis swelling → lysis → shrinkage).
/// Returns true if the cell should be removed.
bool advanceDeathModel(CellData& cells, uint32_t i, double dt);

/// Process secretion/uptake for cell i against the substrate density grid.
/// Uses the volume-scaled implicit solve from PhysiCell/BioFVM.
void processSecretion(CellData& cells, float* density,
                      const GridParams& grid, uint32_t i, double dt);

/// Handle cell division: halve all volume compartments, create daughter cell.
/// Returns the index of the new daughter cell, or UINT32_MAX on failure.
uint32_t divideCell(CellData& cells, uint32_t parent_index);

/// Run contact-based cell-cell interactions for cell i.
/// Uses the GPU spatial hash (via CellMechanics) for efficient neighbor lookup.
/// Falls back to brute-force if mech is nullptr.
void standardCellCellInteractions(CellData& cells, uint32_t i, double dt,
                                   const GridParams& grid,
                                   const CellMechanics* mech = nullptr);

/// Run cell transformations for cell i.
/// Checks per-type transformation rates and stochastically converts
/// the cell to a new type. Matches PhysiCell's standard_cell_transformations().
void standardCellTransformations(CellData& cells, uint32_t i, double dt);

/// Update cell integrity (damage accumulation/repair) for cell i.
void updateCellIntegrity(CellData& cells, uint32_t i, double dt);

/// Convert cell i to a new cell type by re-applying defaults from the registry.
class CellTypeRegistry;
void convertCellType(CellData& cells, uint32_t i, int new_type_id,
                     const CellTypeRegistry& registry);

#endif // PHYSICELL_CELL_PHENOTYPE_H
