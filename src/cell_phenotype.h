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

// ─── Death model codes ───
static constexpr uint32_t DEATH_NONE      = 0;
static constexpr uint32_t DEATH_APOPTOSIS = 1;
static constexpr uint32_t DEATH_NECROSIS  = 2;

// ─── Cycle model codes (matching PhysiCell constants) ───
static constexpr uint32_t CYCLE_LIVE       = 5;  // live cells cycle model
static constexpr uint32_t CYCLE_KI67_BASIC = 1;  // Ki67 basic

// ─── Phase indices ───
// For live model:  phase 0 = Live (division_at_phase_exit)
// For Ki67 basic:  phase 0 = Ki67-, phase 1 = Ki67+
static constexpr uint32_t PHASE_KI67_NEG = 0;
static constexpr uint32_t PHASE_KI67_POS = 1;

/// Update phenotype for all alive cells.
/// Called once per dt_phenotype timestep.
void updatePhenotype(CellData& cells, double dt, double current_time,
                     const float* density, const GridParams& grid);

/// Advance the cell cycle for cell i using the full phase-based model.
/// Supports live (single-phase) and Ki67 basic (two-phase) models.
/// Returns true if cell should divide.
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

#endif // PHYSICELL_CELL_PHENOTYPE_H
