#ifndef PHYSICELL_METAL_MOTILITY_H
#define PHYSICELL_METAL_MOTILITY_H

// ─────────────────────────────────────────────────────────────────────
// Motility — persistent random walk with chemotaxis bias
//
// Runs on CPU (complex branching, RNG). Writes motility contribution
// into the GPU-shared velocity fields so the GPU integrator picks it up.
//
// Port of PhysiCell Cell::update_motility_vector() to SoA layout.
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"
#include "../shaders/types.h"

// ─── Per-cell motility state (CPU-only, double precision) ────────────
// These are stored externally to CellData so we don't modify cell_data.h.
// The caller allocates this alongside CellData and keeps them in sync.

struct MotilityData {
    uint32_t max_cells;
    uint32_t num_cells;    // mirrors CellData::num_cells

    // Persistence timing
    double* persistence_time;   // mean time between direction changes (min)
    double* motility_elapsed;   // time since last direction change (min)

    // Motility bias strength (CPU-side double for precision; the GPU
    // float motility_bias in CellData is written from this each step)
    double* motility_bias;      // 0..1

    // Current motility vector (direction * speed)
    double* motility_vec_x;
    double* motility_vec_y;
    double* motility_vec_z;

    // Flags
    uint32_t* is_motile;        // 0 = not motile, 1 = motile
    uint32_t* restrict_to_2D;   // 0 = 3D, 1 = 2D only

    // Chemotaxis
    int32_t* chemotaxis_index;      // substrate index for gradient bias (legacy single-substrate)
    int32_t* chemotaxis_direction;  // +1 = up gradient, -1 = down (legacy)

    // Multi-substrate chemotaxis
    double* chemotactic_sensitivity;  // size = max_cells * n_substrates
    uint32_t n_substrates;            // number of substrates (set in allocateSubstrate)

    MotilityData() = default;

    /// Allocate all arrays for max_cells. Zero-initializes.
    void allocate(uint32_t max_cells);

    /// Allocate per-substrate chemotactic_sensitivity array.
    /// Must be called after allocate() and after n_substrates is known.
    void allocateSubstrate(uint32_t n_substrates);

    /// Free all arrays.
    void free_all();

    /// Set defaults for cell at index.
    void setDefaults(uint32_t index);

    /// Swap-remove cell (mirrors CellData::removeCell).
    void swapRemove(uint32_t index, uint32_t last);
};

// ─── Main update function ────────────────────────────────────────────

/// Update motility vectors for all cells, then add motility contribution
/// to CellData velocity fields. Call BEFORE the GPU mechanics dispatch
/// so that motility is included in the integrated velocity.
///
/// @param cells       SoA cell data (GPU-shared fields)
/// @param mot         Per-cell motility state (CPU-only)
/// @param density     Microenvironment density buffer (for gradient computation)
/// @param grid        Grid parameters (for gradient computation)
/// @param dt          Phenotype timestep (minutes)
void updateMotility(CellData& cells, MotilityData& mot,
                    const float* density, const GridParams& grid,
                    double dt);

#endif // PHYSICELL_METAL_MOTILITY_H
