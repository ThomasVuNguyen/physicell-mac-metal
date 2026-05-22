#ifndef PHYSICELL_METAL_CELL_MECHANICS_H
#define PHYSICELL_METAL_CELL_MECHANICS_H

#include "cell_data.h"
#include "microenvironment.h"
#include "../shaders/types.h"

class MetalContext;

// ─────────────────────────────────────────────────────────────────────
// CellMechanics — dispatches spatial hashing and force computation
// to Metal GPU compute shaders
// ─────────────────────────────────────────────────────────────────────

class CellMechanics {
public:
    CellMechanics();
    ~CellMechanics();

    void initialize(const SimConfig& config, MetalContext* ctx);

    // Run full mechanics step: hash → forces → integrate
    void update(CellData& cells, float dt);

    // Individual steps (for testing)
    void clearSpatialHash();
    void buildSpatialHash(CellData& cells);
    void computeForces(CellData& cells);
    void integratePositions(CellData& cells, float dt);

private:
    MechanicsParams params_;
    MetalContext* metal_ctx_;

    // Spatial hash buffers (Metal shared)
    void* hash_counts_buffer_;   // id<MTLBuffer> — uint32_t per mech voxel
    void* hash_cells_buffer_;    // id<MTLBuffer> — cell indices per voxel slot
    void* mech_params_buffer_;   // id<MTLBuffer>
    void* grid_params_buffer_;   // id<MTLBuffer> — domain origin for hash

    uint32_t total_mech_voxels_;
};

#endif
