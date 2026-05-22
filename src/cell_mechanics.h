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

    // Run full mechanics step: clear_hash → build_hash → forces → integrate
    // Uses a single Metal command buffer with memory barriers between kernels.
    void update(CellData& cells, float dt);

    // Accessors for spatial hash (CPU-side neighbor lookups in cell interactions)
    uint32_t totalMechVoxels() const { return total_mech_voxels_; }
    const MechanicsParams& params() const { return params_; }

    // ── CPU-side spatial hash access (reads GPU shared memory, zero-copy on UMA) ──

    // Compute mechanics voxel index from world position
    inline uint32_t voxelIndexFromPosition(float px, float py, float pz) const {
        int vi = (int)((px - grid_origin_x_) / params_.mechanics_voxel_size);
        int vj = (int)((py - grid_origin_y_) / params_.mechanics_voxel_size);
        int vk = (int)((pz - grid_origin_z_) / params_.mechanics_voxel_size);
        vi = std::max(0, std::min(vi, (int)params_.mech_grid_nx - 1));
        vj = std::max(0, std::min(vj, (int)params_.mech_grid_ny - 1));
        vk = std::max(0, std::min(vk, (int)params_.mech_grid_nz - 1));
        return (uint32_t)vk * params_.mech_grid_ny * params_.mech_grid_nx
             + (uint32_t)vj * params_.mech_grid_nx
             + (uint32_t)vi;
    }

    // Get count of cells in a mechanics voxel
    inline uint32_t voxelCellCount(uint32_t voxel_idx) const {
        if (!hash_counts_ptr_ || voxel_idx >= total_mech_voxels_) return 0;
        return hash_counts_ptr_[voxel_idx];
    }

    // Get pointer to cell indices in a mechanics voxel
    inline const uint32_t* voxelCells(uint32_t voxel_idx) const {
        if (!hash_cells_ptr_ || voxel_idx >= total_mech_voxels_) return nullptr;
        return &hash_cells_ptr_[(size_t)voxel_idx * MAX_CELLS_PER_VOXEL];
    }

    // Cache CPU pointers after GPU sync (call after Metal waitForCompletion)
    void syncHashPointers();

private:
    MechanicsParams params_;
    MetalContext* metal_ctx_;

    // Spatial hash buffers (Metal shared)
    void* hash_counts_buffer_;   // id<MTLBuffer> — uint32_t per mech voxel
    void* hash_cells_buffer_;    // id<MTLBuffer> — cell indices per voxel slot
    void* mech_params_buffer_;   // id<MTLBuffer>
    void* grid_params_buffer_;   // id<MTLBuffer> — domain origin for hash

    // CPU-readable pointers into shared Metal buffers (cached after sync)
    const uint32_t* hash_counts_ptr_ = nullptr;
    const uint32_t* hash_cells_ptr_ = nullptr;

    // Grid origin (cached from grid_params_buffer_)
    float grid_origin_x_ = 0.0f;
    float grid_origin_y_ = 0.0f;
    float grid_origin_z_ = 0.0f;

    uint32_t total_mech_voxels_;
};

#endif
