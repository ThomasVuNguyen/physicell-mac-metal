// ─────────────────────────────────────────────────────────────────────
// mechanics.metal
// Spatial hashing and cell-cell force computation
// ─────────────────────────────────────────────────────────────────────

#include "types.h"

// ─── SoA field offsets from shared types.h ───
// Use the CELL_FIELD_* defines from types.h — single source of truth.

// Helper to read/write from SoA cell buffer
#ifndef CELL_SOA_HELPERS
#define CELL_SOA_HELPERS
inline float cell_read(device float* cells, uint field, uint cell_id, uint max_cells) {
    return cells[field * max_cells + cell_id];
}

inline void cell_write(device float* cells, uint field, uint cell_id, uint max_cells, float val) {
    cells[field * max_cells + cell_id] = val;
}
#endif

// ─────────────────────────────────────────────────────────────────────
// Clear spatial hash: 1 thread per mechanics voxel
// Zeros out the hash_counts array
// ─────────────────────────────────────────────────────────────────────
kernel void clear_spatial_hash(
    device atomic_uint*     hash_counts     [[buffer(0)]],
    constant uint&          num_voxels      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= num_voxels) return;
    atomic_store_explicit(&hash_counts[tid], 0, memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────
// Build spatial hash: 1 thread per cell
// Computes mechanics voxel from position, inserts into hash
// ─────────────────────────────────────────────────────────────────────
kernel void build_spatial_hash(
    device float*           cells           [[buffer(0)]],
    device atomic_uint*     hash_counts     [[buffer(1)]],
    device uint*            hash_cells      [[buffer(2)]],
    constant MechanicsParams& mech          [[buffer(3)]],
    constant GridParams&    grid            [[buffer(4)]],
    constant uint&          num_cells       [[buffer(5)]],
    constant uint&          max_cells       [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= num_cells) return;

    float px = cell_read(cells, CELL_FIELD_POS_X, tid, max_cells);
    float py = cell_read(cells, CELL_FIELD_POS_Y, tid, max_cells);
    float pz = cell_read(cells, CELL_FIELD_POS_Z, tid, max_cells);

    float voxel_size = mech.mechanics_voxel_size;

    // Compute mechanics voxel indices
    int vi = (int)((px - grid.x_min) / voxel_size);
    int vj = (int)((py - grid.y_min) / voxel_size);
    int vk = (int)((pz - grid.z_min) / voxel_size);

    // Clamp to grid bounds
    vi = clamp(vi, 0, (int)mech.mech_grid_nx - 1);
    vj = clamp(vj, 0, (int)mech.mech_grid_ny - 1);
    vk = clamp(vk, 0, (int)mech.mech_grid_nz - 1);

    uint voxel_index = (uint)vk * mech.mech_grid_ny * mech.mech_grid_nx
                     + (uint)vj * mech.mech_grid_nx
                     + (uint)vi;

    // Store the mechanics voxel index back into the cell data
    // Using as_type to store uint as float bits
    cell_write(cells, CELL_FIELD_MECH_VOXEL, tid, max_cells, as_type<float>(voxel_index));

    // Atomic increment to get a slot in the hash bucket
    uint slot = atomic_fetch_add_explicit(&hash_counts[voxel_index], 1, memory_order_relaxed);

    // Insert if there's room
    if (slot < MAX_CELLS_PER_VOXEL) {
        hash_cells[voxel_index * MAX_CELLS_PER_VOXEL + slot] = tid;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Compute forces: 1 thread per cell
// Moore neighborhood search with repulsion + adhesion
// ─────────────────────────────────────────────────────────────────────
kernel void compute_forces(
    device float*           cells           [[buffer(0)]],
    device atomic_uint*     hash_counts     [[buffer(1)]],
    device uint*            hash_cells      [[buffer(2)]],
    constant MechanicsParams& mech          [[buffer(3)]],
    constant uint&          num_cells       [[buffer(4)]],
    constant uint&          max_cells       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= num_cells) return;

    float px = cell_read(cells, CELL_FIELD_POS_X, tid, max_cells);
    float py = cell_read(cells, CELL_FIELD_POS_Y, tid, max_cells);
    float pz = cell_read(cells, CELL_FIELD_POS_Z, tid, max_cells);
    float radius_i = cell_read(cells, CELL_FIELD_RADIUS, tid, max_cells);
    float rep_i = cell_read(cells, CELL_FIELD_REPULSION, tid, max_cells);
    float adh_i = cell_read(cells, CELL_FIELD_ADHESION, tid, max_cells);
    float rel_max_adh = cell_read(cells, CELL_FIELD_REL_MAX_ADH, tid, max_cells);

    // Read mechanics voxel index
    uint my_voxel = as_type<uint>(cell_read(cells, CELL_FIELD_MECH_VOXEL, tid, max_cells));

    float max_dist = mech.max_interaction_distance;

    // Accumulate forces
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Accumulate simple_pressure (PhysiCell contact pressure)
    float local_pressure = 0.0f;

    // Decompose voxel index into (vi, vj, vk)
    uint mnx = mech.mech_grid_nx;
    uint mny = mech.mech_grid_ny;
    uint mnz = mech.mech_grid_nz;

    uint vi = my_voxel % mnx;
    uint vj = (my_voxel / mnx) % mny;
    uint vk = my_voxel / (mnx * mny);

    // Iterate over Moore neighborhood (3x3x3 in 3D, 3x3 in 2D when nz==1)
    int dk_min = (mnz > 1) ? -1 : 0;
    int dk_max = (mnz > 1) ?  1 : 0;

    for (int dk = dk_min; dk <= dk_max; dk++) {
        int nk = (int)vk + dk;
        if (nk < 0 || nk >= (int)mnz) continue;

        for (int dj = -1; dj <= 1; dj++) {
            int nj = (int)vj + dj;
            if (nj < 0 || nj >= (int)mny) continue;

            for (int di = -1; di <= 1; di++) {
                int ni = (int)vi + di;
                if (ni < 0 || ni >= (int)mnx) continue;

                uint neighbor_voxel = (uint)nk * mny * mnx + (uint)nj * mnx + (uint)ni;
                uint count = atomic_load_explicit(&hash_counts[neighbor_voxel], memory_order_relaxed);
                count = min(count, (uint)MAX_CELLS_PER_VOXEL);

                for (uint s = 0; s < count; s++) {
                    uint other = hash_cells[neighbor_voxel * MAX_CELLS_PER_VOXEL + s];
                    if (other == tid) continue;  // skip self

                    float ox = cell_read(cells, CELL_FIELD_POS_X, other, max_cells);
                    float oy = cell_read(cells, CELL_FIELD_POS_Y, other, max_cells);
                    float oz = cell_read(cells, CELL_FIELD_POS_Z, other, max_cells);

                    float ddx = px - ox;
                    float ddy = py - oy;
                    float ddz = pz - oz;
                    float dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if (dist_sq >= max_dist * max_dist) continue;
                    if (dist_sq < 1e-12f) continue;  // avoid division by zero

                    float dist = sqrt(dist_sq);
                    float inv_dist = 1.0f / dist;

                    // Direction from other to this cell
                    float nx = ddx * inv_dist;
                    float ny = ddy * inv_dist;
                    float nz = ddz * inv_dist;

                    float radius_j = cell_read(cells, CELL_FIELD_RADIUS, other, max_cells);
                    float R_sum = radius_i + radius_j;

                    float force_mag = 0.0f;

                    // ─── Repulsion: if overlapping (dist < R_sum) ───
                    if (dist < R_sum) {
                        float rep_j = cell_read(cells, CELL_FIELD_REPULSION, other, max_cells);
                        float overlap = 1.0f - dist / R_sum;
                        force_mag += overlap * overlap * sqrt(rep_i * rep_j);

                        // Accumulate contact pressure (PhysiCell simple_pressure)
                        // Scale factor = 1 / (0.5 * pi * R_sum^2) ≈ 0.027288820670331 for default R
                        float simple_pressure_scale = 0.027288820670331f;
                        local_pressure += overlap * overlap / simple_pressure_scale;
                    }

                    // ─── Adhesion: attractive force ───
                    float adhesion_distance = rel_max_adh * R_sum;
                    if (dist < adhesion_distance) {
                        float adh_j = cell_read(cells, CELL_FIELD_ADHESION, other, max_cells);
                        float t = 1.0f - dist / adhesion_distance;
                        force_mag -= t * t * sqrt(adh_i * adh_j);
                    }

                    fx += force_mag * nx;
                    fy += force_mag * ny;
                    fz += force_mag * nz;
                }
            }
        }
    }

    // Write accumulated force into velocity
    // (mechanics dt scaling is applied during integration)
    cell_write(cells, CELL_FIELD_VEL_X, tid, max_cells, fx);
    cell_write(cells, CELL_FIELD_VEL_Y, tid, max_cells, fy);
    cell_write(cells, CELL_FIELD_VEL_Z, tid, max_cells, fz);

    // Write accumulated simple_pressure
    cell_write(cells, CELL_FIELD_PRESSURE, tid, max_cells, local_pressure);
}
