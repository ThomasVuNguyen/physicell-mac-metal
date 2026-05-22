// ─────────────────────────────────────────────────────────────────────
// integrate.metal
// Position integration using Adams-Bashforth 2nd order method
// ─────────────────────────────────────────────────────────────────────

#include "types.h"

// ─── SoA field offsets from shared types.h ───
// Use the CELL_FIELD_* defines — single source of truth.

// Helper to read/write from SoA cell buffer (guarded — may already be defined in mechanics.metal)
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
// Adams-Bashforth 2nd order position integration
// pos += dt * (1.5 * vel - 0.5 * prev_vel)
// Then: prev_vel = vel
// Finally: clamp to domain bounds
//
// 1 thread per cell
// ─────────────────────────────────────────────────────────────────────
kernel void integrate_positions(
    device float*           cells           [[buffer(0)]],
    constant GridParams&    grid            [[buffer(1)]],
    constant float&         dt              [[buffer(2)]],
    constant uint&          num_cells       [[buffer(3)]],
    constant uint&          max_cells       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= num_cells) return;

    // Read current position
    float px = cell_read(cells, CELL_FIELD_POS_X, tid, max_cells);
    float py = cell_read(cells, CELL_FIELD_POS_Y, tid, max_cells);
    float pz = cell_read(cells, CELL_FIELD_POS_Z, tid, max_cells);

    // Read current velocity
    float vx = cell_read(cells, CELL_FIELD_VEL_X, tid, max_cells);
    float vy = cell_read(cells, CELL_FIELD_VEL_Y, tid, max_cells);
    float vz = cell_read(cells, CELL_FIELD_VEL_Z, tid, max_cells);

    // Read previous velocity
    float pvx = cell_read(cells, CELL_FIELD_PREV_VX, tid, max_cells);
    float pvy = cell_read(cells, CELL_FIELD_PREV_VY, tid, max_cells);
    float pvz = cell_read(cells, CELL_FIELD_PREV_VZ, tid, max_cells);

    // Adams-Bashforth 2nd order
    px += dt * (1.5f * vx - 0.5f * pvx);
    py += dt * (1.5f * vy - 0.5f * pvy);
    pz += dt * (1.5f * vz - 0.5f * pvz);

    // Domain bounds: [x_min, x_min + nx*dx] etc.
    float x_max = grid.x_min + (float)grid.nx * grid.dx;
    float y_max = grid.y_min + (float)grid.ny * grid.dy;
    float z_max = grid.z_min + (float)grid.nz * grid.dz;

    px = clamp(px, grid.x_min, x_max);
    py = clamp(py, grid.y_min, y_max);
    pz = clamp(pz, grid.z_min, z_max);

    // Write updated position
    cell_write(cells, CELL_FIELD_POS_X, tid, max_cells, px);
    cell_write(cells, CELL_FIELD_POS_Y, tid, max_cells, py);
    cell_write(cells, CELL_FIELD_POS_Z, tid, max_cells, pz);

    // Store current velocity as previous for next step
    cell_write(cells, CELL_FIELD_PREV_VX, tid, max_cells, vx);
    cell_write(cells, CELL_FIELD_PREV_VY, tid, max_cells, vy);
    cell_write(cells, CELL_FIELD_PREV_VZ, tid, max_cells, vz);
}
