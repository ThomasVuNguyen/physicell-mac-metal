// ─────────────────────────────────────────────────────────────────────
// diffusion_2d.metal
// 2D LOD (Locally One-Dimensional) diffusion-decay solver
// Thomas algorithm on tridiagonal system per row/column
//
// Decay is embedded into the tridiagonal diagonal via LOD splitting:
//   half_decay = 0.5 * lambda * dt
//   Interior diagonal:  1 + 2*coeff + half_decay
//   Boundary diagonal:  1 + coeff + half_decay
// This matches PhysiCell BioFVM_solvers.cpp LOD_2D exactly.
//
// Supports Dirichlet BCs: boundary voxels are fixed at target_value.
// ─────────────────────────────────────────────────────────────────────

#include "types.h"

// ─── Thomas algorithm workspace in threadgroup memory ───
// Maximum grid dimension we support in threadgroup memory.
#define MAX_LINE_LENGTH 2048

// ─────────────────────────────────────────────────────────────────────
// X-sweep: 1 thread per y-row
// Solves the tridiagonal system along x for every substrate in the row.
//
// For Dirichlet BCs: boundary nodes (i=0, i=nx-1) are fixed at target.
// The tridiagonal system is solved for interior nodes i=1..nx-2.
// ─────────────────────────────────────────────────────────────────────
kernel void diffusion_sweep_x(
    device float*               density         [[buffer(0)]],
    constant GridParams&        grid            [[buffer(1)]],
    constant SubstrateParams*   substrates      [[buffer(2)]],
    constant ThomasCoeffs*      coeffs          [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
)
{
    const uint j = tid;  // row index (y)
    if (j >= grid.ny) return;

    const uint nx = grid.nx;
    const uint n_voxels = grid.nx * grid.ny;

    for (uint s = 0; s < grid.n_substrates; s++) {
        const float coeff = coeffs[s].coeff;  // D * dt / (dx * dx)
        const float half_decay = coeffs[s].half_decay;  // 0.5 * lambda * dt

        // LOD tridiagonal diagonals (matching PhysiCell BioFVM_solvers.cpp)
        const float diag_interior = 1.0f + 2.0f * coeff + half_decay;  // thomas_constant3
        const float diag_boundary = 1.0f + coeff + half_decay;          // thomas_constant3a

        const uint base = s * n_voxels;
        const bool dirichlet = substrates[s].dirichlet_enabled != 0;
        const float target   = substrates[s].target_value;

        // Set boundary values if Dirichlet
        if (dirichlet) {
            density[base + j * nx + 0] = target;
            density[base + j * nx + (nx - 1)] = target;
        }

        if (nx <= 2) {
            // Only boundary nodes, nothing to solve
            continue;
        }

        // Special case: single interior node (nx == 3)
        if (nx == 3) {
            uint idx = base + j * nx + 1;
            float d_i = density[idx];

            // Both neighbors are boundary nodes
            float b = diag_interior;
            if (dirichlet) {
                d_i += coeff * target;  // left boundary contribution
                d_i += coeff * target;  // right boundary contribution
            }
            density[idx] = d_i / b;

            // Re-apply Dirichlet
            if (dirichlet) {
                density[base + j * nx + 0] = target;
                density[base + j * nx + (nx - 1)] = target;
            }
            continue;
        }

        // ─── Thomas algorithm for interior nodes i=1..nx-2 ───
        float c_prime[MAX_LINE_LENGTH];

        // i = 1 (first interior node, adjacent to left boundary)
        {
            uint idx = base + j * nx + 1;
            float d_i = density[idx];
            float b_1 = diag_boundary;  // boundary node: only one interior neighbor to the right
            float c_1 = -coeff;

            // If left boundary is Dirichlet, fold known value into RHS
            if (dirichlet) {
                d_i += coeff * target;  // a_1 * target moves to RHS
            }

            float c_p = c_1 / b_1;
            float d_p = d_i / b_1;

            c_prime[1] = c_p;
            density[idx] = d_p;

            float c_prev = c_p;
            float d_prev = d_p;

            // i = 2 .. nx-3 (inner interior nodes)
            for (uint i = 2; i < nx - 2; i++) {
                idx = base + j * nx + i;
                float a_i = -coeff;
                float b_i = diag_interior;
                float c_i = -coeff;
                d_i = density[idx];

                float denom = b_i - a_i * c_prev;
                c_prev = c_i / denom;
                d_prev = (d_i - a_i * d_prev) / denom;

                c_prime[i] = c_prev;
                density[idx] = d_prev;
            }

            // i = nx-2 (last interior node, adjacent to right boundary)
            {
                uint last_i = nx - 2;
                idx = base + j * nx + last_i;
                float a_last = -coeff;
                float b_last = diag_boundary;  // boundary node: only one interior neighbor to the left
                d_i = density[idx];

                // If right boundary is Dirichlet, fold known value into RHS
                if (dirichlet) {
                    d_i += coeff * target;
                }

                float denom = b_last - a_last * c_prev;
                d_prev = (d_i - a_last * d_prev) / denom;
                density[idx] = d_prev;
            }

            // ─── Back substitution for i = nx-3 down to 1 ───
            for (int i = (int)nx - 3; i >= 1; i--) {
                idx = base + j * nx + (uint)i;
                density[idx] = density[idx] - c_prime[i] * density[idx + 1];
            }
        }

        // Re-apply Dirichlet after solve (ensure boundary values are exact)
        if (dirichlet) {
            density[base + j * nx + 0] = target;
            density[base + j * nx + (nx - 1)] = target;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────
// Y-sweep: 1 thread per x-column
// Solves the tridiagonal system along y for every substrate in the column.
// Same LOD decay embedding as X-sweep.
// ─────────────────────────────────────────────────────────────────────
kernel void diffusion_sweep_y(
    device float*               density         [[buffer(0)]],
    constant GridParams&        grid            [[buffer(1)]],
    constant SubstrateParams*   substrates      [[buffer(2)]],
    constant ThomasCoeffs*      coeffs          [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
)
{
    const uint i = tid;  // column index (x)
    if (i >= grid.nx) return;

    const uint nx = grid.nx;
    const uint ny = grid.ny;
    const uint n_voxels = nx * ny;

    for (uint s = 0; s < grid.n_substrates; s++) {
        float D = substrates[s].diffusion_coefficient;
        float dt = grid.dt;
        float dy = grid.dy;
        float coeff = D * dt / (dy * dy);
        const float half_decay = coeffs[s].half_decay;  // 0.5 * lambda * dt

        // LOD tridiagonal diagonals (same formulation as X-sweep)
        const float diag_interior = 1.0f + 2.0f * coeff + half_decay;
        const float diag_boundary = 1.0f + coeff + half_decay;

        const uint base = s * n_voxels;
        const bool dirichlet = substrates[s].dirichlet_enabled != 0;
        const float target   = substrates[s].target_value;

        // Set boundary values
        if (dirichlet) {
            density[base + 0 * nx + i] = target;
            density[base + (ny - 1) * nx + i] = target;
        }

        if (ny <= 2) continue;

        // Special case: single interior node (ny == 3)
        if (ny == 3) {
            uint idx = base + 1 * nx + i;
            float d_j = density[idx];

            float b = diag_interior;
            if (dirichlet) {
                d_j += coeff * target;
                d_j += coeff * target;
            }
            density[idx] = d_j / b;

            if (dirichlet) {
                density[base + 0 * nx + i] = target;
                density[base + (ny - 1) * nx + i] = target;
            }
            continue;
        }

        // ─── Thomas algorithm for interior nodes j=1..ny-2 ───
        float c_prime[MAX_LINE_LENGTH];

        // j = 1 (first interior node, adjacent to top boundary)
        {
            uint idx = base + 1 * nx + i;
            float d_j = density[idx];
            float b_1 = diag_boundary;
            float c_1 = -coeff;

            if (dirichlet) {
                d_j += coeff * target;
            }

            float c_p = c_1 / b_1;
            float d_p = d_j / b_1;

            c_prime[1] = c_p;
            density[idx] = d_p;

            float c_prev = c_p;
            float d_prev = d_p;

            // j = 2 .. ny-3
            for (uint j = 2; j < ny - 2; j++) {
                idx = base + j * nx + i;
                float a_j = -coeff;
                float b_j = diag_interior;
                float c_j = -coeff;
                d_j = density[idx];

                float denom = b_j - a_j * c_prev;
                c_prev = c_j / denom;
                d_prev = (d_j - a_j * d_prev) / denom;

                c_prime[j] = c_prev;
                density[idx] = d_prev;
            }

            // j = ny-2 (last interior node, adjacent to bottom boundary)
            {
                uint last_j = ny - 2;
                idx = base + last_j * nx + i;
                float a_last = -coeff;
                float b_last = diag_boundary;
                d_j = density[idx];

                if (dirichlet) {
                    d_j += coeff * target;
                }

                float denom = b_last - a_last * c_prev;
                d_prev = (d_j - a_last * d_prev) / denom;
                density[idx] = d_prev;
            }

            // ─── Back substitution for j = ny-3 down to 1 ───
            for (int j = (int)ny - 3; j >= 1; j--) {
                idx = base + (uint)j * nx + i;
                uint idx_next = base + ((uint)j + 1) * nx + i;
                density[idx] = density[idx] - c_prime[j] * density[idx_next];
            }
        }

        // Re-apply Dirichlet after solve
        if (dirichlet) {
            density[base + 0 * nx + i] = target;
            density[base + (ny - 1) * nx + i] = target;
        }
    }
}
