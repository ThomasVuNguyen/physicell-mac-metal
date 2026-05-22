// ─────────────────────────────────────────────────────────────────────
// Motility — persistent random walk with chemotaxis bias
//
// Port of PhysiCell Cell::update_motility_vector() to SoA layout.
// CPU-only (complex branching, RNG, gradient sampling).
// ─────────────────────────────────────────────────────────────────────

#include "motility.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>

// ─── Thread-local RNG ────────────────────────────────────────────────

static thread_local std::mt19937 rng{std::random_device{}()};

static double uniformRandom() {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// ─── Random direction helpers ────────────────────────────────────────

static void uniformOnUnitCircle(double& x, double& y, double& z) {
    double theta = 2.0 * M_PI * uniformRandom();
    x = cos(theta);
    y = sin(theta);
    z = 0.0;
}

static void uniformOnUnitSphere(double& x, double& y, double& z) {
    double theta = 2.0 * M_PI * uniformRandom();
    double phi = M_PI * uniformRandom();
    double sin_phi = sin(phi);
    x = sin_phi * cos(theta);
    y = sin_phi * sin(theta);
    z = cos(phi);
}

// ─── Gradient computation ────────────────────────────────────────────
// Finite-difference gradient of density at cell position.
// Uses central differences when possible, forward/backward at boundaries.

static void computeGradient(const float* density, const GridParams& grid,
                            float px, float py, float pz,
                            int substrate_index,
                            double& gx, double& gy, double& gz)
{
    uint32_t nx = grid.nx;
    uint32_t ny = grid.ny;
    uint32_t nz = grid.nz;
    uint32_t n_sub = grid.n_substrates;
    float dx = grid.dx;
    float dy = grid.dy;
    float dz = grid.dz;

    // Find voxel indices
    int ix = static_cast<int>((px - grid.x_min) / dx);
    int iy = static_cast<int>((py - grid.y_min) / dy);
    int iz = static_cast<int>((pz - grid.z_min) / dz);

    // Clamp
    ix = std::max(0, std::min(ix, static_cast<int>(nx) - 1));
    iy = std::max(0, std::min(iy, static_cast<int>(ny) - 1));
    iz = std::max(0, std::min(iz, static_cast<int>(nz) - 1));

    // Lambda to read density at (i,j,k) for the given substrate
    auto D = [&](int i, int j, int k) -> double {
        uint32_t vi = static_cast<uint32_t>(std::max(0, std::min(i, static_cast<int>(nx)-1)));
        uint32_t vj = static_cast<uint32_t>(std::max(0, std::min(j, static_cast<int>(ny)-1)));
        uint32_t vk = static_cast<uint32_t>(std::max(0, std::min(k, static_cast<int>(nz)-1)));
        uint32_t voxel = vk * ny * nx + vj * nx + vi;
        return static_cast<double>(density[voxel * n_sub + substrate_index]);
    };

    // Central differences (with boundary clamping handled by D)
    gx = (D(ix+1, iy, iz) - D(ix-1, iy, iz)) / (2.0 * dx);
    gy = (D(ix, iy+1, iz) - D(ix, iy-1, iz)) / (2.0 * dy);
    gz = 0.0;
    if (nz > 1) {
        gz = (D(ix, iy, iz+1) - D(ix, iy, iz-1)) / (2.0 * dz);
    }
}

// ─── MotilityData implementation ─────────────────────────────────────

void MotilityData::allocate(uint32_t max_c) {
    max_cells = max_c;
    num_cells = 0;
    n_substrates = 0;
    size_t n = max_c;

    persistence_time   = static_cast<double*>(calloc(n, sizeof(double)));
    motility_elapsed   = static_cast<double*>(calloc(n, sizeof(double)));
    motility_bias      = static_cast<double*>(calloc(n, sizeof(double)));
    motility_vec_x     = static_cast<double*>(calloc(n, sizeof(double)));
    motility_vec_y     = static_cast<double*>(calloc(n, sizeof(double)));
    motility_vec_z     = static_cast<double*>(calloc(n, sizeof(double)));
    is_motile          = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    restrict_to_2D     = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    chemotaxis_index   = static_cast<int32_t*>(calloc(n, sizeof(int32_t)));
    chemotaxis_direction = static_cast<int32_t*>(calloc(n, sizeof(int32_t)));
    chemotactic_sensitivity = nullptr;  // allocated separately via allocateSubstrate()
}

void MotilityData::allocateSubstrate(uint32_t nsubs) {
    if (chemotactic_sensitivity) {
        free(chemotactic_sensitivity);
    }
    n_substrates = nsubs;
    size_t total = static_cast<size_t>(max_cells) * nsubs;
    chemotactic_sensitivity = static_cast<double*>(calloc(total, sizeof(double)));
}

void MotilityData::free_all() {
    free(persistence_time);   persistence_time = nullptr;
    free(motility_elapsed);   motility_elapsed = nullptr;
    free(motility_bias);      motility_bias = nullptr;
    free(motility_vec_x);     motility_vec_x = nullptr;
    free(motility_vec_y);     motility_vec_y = nullptr;
    free(motility_vec_z);     motility_vec_z = nullptr;
    free(is_motile);          is_motile = nullptr;
    free(restrict_to_2D);     restrict_to_2D = nullptr;
    free(chemotaxis_index);   chemotaxis_index = nullptr;
    free(chemotaxis_direction); chemotaxis_direction = nullptr;
    free(chemotactic_sensitivity); chemotactic_sensitivity = nullptr;
    n_substrates = 0;
}

void MotilityData::setDefaults(uint32_t index) {
    assert(index < max_cells);
    persistence_time[index]   = 15.0;  // 15 min default (PhysiCell)
    motility_elapsed[index]   = 0.0;
    motility_bias[index]      = 0.5;
    motility_vec_x[index]     = 0.0;
    motility_vec_y[index]     = 0.0;
    motility_vec_z[index]     = 0.0;
    is_motile[index]          = 0;     // disabled by default
    restrict_to_2D[index]     = 1;     // 2D default
    chemotaxis_index[index]   = 0;     // first substrate
    chemotaxis_direction[index] = 1;   // up gradient

    // Zero out all per-substrate chemotactic sensitivities
    if (chemotactic_sensitivity && n_substrates > 0) {
        for (uint32_t s = 0; s < n_substrates; s++) {
            chemotactic_sensitivity[static_cast<size_t>(index) * n_substrates + s] = 0.0;
        }
    }
}

void MotilityData::swapRemove(uint32_t index, uint32_t last) {
    if (index == last) return;
    persistence_time[index]     = persistence_time[last];
    motility_elapsed[index]     = motility_elapsed[last];
    motility_bias[index]        = motility_bias[last];
    motility_vec_x[index]       = motility_vec_x[last];
    motility_vec_y[index]       = motility_vec_y[last];
    motility_vec_z[index]       = motility_vec_z[last];
    is_motile[index]            = is_motile[last];
    restrict_to_2D[index]       = restrict_to_2D[last];
    chemotaxis_index[index]     = chemotaxis_index[last];
    chemotaxis_direction[index] = chemotaxis_direction[last];

    // Copy per-substrate chemotactic sensitivities
    if (chemotactic_sensitivity && n_substrates > 0) {
        for (uint32_t s = 0; s < n_substrates; s++) {
            size_t off_i = static_cast<size_t>(index) * n_substrates + s;
            size_t off_l = static_cast<size_t>(last)  * n_substrates + s;
            chemotactic_sensitivity[off_i] = chemotactic_sensitivity[off_l];
        }
    }
}

// ─── updateMotility ──────────────────────────────────────────────────

void updateMotility(CellData& cells, MotilityData& mot,
                    const float* density, const GridParams& grid,
                    double dt)
{
    uint32_t n = cells.num_cells;
    mot.num_cells = n;

    for (uint32_t i = 0; i < n; i++) {
        if (cells.is_alive[i] == 0) continue;

        // Non-motile cells: zero motility vector, skip
        if (mot.is_motile[i] == 0) {
            mot.motility_vec_x[i] = 0.0;
            mot.motility_vec_y[i] = 0.0;
            mot.motility_vec_z[i] = 0.0;
            continue;
        }

        // Accumulate elapsed time
        mot.motility_elapsed[i] += dt;

        // Check if it's time to resample direction
        // PhysiCell: if( UniformRandom() < dt / persistence_time || persistence_time < dt )
        double pt = mot.persistence_time[i];
        bool resample = (pt < dt) || (uniformRandom() < dt / pt);

        if (resample) {
            mot.motility_elapsed[i] = 0.0;

            // Generate random unit vector
            double rx, ry, rz;
            if (mot.restrict_to_2D[i]) {
                uniformOnUnitCircle(rx, ry, rz);
            } else {
                uniformOnUnitSphere(rx, ry, rz);
            }

            // Compute bias direction from multi-substrate chemotaxis.
            // bias_direction = sum_s(sensitivity[i*n_subs+s] * gradient[s])
            // Falls back to single-substrate if chemotactic_sensitivity is not allocated.
            double bx = 0.0, by = 0.0, bz = 0.0;
            double bias = mot.motility_bias[i];

            if (bias > 0.0 && density != nullptr) {
                if (mot.chemotactic_sensitivity && mot.n_substrates > 0) {
                    // Multi-substrate weighted gradient sum
                    for (uint32_t s = 0; s < mot.n_substrates; s++) {
                        double sens = mot.chemotactic_sensitivity[
                            static_cast<size_t>(i) * mot.n_substrates + s];
                        if (std::fabs(sens) < 1e-16) continue;

                        double gx, gy, gz;
                        computeGradient(density, grid,
                                        cells.position_x[i],
                                        cells.position_y[i],
                                        cells.position_z[i],
                                        static_cast<int>(s), gx, gy, gz);
                        bx += sens * gx;
                        by += sens * gy;
                        bz += sens * gz;
                    }
                } else {
                    // Legacy single-substrate chemotaxis
                    int sub_idx = mot.chemotaxis_index[i];
                    int direction = mot.chemotaxis_direction[i];

                    double gx, gy, gz;
                    computeGradient(density, grid,
                                    cells.position_x[i],
                                    cells.position_y[i],
                                    cells.position_z[i],
                                    sub_idx, gx, gy, gz);
                    bx = direction * gx;
                    by = direction * gy;
                    bz = direction * gz;
                }

                // Normalize bias direction vector
                double bmag = sqrt(bx*bx + by*by + bz*bz);
                if (bmag > 1e-16) {
                    bx /= bmag;
                    by /= bmag;
                    bz /= bmag;
                }
            }

            // Combine: motility = (1-bias)*random + bias*bias_direction
            // Then normalize, then scale by migration speed
            double one_minus_bias = 1.0 - bias;
            double mx = one_minus_bias * rx + bias * bx;
            double my = one_minus_bias * ry + bias * by;
            double mz = one_minus_bias * rz + bias * bz;

            // Normalize
            double mag = sqrt(mx*mx + my*my + mz*mz);
            if (mag > 1e-16) {
                mx /= mag;
                my /= mag;
                mz /= mag;
            }

            // Scale by migration speed
            double speed = static_cast<double>(cells.motility_speed[i]);
            mot.motility_vec_x[i] = mx * speed;
            mot.motility_vec_y[i] = my * speed;
            mot.motility_vec_z[i] = mz * speed;
        }

        // Add motility vector to cell velocity
        // (Velocity was set to mechanical forces by GPU; we ADD motility on top)
        cells.velocity_x[i] += static_cast<float>(mot.motility_vec_x[i]);
        cells.velocity_y[i] += static_cast<float>(mot.motility_vec_y[i]);
        cells.velocity_z[i] += static_cast<float>(mot.motility_vec_z[i]);
    }
}
