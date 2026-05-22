#import "microenvironment.h"
#import "metal_context.h"
#import <Metal/Metal.h>
#import <cmath>
#import <cstring>
#import <cstdlib>

// ─────────────────────────────────────────────────────────────────────
// Microenvironment — GPU-accelerated diffusion-decay solver
// ─────────────────────────────────────────────────────────────────────

Microenvironment::Microenvironment()
    : n_substrates_(0), density_data_(nullptr), substrate_params_(nullptr),
      gradient_x_(nullptr), gradient_y_(nullptr), gradient_z_(nullptr),
      density_buffer_(nullptr), grid_params_buffer_(nullptr),
      substrate_params_buffer_(nullptr), thomas_coeffs_buffer_(nullptr),
      metal_ctx_(nullptr) {}

Microenvironment::~Microenvironment() {
    if (density_buffer_) CFRelease(density_buffer_);
    if (grid_params_buffer_) CFRelease(grid_params_buffer_);
    if (substrate_params_buffer_) CFRelease(substrate_params_buffer_);
    if (thomas_coeffs_buffer_) CFRelease(thomas_coeffs_buffer_);
    free(gradient_x_);
    free(gradient_y_);
    free(gradient_z_);
}

void Microenvironment::initialize(const SimConfig& config, MetalContext* ctx) {
    metal_ctx_ = ctx;

    // Set up grid from config domain bounds
    grid_.nx = (uint32_t)((config.x_max - config.x_min) / config.grid.dx);
    grid_.ny = (uint32_t)((config.y_max - config.y_min) / config.grid.dy);
    grid_.nz = config.use_2D ? 1 : (uint32_t)((config.z_max - config.z_min) / config.grid.dz);
    grid_.dx = config.grid.dx;
    grid_.dy = config.grid.dy;
    grid_.dz = config.grid.dz;
    grid_.x_min = config.x_min;
    grid_.y_min = config.y_min;
    grid_.z_min = config.z_min;
    grid_.dt = config.dt_diffusion;
    n_substrates_ = (uint32_t)config.substrates.size();
    grid_.n_substrates = n_substrates_;

    uint32_t total_voxels = grid_.nx * grid_.ny * grid_.nz;

    // Allocate density buffer in Metal shared memory
    size_t density_bytes = total_voxels * n_substrates_ * sizeof(float);
    id<MTLBuffer> dBuf = ctx->allocateBuffer(density_bytes);
    density_buffer_ = (__bridge_retained void*)dBuf;
    density_data_ = (float*)[dBuf contents];

    // Initialize densities
    for (uint32_t s = 0; s < n_substrates_; s++) {
        float init_val = config.substrates[s].initial_value;
        for (uint32_t v = 0; v < total_voxels; v++) {
            density_data_[s * total_voxels + v] = init_val;
        }
    }

    // Grid params buffer
    id<MTLBuffer> gpBuf = ctx->allocateBuffer(sizeof(GridParams));
    grid_params_buffer_ = (__bridge_retained void*)gpBuf;
    memcpy([gpBuf contents], &grid_, sizeof(GridParams));

    // Substrate params buffer
    size_t sp_bytes = n_substrates_ * sizeof(SubstrateParams);
    id<MTLBuffer> spBuf = ctx->allocateBuffer(sp_bytes);
    substrate_params_buffer_ = (__bridge_retained void*)spBuf;
    SubstrateParams* sp = (SubstrateParams*)[spBuf contents];
    for (uint32_t s = 0; s < n_substrates_; s++) {
        sp[s].diffusion_coefficient = config.substrates[s].diffusion_coeff;
        sp[s].decay_rate = config.substrates[s].decay_rate;
        sp[s].supply_rate = 0.0f;
        sp[s].target_value = config.substrates[s].dirichlet_value;
        sp[s].dirichlet_enabled = config.substrates[s].dirichlet_enabled ? 1 : 0;
    }
    substrate_params_ = sp;

    // Thomas coefficients buffer (precomputed per substrate)
    size_t tc_bytes = n_substrates_ * sizeof(ThomasCoeffs);
    id<MTLBuffer> tcBuf = ctx->allocateBuffer(tc_bytes);
    thomas_coeffs_buffer_ = (__bridge_retained void*)tcBuf;
    ThomasCoeffs* tc = (ThomasCoeffs*)[tcBuf contents];
    for (uint32_t s = 0; s < n_substrates_; s++) {
        float D = config.substrates[s].diffusion_coeff;
        float lambda = config.substrates[s].decay_rate;
        float dt = config.dt_diffusion;
        tc[s].coeff = D * dt / (grid_.dx * grid_.dx);  // note: will be recalculated per-axis in shader
        tc[s].decay_coeff = 1.0f / (1.0f + lambda * dt);
    }

    printf("  Microenvironment: %u x %u x %u voxels, %u substrates\n",
           grid_.nx, grid_.ny, grid_.nz, n_substrates_);
    printf("  Density buffer: %.2f KB (Metal shared)\n", density_bytes / 1024.0);

    // Allocate gradient buffers (CPU-only, substrate-major layout)
    size_t grad_count = static_cast<size_t>(total_voxels) * n_substrates_;
    gradient_x_ = static_cast<float*>(calloc(grad_count, sizeof(float)));
    gradient_y_ = static_cast<float*>(calloc(grad_count, sizeof(float)));
    gradient_z_ = static_cast<float*>(calloc(grad_count, sizeof(float)));
    printf("  Gradient buffers: %.2f KB (CPU)\n", (3 * grad_count * sizeof(float)) / 1024.0);
}

void Microenvironment::simulateDiffusionDecay() {
    id<MTLBuffer> dBuf = (__bridge id<MTLBuffer>)density_buffer_;
    id<MTLBuffer> gpBuf = (__bridge id<MTLBuffer>)grid_params_buffer_;
    id<MTLBuffer> spBuf = (__bridge id<MTLBuffer>)substrate_params_buffer_;
    id<MTLBuffer> tcBuf = (__bridge id<MTLBuffer>)thomas_coeffs_buffer_;

    // Dispatch X-sweep then Y-sweep (matching MetalContext API)
    metal_ctx_->dispatchDiffusionX(dBuf, gpBuf, spBuf, tcBuf, grid_.ny);
    metal_ctx_->dispatchDiffusionY(dBuf, gpBuf, spBuf, tcBuf, grid_.nx);
}

// ─────────────────────────────────────────────────────────────────────
// Dirichlet boundary condition enforcement
//
// After each diffusion step, reset boundary voxels to their target values.
// PhysiCell applies Dirichlet BCs on all 4 edges of the 2D domain
// (or 6 faces in 3D) for each substrate where dirichlet_enabled == true.
//
// The density layout is substrate-major:
//   density[s * total_voxels + (j * nx + i)]
// ─────────────────────────────────────────────────────────────────────
void Microenvironment::applyDirichletBC() {
    if (!density_data_ || !substrate_params_) return;

    const uint32_t nx = grid_.nx;
    const uint32_t ny = grid_.ny;
    const uint32_t total_voxels = nx * ny * grid_.nz;

    for (uint32_t s = 0; s < n_substrates_; s++) {
        if (!substrate_params_[s].dirichlet_enabled) continue;

        float target = substrate_params_[s].target_value;
        uint32_t base = s * total_voxels;

        // Top and bottom edges (y = 0 and y = ny-1)
        for (uint32_t i = 0; i < nx; i++) {
            density_data_[base + 0 * nx + i] = target;         // y = 0
            density_data_[base + (ny - 1) * nx + i] = target;  // y = ny-1
        }

        // Left and right edges (x = 0 and x = nx-1)
        for (uint32_t j = 0; j < ny; j++) {
            density_data_[base + j * nx + 0] = target;         // x = 0
            density_data_[base + j * nx + (nx - 1)] = target;  // x = nx-1
        }
    }
}

float Microenvironment::getDensity(uint32_t voxel_index, uint32_t substrate_index) const {
    uint32_t total_voxels = grid_.nx * grid_.ny * grid_.nz;
    return density_data_[substrate_index * total_voxels + voxel_index];
}

void Microenvironment::setDensity(uint32_t voxel_index, uint32_t substrate_index, float value) {
    uint32_t total_voxels = grid_.nx * grid_.ny * grid_.nz;
    density_data_[substrate_index * total_voxels + voxel_index] = value;
}

void Microenvironment::applySecretion(uint32_t voxel_index, uint32_t substrate_index,
                                       float secretion_rate, float uptake_rate,
                                       float saturation_density, float dt) {
    uint32_t total_voxels = grid_.nx * grid_.ny * grid_.nz;
    uint32_t idx = substrate_index * total_voxels + voxel_index;

    float rho = density_data_[idx];
    float denom = 1.0f + dt * (secretion_rate + uptake_rate);
    rho = (rho + dt * secretion_rate * saturation_density) / denom;
    density_data_[idx] = rho;
}

uint32_t Microenvironment::voxelIndex(float x, float y, float z) const {
    int i = (int)((x - grid_.x_min) / grid_.dx);
    int j = (int)((y - grid_.y_min) / grid_.dy);
    int k = (int)((z - grid_.z_min) / grid_.dz);
    i = (i < 0) ? 0 : (i >= (int)grid_.nx ? (int)grid_.nx - 1 : i);
    j = (j < 0) ? 0 : (j >= (int)grid_.ny ? (int)grid_.ny - 1 : j);
    k = (k < 0) ? 0 : (k >= (int)grid_.nz ? (int)grid_.nz - 1 : k);
    return k * grid_.nx * grid_.ny + j * grid_.nx + i;
}

// ─────────────────────────────────────────────────────────────────────
// Gradient computation — central differences on the density grid
//
// Matches PhysiCell BioFVM_microenvironment.cpp::compute_gradient_vector.
// Layout: density is substrate-major: density[s * total_voxels + (k*ny*nx + j*nx + i)]
//         gradient buffers use the same layout.
//
// At interior voxels: central difference  (D[i+1] - D[i-1]) / (2*dx)
// At boundaries: one-sided difference     (D[i+1] - D[i]) / dx  or  (D[i] - D[i-1]) / dx
// ─────────────────────────────────────────────────────────────────────

void Microenvironment::computeGradients() {
    if (!density_data_ || !gradient_x_ || n_substrates_ == 0) return;

    const uint32_t nx = grid_.nx;
    const uint32_t ny = grid_.ny;
    const uint32_t nz = grid_.nz;
    const uint32_t total_voxels = nx * ny * nz;
    const float inv_2dx = 1.0f / (2.0f * grid_.dx);
    const float inv_2dy = 1.0f / (2.0f * grid_.dy);
    const float inv_2dz = (nz > 1) ? 1.0f / (2.0f * grid_.dz) : 0.0f;
    const float inv_dx = 1.0f / grid_.dx;
    const float inv_dy = 1.0f / grid_.dy;
    const float inv_dz = (nz > 1) ? 1.0f / grid_.dz : 0.0f;

    for (uint32_t s = 0; s < n_substrates_; s++) {
        const uint32_t base = s * total_voxels;

        for (uint32_t k = 0; k < nz; k++) {
            for (uint32_t j = 0; j < ny; j++) {
                for (uint32_t i = 0; i < nx; i++) {
                    uint32_t n = base + k * ny * nx + j * nx + i;

                    // ─── d/dx ───
                    if (i == 0) {
                        // Forward difference at left boundary
                        gradient_x_[n] = (density_data_[n + 1] - density_data_[n]) * inv_dx;
                    } else if (i == nx - 1) {
                        // Backward difference at right boundary
                        gradient_x_[n] = (density_data_[n] - density_data_[n - 1]) * inv_dx;
                    } else {
                        // Central difference
                        gradient_x_[n] = (density_data_[n + 1] - density_data_[n - 1]) * inv_2dx;
                    }

                    // ─── d/dy ───
                    if (j == 0) {
                        gradient_y_[n] = (density_data_[n + nx] - density_data_[n]) * inv_dy;
                    } else if (j == ny - 1) {
                        gradient_y_[n] = (density_data_[n] - density_data_[n - nx]) * inv_dy;
                    } else {
                        gradient_y_[n] = (density_data_[n + nx] - density_data_[n - nx]) * inv_2dy;
                    }

                    // ─── d/dz ───
                    if (nz > 1) {
                        uint32_t stride_z = ny * nx;
                        if (k == 0) {
                            gradient_z_[n] = (density_data_[n + stride_z] - density_data_[n]) * inv_dz;
                        } else if (k == nz - 1) {
                            gradient_z_[n] = (density_data_[n] - density_data_[n - stride_z]) * inv_dz;
                        } else {
                            gradient_z_[n] = (density_data_[n + stride_z] - density_data_[n - stride_z]) * inv_2dz;
                        }
                    } else {
                        gradient_z_[n] = 0.0f;
                    }
                }
            }
        }
    }
}

void Microenvironment::getGradient(uint32_t voxel_index, uint32_t substrate_index,
                                   float& gx, float& gy, float& gz) const {
    uint32_t total_voxels = grid_.nx * grid_.ny * grid_.nz;
    uint32_t idx = substrate_index * total_voxels + voxel_index;
    gx = gradient_x_ ? gradient_x_[idx] : 0.0f;
    gy = gradient_y_ ? gradient_y_[idx] : 0.0f;
    gz = gradient_z_ ? gradient_z_[idx] : 0.0f;
}
