#import "cell_mechanics.h"
#import "metal_context.h"
#import <Metal/Metal.h>
#import <cmath>
#import <cstring>

// ─────────────────────────────────────────────────────────────────────
// CellMechanics — GPU-dispatched spatial hashing and force computation
// ─────────────────────────────────────────────────────────────────────

CellMechanics::CellMechanics()
    : metal_ctx_(nullptr), hash_counts_buffer_(nullptr),
      hash_cells_buffer_(nullptr), mech_params_buffer_(nullptr),
      grid_params_buffer_(nullptr), total_mech_voxels_(0) {}

CellMechanics::~CellMechanics() {
    if (hash_counts_buffer_) CFRelease(hash_counts_buffer_);
    if (hash_cells_buffer_) CFRelease(hash_cells_buffer_);
    if (mech_params_buffer_) CFRelease(mech_params_buffer_);
    if (grid_params_buffer_) CFRelease(grid_params_buffer_);
}

void CellMechanics::initialize(const SimConfig& config, MetalContext* ctx) {
    metal_ctx_ = ctx;

    // Set up mechanics grid (coarser than diffusion grid)
    float mech_voxel_size = 30.0f; // default PhysiCell mechanics voxel
    float domain_x = config.x_max - config.x_min;
    float domain_y = config.y_max - config.y_min;
    float domain_z = config.z_max - config.z_min;

    params_.dt = config.dt_mechanics;
    params_.cell_cell_repulsion = 10.0f;
    params_.cell_cell_adhesion = 0.4f;
    params_.max_interaction_distance = mech_voxel_size;
    params_.mechanics_voxel_size = mech_voxel_size;
    params_.mech_grid_nx = std::max(1u, (uint32_t)(domain_x / mech_voxel_size));
    params_.mech_grid_ny = std::max(1u, (uint32_t)(domain_y / mech_voxel_size));
    params_.mech_grid_nz = config.use_2D ? 1 : std::max(1u, (uint32_t)(domain_z / mech_voxel_size));

    total_mech_voxels_ = params_.mech_grid_nx * params_.mech_grid_ny * params_.mech_grid_nz;

    // Allocate spatial hash buffers (Metal shared)
    size_t counts_bytes = total_mech_voxels_ * sizeof(uint32_t);
    size_t cells_bytes = total_mech_voxels_ * MAX_CELLS_PER_VOXEL * sizeof(uint32_t);

    id<MTLBuffer> hcBuf = ctx->allocateBuffer(counts_bytes);
    hash_counts_buffer_ = (__bridge_retained void*)hcBuf;

    id<MTLBuffer> hcellBuf = ctx->allocateBuffer(cells_bytes);
    hash_cells_buffer_ = (__bridge_retained void*)hcellBuf;

    id<MTLBuffer> mpBuf = ctx->allocateBuffer(sizeof(MechanicsParams));
    mech_params_buffer_ = (__bridge_retained void*)mpBuf;
    memcpy([mpBuf contents], &params_, sizeof(MechanicsParams));

    // Grid params buffer for spatial hash computation (needs domain origin)
    GridParams gp{};
    gp.x_min = config.x_min;
    gp.y_min = config.y_min;
    gp.z_min = config.z_min;
    gp.nx = params_.mech_grid_nx;
    gp.ny = params_.mech_grid_ny;
    gp.nz = params_.mech_grid_nz;
    gp.dx = mech_voxel_size;
    gp.dy = mech_voxel_size;
    gp.dz = mech_voxel_size;
    id<MTLBuffer> gpBuf = ctx->allocateBuffer(sizeof(GridParams));
    grid_params_buffer_ = (__bridge_retained void*)gpBuf;
    memcpy([gpBuf contents], &gp, sizeof(GridParams));

    printf("  Mechanics grid: %u x %u x %u voxels (%.0f µm)\n",
           params_.mech_grid_nx, params_.mech_grid_ny, params_.mech_grid_nz,
           mech_voxel_size);
    printf("  Spatial hash: %.2f KB counts + %.2f MB cells\n",
           counts_bytes / 1024.0, cells_bytes / (1024.0 * 1024.0));
}

void CellMechanics::update(CellData& cells, float dt) {
    if (cells.num_cells == 0) return;

    clearSpatialHash();
    buildSpatialHash(cells);
    computeForces(cells);
    integratePositions(cells, dt);
}

void CellMechanics::clearSpatialHash() {
    id<MTLBuffer> hcBuf = (__bridge id<MTLBuffer>)hash_counts_buffer_;
    metal_ctx_->dispatchClearHash(hcBuf, total_mech_voxels_);
}

void CellMechanics::buildSpatialHash(CellData& cells) {
    id<MTLBuffer> gpuBuf = (__bridge id<MTLBuffer>)(cells.gpu_buffer);
    id<MTLBuffer> hcBuf = (__bridge id<MTLBuffer>)hash_counts_buffer_;
    id<MTLBuffer> hcellBuf = (__bridge id<MTLBuffer>)hash_cells_buffer_;
    id<MTLBuffer> mpBuf = (__bridge id<MTLBuffer>)mech_params_buffer_;
    id<MTLBuffer> gpBuf = (__bridge id<MTLBuffer>)grid_params_buffer_;

    metal_ctx_->dispatchBuildHash(gpuBuf, hcBuf, hcellBuf, mpBuf, gpBuf,
                                  cells.num_cells, cells.max_cells);
}

void CellMechanics::computeForces(CellData& cells) {
    id<MTLBuffer> gpuBuf = (__bridge id<MTLBuffer>)(cells.gpu_buffer);
    id<MTLBuffer> hcBuf = (__bridge id<MTLBuffer>)hash_counts_buffer_;
    id<MTLBuffer> hcellBuf = (__bridge id<MTLBuffer>)hash_cells_buffer_;
    id<MTLBuffer> mpBuf = (__bridge id<MTLBuffer>)mech_params_buffer_;

    metal_ctx_->dispatchForces(gpuBuf, hcBuf, hcellBuf, mpBuf,
                               cells.num_cells, cells.max_cells);
}

void CellMechanics::integratePositions(CellData& cells, float dt) {
    id<MTLBuffer> gpuBuf = (__bridge id<MTLBuffer>)(cells.gpu_buffer);
    id<MTLBuffer> gpBuf = (__bridge id<MTLBuffer>)grid_params_buffer_;

    metal_ctx_->dispatchIntegrate(gpuBuf, gpBuf, dt, cells.num_cells, cells.max_cells);
}
