#ifndef PHYSICELL_METAL_MICROENVIRONMENT_H
#define PHYSICELL_METAL_MICROENVIRONMENT_H

#include "config_parser.h"
#include "../shaders/types.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

// ─────────────────────────────────────────────────────────────────────
// Microenvironment — manages substrate densities and diffusion on GPU
// ─────────────────────────────────────────────────────────────────────

class MetalContext; // forward

class Microenvironment {
public:
    Microenvironment();
    ~Microenvironment();

    // Initialize from config
    void initialize(const SimConfig& config, MetalContext* ctx);

    // Run one diffusion-decay step on GPU
    void simulateDiffusionDecay();

    /// Enforce Dirichlet boundary conditions: reset boundary voxels to target values.
    /// Must be called after each diffusion step.
    void applyDirichletBC();

    // Get density at a voxel (CPU read from shared buffer)
    float getDensity(uint32_t voxel_index, uint32_t substrate_index) const;
    void setDensity(uint32_t voxel_index, uint32_t substrate_index, float value);

    // Apply cell secretion/uptake (CPU-side, called per cell)
    void applySecretion(uint32_t voxel_index, uint32_t substrate_index,
                        float secretion_rate, float uptake_rate,
                        float saturation_density, float dt);

    // Grid info
    uint32_t numVoxels() const { return grid_.nx * grid_.ny * grid_.nz; }
    uint32_t voxelIndex(float x, float y, float z) const;
    GridParams getGridParams() const { return grid_; }

    // ─── Gradient computation (Tier 3) ───
    // Central-difference gradients of density per voxel per substrate.
    // Call computeGradients() after each diffusion step.
    void computeGradients();

    // Get gradient for a specific voxel and substrate.
    // Returns gradient components via output parameters.
    void getGradient(uint32_t voxel_index, uint32_t substrate_index,
                     float& gx, float& gy, float& gz) const;

    // Direct gradient buffer access (substrate-major layout)
    // Layout: gradient_x[substrate * total_voxels + voxel]
    float* gradientX() { return gradient_x_; }
    float* gradientY() { return gradient_y_; }
    float* gradientZ() { return gradient_z_; }

    // Direct buffer access (for GPU dispatch)
    float* densityBuffer() { return density_data_; }
    const SubstrateParams* substrateParams() const { return substrate_params_; }

private:
    GridParams grid_;
    uint32_t n_substrates_;
    float* density_data_;       // points into MTLBuffer
    SubstrateParams* substrate_params_;

    // Gradient buffers (CPU-allocated, substrate-major layout)
    float* gradient_x_;         // size: n_substrates * total_voxels
    float* gradient_y_;
    float* gradient_z_;

    // Metal resources (void* to keep header C++ compatible)
    void* density_buffer_;      // id<MTLBuffer>
    void* grid_params_buffer_;  // id<MTLBuffer>
    void* substrate_params_buffer_; // id<MTLBuffer>
    void* thomas_coeffs_buffer_;    // id<MTLBuffer>
    MetalContext* metal_ctx_;
};

#endif
