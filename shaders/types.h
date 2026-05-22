#ifndef PHYSICELL_METAL_TYPES_H
#define PHYSICELL_METAL_TYPES_H

// ─────────────────────────────────────────────────────────────────────
// Shared type definitions between CPU (C++/Obj-C++) and GPU (Metal)
// This header is included by both .cpp/.mm files and .metal shaders
// ─────────────────────────────────────────────────────────────────────

#ifdef __METAL_VERSION__
    // Metal shader context
    #include <metal_stdlib>
    using namespace metal;
    #define DEVICE_CONSTANT constant
    #define DEVICE_BUFFER   device
#else
    // CPU context
    #include <cstdint>
    #define DEVICE_CONSTANT
    #define DEVICE_BUFFER
#endif

// ─── Grid Parameters (for diffusion solver) ───
struct GridParams {
    uint32_t nx;        // number of voxels in x
    uint32_t ny;        // number of voxels in y
    uint32_t nz;        // number of voxels in z
    uint32_t n_substrates;  // number of chemical substrates

    float dx;           // voxel spacing x (microns)
    float dy;           // voxel spacing y
    float dz;           // voxel spacing z
    float dt;           // diffusion timestep (minutes)

    float x_min;
    float y_min;
    float z_min;
    float _pad0;        // alignment padding
};

// ─── Per-substrate diffusion parameters ───
struct SubstrateParams {
    float diffusion_coefficient;    // D (micron²/min)
    float decay_rate;               // lambda (1/min)
    float supply_rate;              // source term
    float target_value;             // Dirichlet boundary value

    uint32_t dirichlet_enabled;     // apply Dirichlet BCs?
    float _pad[3];
};

// ─── Mechanics parameters ───
struct MechanicsParams {
    float dt;                       // mechanics timestep (minutes)
    float cell_cell_repulsion;      // base repulsion coefficient
    float cell_cell_adhesion;       // base adhesion coefficient
    float max_interaction_distance; // cutoff distance (microns)

    float mechanics_voxel_size;     // spatial hash voxel size
    uint32_t mech_grid_nx;          // mechanics grid dimensions
    uint32_t mech_grid_ny;
    uint32_t mech_grid_nz;
};

// ─── Thomas algorithm precomputed coefficients ───
struct ThomasCoeffs {
    float coeff;        // D * dt / (dx * dx)  for the tridiagonal
    float decay_coeff;  // 1.0 / (1.0 + decay_rate * dt)
    float _pad[2];
};

// ─── Cell SoA field indices (enum for clarity) ───
// Each field is a separate contiguous array of size max_cells
#ifndef __METAL_VERSION__
enum CellField {
    // Positions (float32, GPU-writable)
    FIELD_POSITION_X = 0,
    FIELD_POSITION_Y,
    FIELD_POSITION_Z,

    // Velocities (float32, GPU-writable)
    FIELD_VELOCITY_X,
    FIELD_VELOCITY_Y,
    FIELD_VELOCITY_Z,

    // Previous velocities for Adams-Bashforth (float32, GPU-writable)
    FIELD_PREV_VELOCITY_X,
    FIELD_PREV_VELOCITY_Y,
    FIELD_PREV_VELOCITY_Z,

    // Cell properties (float32, CPU-writable, GPU-readable)
    FIELD_RADIUS,
    FIELD_NUCLEAR_RADIUS,
    FIELD_TOTAL_VOLUME,
    FIELD_NUCLEAR_VOLUME,

    // Mechanics (float32, CPU-writable)
    FIELD_CELL_CELL_REPULSION,
    FIELD_CELL_CELL_ADHESION,
    FIELD_RELATIVE_MAX_ADHESION_DISTANCE,

    // Motility (float32, CPU-writable)
    FIELD_MOTILITY_SPEED,
    FIELD_MOTILITY_BIAS,
    FIELD_MOTILITY_DIR_X,
    FIELD_MOTILITY_DIR_Y,
    FIELD_MOTILITY_DIR_Z,

    // Custom data
    FIELD_ONCOPROTEIN,

    // Contact pressure (GPU-computed)
    FIELD_SIMPLE_PRESSURE,

    // Integer fields
    FIELD_CELL_TYPE,
    FIELD_CURRENT_PHASE,
    FIELD_IS_ALIVE,

    // Voxel indices (uint32, GPU-writable)
    FIELD_VOXEL_INDEX,          // microenvironment voxel
    FIELD_MECH_VOXEL_INDEX,     // mechanics grid voxel

    NUM_CELL_FIELDS
};
#endif

// ─── Spatial hash constants ───
#define MAX_CELLS_PER_VOXEL 64
#define MAX_CELLS 200000

// ─── Simulation constants ───
#define PI_F 3.14159265358979323846f

#endif // PHYSICELL_METAL_TYPES_H
