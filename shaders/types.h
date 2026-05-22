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
    float half_decay;   // 0.5 * decay_rate * dt  (LOD splitting: decay shared equally between X and Y sweeps)
    float _pad[2];
};

// ─── Cell SoA field indices ───
// Shared between CPU (C++) and GPU (Metal) — single source of truth.
// Each field is a separate contiguous array of size max_cells.
#define CELL_FIELD_POS_X        0
#define CELL_FIELD_POS_Y        1
#define CELL_FIELD_POS_Z        2
#define CELL_FIELD_VEL_X        3
#define CELL_FIELD_VEL_Y        4
#define CELL_FIELD_VEL_Z        5
#define CELL_FIELD_PREV_VX      6
#define CELL_FIELD_PREV_VY      7
#define CELL_FIELD_PREV_VZ      8
#define CELL_FIELD_RADIUS       9
#define CELL_FIELD_NUCLEAR_RAD  10
#define CELL_FIELD_TOTAL_VOL    11
#define CELL_FIELD_NUCLEAR_VOL  12
#define CELL_FIELD_REPULSION    13
#define CELL_FIELD_ADHESION     14
#define CELL_FIELD_REL_MAX_ADH  15
#define CELL_FIELD_MOT_SPEED    16
#define CELL_FIELD_MOT_BIAS     17
#define CELL_FIELD_MOT_DIR_X    18
#define CELL_FIELD_MOT_DIR_Y    19
#define CELL_FIELD_MOT_DIR_Z    20
#define CELL_FIELD_ONCOPROTEIN  21
#define CELL_FIELD_PRESSURE     22
#define CELL_FIELD_CELL_TYPE    23
#define CELL_FIELD_CUR_PHASE    24
#define CELL_FIELD_IS_ALIVE     25
#define CELL_FIELD_VOXEL_IDX    26
#define CELL_FIELD_MECH_VOXEL   27
#define NUM_CELL_FIELDS         28

// CPU-only convenience enum (aliases the defines above)
#ifndef __METAL_VERSION__
enum CellField {
    FIELD_POSITION_X = CELL_FIELD_POS_X,
    FIELD_POSITION_Y = CELL_FIELD_POS_Y,
    FIELD_POSITION_Z = CELL_FIELD_POS_Z,
    FIELD_VELOCITY_X = CELL_FIELD_VEL_X,
    FIELD_VELOCITY_Y = CELL_FIELD_VEL_Y,
    FIELD_VELOCITY_Z = CELL_FIELD_VEL_Z,
    FIELD_PREV_VELOCITY_X = CELL_FIELD_PREV_VX,
    FIELD_PREV_VELOCITY_Y = CELL_FIELD_PREV_VY,
    FIELD_PREV_VELOCITY_Z = CELL_FIELD_PREV_VZ,
    FIELD_RADIUS = CELL_FIELD_RADIUS,
    FIELD_NUCLEAR_RADIUS = CELL_FIELD_NUCLEAR_RAD,
    FIELD_TOTAL_VOLUME = CELL_FIELD_TOTAL_VOL,
    FIELD_NUCLEAR_VOLUME = CELL_FIELD_NUCLEAR_VOL,
    FIELD_CELL_CELL_REPULSION = CELL_FIELD_REPULSION,
    FIELD_CELL_CELL_ADHESION = CELL_FIELD_ADHESION,
    FIELD_RELATIVE_MAX_ADHESION_DISTANCE = CELL_FIELD_REL_MAX_ADH,
    FIELD_MOTILITY_SPEED = CELL_FIELD_MOT_SPEED,
    FIELD_MOTILITY_BIAS = CELL_FIELD_MOT_BIAS,
    FIELD_MOTILITY_DIR_X = CELL_FIELD_MOT_DIR_X,
    FIELD_MOTILITY_DIR_Y = CELL_FIELD_MOT_DIR_Y,
    FIELD_MOTILITY_DIR_Z = CELL_FIELD_MOT_DIR_Z,
    FIELD_ONCOPROTEIN = CELL_FIELD_ONCOPROTEIN,
    FIELD_SIMPLE_PRESSURE = CELL_FIELD_PRESSURE,
    FIELD_CELL_TYPE = CELL_FIELD_CELL_TYPE,
    FIELD_CURRENT_PHASE = CELL_FIELD_CUR_PHASE,
    FIELD_IS_ALIVE = CELL_FIELD_IS_ALIVE,
    FIELD_VOXEL_INDEX = CELL_FIELD_VOXEL_IDX,
    FIELD_MECH_VOXEL_INDEX = CELL_FIELD_MECH_VOXEL
};
#endif

// ─── Spatial hash constants ───
#define MAX_CELLS_PER_VOXEL 64
#define MAX_CELLS 200000

// ─── Simulation constants ───
#define PI_F 3.14159265358979323846f

#endif // PHYSICELL_METAL_TYPES_H
