#ifndef PHYSICELL_CELL_DATA_H
#define PHYSICELL_CELL_DATA_H

// ─────────────────────────────────────────────────────────────────────
// CellData — Struct-of-Arrays (SoA) cell storage
//
// GPU-shared fields are float32/uint32, packed into a single contiguous
// buffer that can be backed by an MTLBuffer (shared memory).
// CPU-only fields are float64 for numerical precision in phenotype
// calculations and are allocated separately with malloc.
// ─────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

class CellData {
public:
    explicit CellData(uint32_t max_cells);
    ~CellData();

    // ─── GPU-shared float32 fields ───
    // These pointers will be set to point into a contiguous GPU buffer.
    // Layout: [field0 * max_cells | field1 * max_cells | ...]
    float* position_x;
    float* position_y;
    float* position_z;

    float* velocity_x;
    float* velocity_y;
    float* velocity_z;

    float* prev_velocity_x;
    float* prev_velocity_y;
    float* prev_velocity_z;

    float* radius;
    float* nuclear_radius;
    float* total_volume;
    float* nuclear_volume;

    float* cell_cell_repulsion;
    float* cell_cell_adhesion;
    float* relative_max_adhesion_distance;

    float* motility_speed;
    float* motility_bias;
    float* motility_dir_x;
    float* motility_dir_y;
    float* motility_dir_z;

    float* oncoprotein;

    float* simple_pressure;  // contact pressure from mechanics (GPU-computed)

    // ─── GPU-shared uint32 fields ───
    uint32_t* cell_type;
    uint32_t* current_phase;
    uint32_t* is_alive;
    uint32_t* voxel_index;
    uint32_t* mech_voxel_index;

    // ─── CPU-only float64 fields (phenotype precision) ───
    // Original fields
    double* elapsed_time_in_phase;
    double* phase_duration;
    double* birth_rate;
    double* death_rate;
    double* target_volume;      // target volume for volume ODE (legacy, kept for compat)
    double* volume_change_rate; // tau for V relaxation (legacy, kept for compat)

    // ── Volume fields (matching PhysiCell Volume class) ──
    double* fluid_fraction;               // current fluid fraction
    double* solid_cytoplasmic;            // cytoplasmic solid volume
    double* solid_nuclear;                // nuclear solid volume
    double* fluid;                        // total fluid volume
    double* cytoplasmic_biomass_change_rate;
    double* nuclear_biomass_change_rate;
    double* fluid_change_rate;
    double* calcification_rate;
    double* calcified_fraction;
    double* target_solid_cytoplasmic;
    double* target_solid_nuclear;
    double* target_fluid_fraction;
    double* rupture_volume;               // absolute volume at which cell ruptures
    double* relative_rupture_volume;      // ratio to initial volume at time of death
    double* target_cyto_to_nuclear_ratio; // PhysiCell: target_solid_cyto = ratio * target_solid_nuclear (dynamic)

    // ── Death fields ──
    double* necrosis_rate;                // rate for necrosis check (1/min)
    double* necrosis_threshold;           // pO2 level below which necrosis triggers
    uint32_t* current_death_model;        // 0=none, 1=apoptosis, 2=necrosis
    uint32_t* lysed;                      // flag: has necrotic cell lysed/ruptured?
    double* apoptosis_duration;           // elapsed time since apoptosis started
    double* unlysed_fluid_change_rate;    // fluid change rate before lysis
    double* lysed_fluid_change_rate;      // fluid change rate after lysis

    // ── O₂-dependent per-cell parameters ──
    double* o2_proliferation_saturation;  // pO2 level at which proliferation is maximal
    double* o2_proliferation_threshold;   // pO2 level below which proliferation stops
    double* o2_necrosis_threshold;        // pO2 level below which necrosis may trigger
    double* o2_necrosis_max;              // pO2 level below which necrosis rate is maximal
    double* max_necrosis_rate;            // maximum necrosis rate (1/min)

    // ── Cycle fields ──
    uint32_t* cycle_model_code;           // which cycle model (0=live, 1=Ki67_basic, etc.)
    uint32_t* num_phases;                 // number of phases in cell's cycle model
    double* transition_rate_01;           // rate from phase 0 to next
    double* transition_rate_10;           // rate from phase 1 to next (for Ki67)
    double* transition_rate_20;           // rate from phase 2 to next (Ki67 advanced)
    double* transition_rate_23;           // G2→M rate (flow cytometry)
    double* transition_rate_30;           // M→G0/G1 rate (flow cytometry)
    uint32_t* phase_duration_fixed;       // bitfield: which transitions are fixed duration

    // ── Cell Interaction / Integrity fields ──
    double* damage;                       // accumulated damage (0.0 = healthy)
    double* damage_rate;                  // rate of damage accumulation
    double* damage_repair_rate;           // rate of damage repair
    double* attack_elapsed;               // time since attack started
    uint32_t* attacking_cell;             // index of cell currently attacking this cell (UINT32_MAX = none)

    // ── Spring Attachment fields (CPU-only) ──
    static constexpr uint32_t MAX_ATTACHMENTS = 12;
    uint32_t* attachment_count;            // number of current attachments per cell
    uint32_t* attachment_targets;          // flat array: max_cells * MAX_ATTACHMENTS
    float* attachment_elastic_constant;    // spring constant per cell
    float* attachment_rate;                // attachment formation rate (1/min)
    float* detachment_rate;               // attachment break rate (1/min)

    // Per-cell interaction rates (looked up from type, but stored per-cell for rules overrides)
    float* attack_damage_rate;            // damage inflicted per minute when attacking
    float* attack_duration_field;         // expected attack duration (min)
    float* apoptotic_phagocytosis_rate;   // rate to phagocytose apoptotic cells
    float* necrotic_phagocytosis_rate;    // rate to phagocytose necrotic cells
    float* other_dead_phagocytosis_rate;  // rate to phagocytose other dead cells

    // Per-cell-per-type interaction rates (size = max_cells * num_cell_types)
    // These are flat arrays indexed as [cell_index * num_types + target_type]
    float* live_phagocytosis_rates;       // rate to phagocytose live cells of each type
    float* attack_rates;                  // attack rate toward each cell type
    float* fusion_rates;                  // fusion rate toward each cell type
    float* transformation_rates;          // transformation rate to each cell type
    float* immunogenicities;              // immunogenicity to each attacker type (PhysiCell: target.immunogenicity(attacker_type))
    uint32_t num_cell_types;              // number of registered cell types

    // Per-cell per-substrate arrays (size = max_cells * n_substrates)
    double* secretion_rate;
    double* uptake_rate;
    double* saturation_density;
    double* net_export_rate;

    // ── Custom data — dynamic key-value store (one double per key per cell) ──
    static constexpr uint32_t MAX_CUSTOM_VARS = 16;
    double* custom_data;         // flat: max_cells * MAX_CUSTOM_VARS
    uint32_t num_custom_vars;    // number of registered custom variables

    // ─── Counts ───
    uint32_t num_cells;
    uint32_t max_cells;
    uint32_t n_substrates;  // set after config parse

    // Raw pointer to the MTLBuffer (stored as void* to keep header C++-only)
    void* gpu_buffer = nullptr;

    // ─── Custom variable management ───
    /// Register a custom variable with a name and default value.
    /// Returns the variable slot ID (0-based), or -1 if MAX_CUSTOM_VARS reached.
    int registerCustomVariable(const std::string& name, double default_value = 0.0);

    /// Get the value of a custom variable for a specific cell.
    double getCustomVariable(uint32_t cell_index, int var_id) const;

    /// Set the value of a custom variable for a specific cell.
    void setCustomVariable(uint32_t cell_index, int var_id, double value);

    /// Get the list of registered custom variable names.
    const std::vector<std::string>& getCustomVarNames() const;

    /// Allocate the custom data buffer (call after registering variables).
    void allocateCustomDataBuffer();

    /// Allocate per-cell-per-type interaction rate arrays.
    /// Call after CPU buffers are allocated and num_cell_types is set.
    void allocateInteractionRateBuffers();

    // ─── Cell management ───
    /// Add a new cell. Returns the index of the new cell.
    /// Increments num_cells. Returns UINT32_MAX if full.
    uint32_t addCell();

    /// Remove cell at `index` by swapping with the last cell and decrementing.
    void removeCell(uint32_t index);

    /// Set default values for cell at `index`.
    void setDefaults(uint32_t index);

    // ─── Memory management ───

    /// Number of float32 fields packed into the GPU buffer.
    static constexpr uint32_t NUM_FLOAT_FIELDS = 23;

    /// Number of uint32 fields packed into the GPU buffer.
    static constexpr uint32_t NUM_UINT_FIELDS = 5;

    /// Number of CPU-only double fields.
    static constexpr uint32_t NUM_CPU_DOUBLE_FIELDS = 33;

    /// Number of CPU-only uint32 fields.
    static constexpr uint32_t NUM_CPU_UINT_FIELDS = 6;  // +1 for attachment_count

    /// Number of CPU-only float fields (attachment spring params).
    static constexpr uint32_t NUM_CPU_FLOAT_FIELDS = 3;  // elastic_constant, attach_rate, detach_rate

    /// Total bytes required for the GPU-shared buffer.
    size_t gpuBufferSize() const;

    /// Total bytes required for CPU-only fields (excluding per-substrate).
    size_t cpuBufferSize() const;

    /// Allocate CPU-only buffers with malloc.
    void allocateCPUBuffers();

    /// Point all GPU-shared float*/uint32_t* pointers into `base`.
    /// `base` must be at least gpuBufferSize() bytes, 16-byte aligned.
    void setGPUBufferBase(void* base);

    /// Free CPU-only buffers.
    void freeCPUBuffers();

    /// Allocate per-substrate arrays (call after n_substrates is known).
    void allocateSubstrateBuffers(uint32_t n_substrates);

private:
    bool cpu_buffers_owned_ = false;
    bool substrate_buffers_owned_ = false;
    bool custom_data_owned_ = false;

    // Custom variable registry (names + default values)
    std::vector<std::string> custom_var_names_;
    std::vector<double> custom_var_defaults_;
};

#endif // PHYSICELL_CELL_DATA_H
