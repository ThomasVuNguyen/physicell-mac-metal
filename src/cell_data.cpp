// ─────────────────────────────────────────────────────────────────────
// CellData — Struct-of-Arrays (SoA) cell storage implementation
//
// Tier 1 feature parity: full PhysiCell Volume, Death, and Cycle fields.
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"
#include "../shaders/types.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

// ─── Constructor / Destructor ────────────────────────────────────────

CellData::CellData(uint32_t max_cells)
    : num_cells(0), max_cells(max_cells), n_substrates(0)
{
    // Zero-initialize all pointers so we can safely check for nullptr
    position_x = position_y = position_z = nullptr;
    velocity_x = velocity_y = velocity_z = nullptr;
    prev_velocity_x = prev_velocity_y = prev_velocity_z = nullptr;
    radius = nuclear_radius = nullptr;
    total_volume = nuclear_volume = nullptr;
    cell_cell_repulsion = cell_cell_adhesion = nullptr;
    relative_max_adhesion_distance = nullptr;
    motility_speed = motility_bias = nullptr;
    motility_dir_x = motility_dir_y = motility_dir_z = nullptr;
    oncoprotein = nullptr;
    simple_pressure = nullptr;

    cell_type = current_phase = is_alive = nullptr;
    voxel_index = mech_voxel_index = nullptr;

    // Original CPU-only fields
    elapsed_time_in_phase = phase_duration = nullptr;
    birth_rate = death_rate = nullptr;
    target_volume = volume_change_rate = nullptr;

    // Volume fields
    fluid_fraction = nullptr;
    solid_cytoplasmic = solid_nuclear = nullptr;
    fluid = nullptr;
    cytoplasmic_biomass_change_rate = nuclear_biomass_change_rate = nullptr;
    fluid_change_rate = nullptr;
    calcification_rate = calcified_fraction = nullptr;
    target_solid_cytoplasmic = target_solid_nuclear = nullptr;
    target_fluid_fraction = nullptr;
    rupture_volume = relative_rupture_volume = nullptr;

    // Death fields
    necrosis_rate = necrosis_threshold = nullptr;
    current_death_model = lysed = nullptr;
    apoptosis_duration = nullptr;
    unlysed_fluid_change_rate = lysed_fluid_change_rate = nullptr;
    o2_proliferation_saturation = o2_proliferation_threshold = nullptr;
    o2_necrosis_threshold = o2_necrosis_max = nullptr;
    max_necrosis_rate = nullptr;

    // Cycle fields
    cycle_model_code = num_phases = nullptr;
    transition_rate_01 = transition_rate_10 = nullptr;
    transition_rate_20 = transition_rate_23 = transition_rate_30 = nullptr;
    phase_duration_fixed = nullptr;

    // Interaction / integrity fields
    damage = damage_rate = damage_repair_rate = attack_elapsed = nullptr;
    attacking_cell = nullptr;
    attack_damage_rate = attack_duration_field = nullptr;
    apoptotic_phagocytosis_rate = necrotic_phagocytosis_rate = nullptr;
    other_dead_phagocytosis_rate = nullptr;
    live_phagocytosis_rates = attack_rates = fusion_rates = nullptr;
    transformation_rates = nullptr;
    num_cell_types = 0;

    // Spring attachment fields
    attachment_count = nullptr;
    attachment_targets = nullptr;
    attachment_elastic_constant = nullptr;
    attachment_rate = nullptr;
    detachment_rate = nullptr;

    // Per-substrate
    secretion_rate = uptake_rate = nullptr;
    saturation_density = net_export_rate = nullptr;

    // Custom data
    custom_data = nullptr;
    num_custom_vars = 0;
}

CellData::~CellData() {
    freeCPUBuffers();
    // Note: GPU buffer is NOT freed here — it's owned by the Metal layer
}

// ─── GPU buffer layout ──────────────────────────────────────────────
//
// The GPU buffer packs all float32 arrays first, then all uint32 arrays:
//
//   Offset (in elements of max_cells):
//     0:  position_x       (float)
//     1:  position_y       (float)
//     2:  position_z       (float)
//     3:  velocity_x       (float)
//     4:  velocity_y       (float)
//     5:  velocity_z       (float)
//     6:  prev_velocity_x  (float)
//     7:  prev_velocity_y  (float)
//     8:  prev_velocity_z  (float)
//     9:  radius           (float)
//    10:  nuclear_radius   (float)
//    11:  total_volume     (float)
//    12:  nuclear_volume   (float)
//    13:  cell_cell_repulsion      (float)
//    14:  cell_cell_adhesion       (float)
//    15:  relative_max_adhesion_distance (float)
//    16:  motility_speed   (float)
//    17:  motility_bias    (float)
//    18:  motility_dir_x   (float)
//    19:  motility_dir_y   (float)
//    20:  motility_dir_z   (float)
//    21:  oncoprotein      (float)
//    22:  simple_pressure  (float)
//   --- uint32 section ---
//    23:  cell_type        (uint32)
//    24:  current_phase    (uint32)
//    25:  is_alive         (uint32)
//    26:  voxel_index      (uint32)
//    27:  mech_voxel_index (uint32)
//
// Both float and uint32_t are 4 bytes, so stride is uniform.

size_t CellData::gpuBufferSize() const {
    // Total 4-byte fields: NUM_FLOAT_FIELDS + NUM_UINT_FIELDS
    return static_cast<size_t>(NUM_FLOAT_FIELDS + NUM_UINT_FIELDS)
           * max_cells * sizeof(float);
}

void CellData::setGPUBufferBase(void* base) {
    auto* fbase = static_cast<float*>(base);
    uint32_t stride = max_cells;

    // Float fields (indices 0..22)
    position_x                  = fbase + 0  * stride;
    position_y                  = fbase + 1  * stride;
    position_z                  = fbase + 2  * stride;
    velocity_x                  = fbase + 3  * stride;
    velocity_y                  = fbase + 4  * stride;
    velocity_z                  = fbase + 5  * stride;
    prev_velocity_x             = fbase + 6  * stride;
    prev_velocity_y             = fbase + 7  * stride;
    prev_velocity_z             = fbase + 8  * stride;
    radius                      = fbase + 9  * stride;
    nuclear_radius              = fbase + 10 * stride;
    total_volume                = fbase + 11 * stride;
    nuclear_volume              = fbase + 12 * stride;
    cell_cell_repulsion         = fbase + 13 * stride;
    cell_cell_adhesion          = fbase + 14 * stride;
    relative_max_adhesion_distance = fbase + 15 * stride;
    motility_speed              = fbase + 16 * stride;
    motility_bias               = fbase + 17 * stride;
    motility_dir_x              = fbase + 18 * stride;
    motility_dir_y              = fbase + 19 * stride;
    motility_dir_z              = fbase + 20 * stride;
    oncoprotein                 = fbase + 21 * stride;
    simple_pressure             = fbase + 22 * stride;

    // Uint32 fields (indices 23..27) — same 4-byte element size
    auto* ubase = reinterpret_cast<uint32_t*>(fbase + NUM_FLOAT_FIELDS * stride);
    cell_type                   = ubase + 0 * stride;
    current_phase               = ubase + 1 * stride;
    is_alive                    = ubase + 2 * stride;
    voxel_index                 = ubase + 3 * stride;
    mech_voxel_index            = ubase + 4 * stride;
}

// ─── CPU-only buffers ────────────────────────────────────────────────

// CPU-only double fields (25 total):
//   6 original: elapsed_time_in_phase, phase_duration, birth_rate, death_rate,
//               target_volume, volume_change_rate
//  14 volume:   fluid_fraction, solid_cytoplasmic, solid_nuclear, fluid,
//               cytoplasmic_biomass_change_rate, nuclear_biomass_change_rate,
//               fluid_change_rate, calcification_rate, calcified_fraction,
//               target_solid_cytoplasmic, target_solid_nuclear, target_fluid_fraction,
//               rupture_volume, relative_rupture_volume
//   5 death:    necrosis_rate, necrosis_threshold, apoptosis_duration,
//               unlysed_fluid_change_rate, lysed_fluid_change_rate
// CPU-only uint32 fields (5 total):
//               current_death_model, lysed, cycle_model_code, num_phases,
//               phase_duration_fixed
// Plus 2 cycle doubles: transition_rate_01, transition_rate_10

size_t CellData::cpuBufferSize() const {
    return static_cast<size_t>(NUM_CPU_DOUBLE_FIELDS) * max_cells * sizeof(double)
         + static_cast<size_t>(NUM_CPU_UINT_FIELDS)   * max_cells * sizeof(uint32_t);
}

void CellData::allocateCPUBuffers() {
    if (cpu_buffers_owned_) return;

    size_t n = max_cells;

    // Original 6 double fields
    elapsed_time_in_phase = static_cast<double*>(calloc(n, sizeof(double)));
    phase_duration        = static_cast<double*>(calloc(n, sizeof(double)));
    birth_rate            = static_cast<double*>(calloc(n, sizeof(double)));
    death_rate            = static_cast<double*>(calloc(n, sizeof(double)));
    target_volume         = static_cast<double*>(calloc(n, sizeof(double)));
    volume_change_rate    = static_cast<double*>(calloc(n, sizeof(double)));

    // Volume fields (14 doubles)
    fluid_fraction                = static_cast<double*>(calloc(n, sizeof(double)));
    solid_cytoplasmic             = static_cast<double*>(calloc(n, sizeof(double)));
    solid_nuclear                 = static_cast<double*>(calloc(n, sizeof(double)));
    fluid                         = static_cast<double*>(calloc(n, sizeof(double)));
    cytoplasmic_biomass_change_rate = static_cast<double*>(calloc(n, sizeof(double)));
    nuclear_biomass_change_rate   = static_cast<double*>(calloc(n, sizeof(double)));
    fluid_change_rate             = static_cast<double*>(calloc(n, sizeof(double)));
    calcification_rate            = static_cast<double*>(calloc(n, sizeof(double)));
    calcified_fraction            = static_cast<double*>(calloc(n, sizeof(double)));
    target_solid_cytoplasmic      = static_cast<double*>(calloc(n, sizeof(double)));
    target_solid_nuclear          = static_cast<double*>(calloc(n, sizeof(double)));
    target_fluid_fraction         = static_cast<double*>(calloc(n, sizeof(double)));
    rupture_volume                = static_cast<double*>(calloc(n, sizeof(double)));
    relative_rupture_volume       = static_cast<double*>(calloc(n, sizeof(double)));

    // Death fields (5 doubles + 2 uint32)
    necrosis_rate                 = static_cast<double*>(calloc(n, sizeof(double)));
    necrosis_threshold            = static_cast<double*>(calloc(n, sizeof(double)));
    apoptosis_duration            = static_cast<double*>(calloc(n, sizeof(double)));
    unlysed_fluid_change_rate     = static_cast<double*>(calloc(n, sizeof(double)));
    lysed_fluid_change_rate       = static_cast<double*>(calloc(n, sizeof(double)));
    current_death_model           = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    lysed                         = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));

    // O2/necrosis per-cell parameter fields (5 doubles)
    o2_proliferation_saturation   = static_cast<double*>(calloc(n, sizeof(double)));
    o2_proliferation_threshold    = static_cast<double*>(calloc(n, sizeof(double)));
    o2_necrosis_threshold         = static_cast<double*>(calloc(n, sizeof(double)));
    o2_necrosis_max               = static_cast<double*>(calloc(n, sizeof(double)));
    max_necrosis_rate             = static_cast<double*>(calloc(n, sizeof(double)));

    // Cycle fields (5 doubles + 3 uint32)
    transition_rate_01            = static_cast<double*>(calloc(n, sizeof(double)));
    transition_rate_10            = static_cast<double*>(calloc(n, sizeof(double)));
    transition_rate_20            = static_cast<double*>(calloc(n, sizeof(double)));
    transition_rate_23            = static_cast<double*>(calloc(n, sizeof(double)));
    transition_rate_30            = static_cast<double*>(calloc(n, sizeof(double)));
    cycle_model_code              = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    num_phases                    = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    phase_duration_fixed          = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));

    // Interaction / integrity fields
    damage                        = static_cast<double*>(calloc(n, sizeof(double)));
    damage_rate                   = static_cast<double*>(calloc(n, sizeof(double)));
    damage_repair_rate            = static_cast<double*>(calloc(n, sizeof(double)));
    attack_elapsed                = static_cast<double*>(calloc(n, sizeof(double)));
    attacking_cell                = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));

    // Interaction rate fields (per-cell scalars)
    attack_damage_rate            = static_cast<float*>(calloc(n, sizeof(float)));
    attack_duration_field         = static_cast<float*>(calloc(n, sizeof(float)));
    apoptotic_phagocytosis_rate   = static_cast<float*>(calloc(n, sizeof(float)));
    necrotic_phagocytosis_rate    = static_cast<float*>(calloc(n, sizeof(float)));
    other_dead_phagocytosis_rate  = static_cast<float*>(calloc(n, sizeof(float)));

    // Spring attachment fields
    attachment_count              = static_cast<uint32_t*>(calloc(n, sizeof(uint32_t)));
    attachment_targets            = static_cast<uint32_t*>(calloc(n * MAX_ATTACHMENTS, sizeof(uint32_t)));
    attachment_elastic_constant   = static_cast<float*>(calloc(n, sizeof(float)));
    attachment_rate               = static_cast<float*>(calloc(n, sizeof(float)));
    detachment_rate               = static_cast<float*>(calloc(n, sizeof(float)));

    cpu_buffers_owned_ = true;
}

void CellData::freeCPUBuffers() {
    if (cpu_buffers_owned_) {
        // Original fields
        free(elapsed_time_in_phase); elapsed_time_in_phase = nullptr;
        free(phase_duration);        phase_duration = nullptr;
        free(birth_rate);            birth_rate = nullptr;
        free(death_rate);            death_rate = nullptr;
        free(target_volume);         target_volume = nullptr;
        free(volume_change_rate);    volume_change_rate = nullptr;

        // Volume fields
        free(fluid_fraction);                 fluid_fraction = nullptr;
        free(solid_cytoplasmic);              solid_cytoplasmic = nullptr;
        free(solid_nuclear);                  solid_nuclear = nullptr;
        free(fluid);                          fluid = nullptr;
        free(cytoplasmic_biomass_change_rate); cytoplasmic_biomass_change_rate = nullptr;
        free(nuclear_biomass_change_rate);    nuclear_biomass_change_rate = nullptr;
        free(fluid_change_rate);             fluid_change_rate = nullptr;
        free(calcification_rate);            calcification_rate = nullptr;
        free(calcified_fraction);            calcified_fraction = nullptr;
        free(target_solid_cytoplasmic);      target_solid_cytoplasmic = nullptr;
        free(target_solid_nuclear);          target_solid_nuclear = nullptr;
        free(target_fluid_fraction);         target_fluid_fraction = nullptr;
        free(rupture_volume);                rupture_volume = nullptr;
        free(relative_rupture_volume);       relative_rupture_volume = nullptr;

        // Death fields
        free(necrosis_rate);                 necrosis_rate = nullptr;
        free(necrosis_threshold);            necrosis_threshold = nullptr;
        free(apoptosis_duration);            apoptosis_duration = nullptr;
        free(unlysed_fluid_change_rate);     unlysed_fluid_change_rate = nullptr;
        free(lysed_fluid_change_rate);       lysed_fluid_change_rate = nullptr;
        free(current_death_model);           current_death_model = nullptr;
        free(lysed);                         lysed = nullptr;
        free(o2_proliferation_saturation);   o2_proliferation_saturation = nullptr;
        free(o2_proliferation_threshold);    o2_proliferation_threshold = nullptr;
        free(o2_necrosis_threshold);         o2_necrosis_threshold = nullptr;
        free(o2_necrosis_max);               o2_necrosis_max = nullptr;
        free(max_necrosis_rate);             max_necrosis_rate = nullptr;

        // Cycle fields
        free(transition_rate_01);            transition_rate_01 = nullptr;
        free(transition_rate_10);            transition_rate_10 = nullptr;
        free(transition_rate_20);            transition_rate_20 = nullptr;
        free(transition_rate_23);            transition_rate_23 = nullptr;
        free(transition_rate_30);            transition_rate_30 = nullptr;
        free(cycle_model_code);              cycle_model_code = nullptr;
        free(num_phases);                    num_phases = nullptr;
        free(phase_duration_fixed);          phase_duration_fixed = nullptr;

        // Interaction / integrity
        free(damage);                        damage = nullptr;
        free(damage_rate);                   damage_rate = nullptr;
        free(damage_repair_rate);            damage_repair_rate = nullptr;
        free(attack_elapsed);                attack_elapsed = nullptr;
        free(attacking_cell);                attacking_cell = nullptr;

        // Interaction rate fields
        free(attack_damage_rate);            attack_damage_rate = nullptr;
        free(attack_duration_field);         attack_duration_field = nullptr;
        free(apoptotic_phagocytosis_rate);   apoptotic_phagocytosis_rate = nullptr;
        free(necrotic_phagocytosis_rate);    necrotic_phagocytosis_rate = nullptr;
        free(other_dead_phagocytosis_rate);  other_dead_phagocytosis_rate = nullptr;

        // Spring attachment fields
        free(attachment_count);              attachment_count = nullptr;
        free(attachment_targets);            attachment_targets = nullptr;
        free(attachment_elastic_constant);   attachment_elastic_constant = nullptr;
        free(attachment_rate);               attachment_rate = nullptr;
        free(detachment_rate);               detachment_rate = nullptr;

        // Per-cell-per-type arrays
        free(live_phagocytosis_rates);       live_phagocytosis_rates = nullptr;
        free(attack_rates);                  attack_rates = nullptr;
        free(fusion_rates);                  fusion_rates = nullptr;
        free(transformation_rates);          transformation_rates = nullptr;

        cpu_buffers_owned_ = false;
    }

    if (substrate_buffers_owned_) {
        free(secretion_rate); secretion_rate = nullptr;
        free(uptake_rate);    uptake_rate = nullptr;
        free(saturation_density); saturation_density = nullptr;
        free(net_export_rate);    net_export_rate = nullptr;
        substrate_buffers_owned_ = false;
    }

    if (custom_data_owned_) {
        free(custom_data); custom_data = nullptr;
        custom_data_owned_ = false;
    }
}

void CellData::allocateSubstrateBuffers(uint32_t nsubs) {
    if (substrate_buffers_owned_) {
        free(secretion_rate);
        free(uptake_rate);
        free(saturation_density);
        free(net_export_rate);
    }
    n_substrates = nsubs;
    size_t total = static_cast<size_t>(max_cells) * nsubs;
    secretion_rate     = static_cast<double*>(calloc(total, sizeof(double)));
    uptake_rate        = static_cast<double*>(calloc(total, sizeof(double)));
    saturation_density = static_cast<double*>(calloc(total, sizeof(double)));
    net_export_rate    = static_cast<double*>(calloc(total, sizeof(double)));
    substrate_buffers_owned_ = true;
}

// ─── Cell management ─────────────────────────────────────────────────

uint32_t CellData::addCell() {
    if (num_cells >= max_cells) {
        return UINT32_MAX;  // no room
    }
    uint32_t idx = num_cells;
    num_cells++;
    setDefaults(idx);
    return idx;
}

void CellData::removeCell(uint32_t index) {
    assert(index < num_cells);
    if (num_cells == 0) return;

    uint32_t last = num_cells - 1;
    if (index != last) {
        // Swap-with-last for all GPU-shared float fields
        auto swap_f = [&](float* arr) { arr[index] = arr[last]; };
        swap_f(position_x);  swap_f(position_y);  swap_f(position_z);
        swap_f(velocity_x);  swap_f(velocity_y);  swap_f(velocity_z);
        swap_f(prev_velocity_x); swap_f(prev_velocity_y); swap_f(prev_velocity_z);
        swap_f(radius);      swap_f(nuclear_radius);
        swap_f(total_volume); swap_f(nuclear_volume);
        swap_f(cell_cell_repulsion); swap_f(cell_cell_adhesion);
        swap_f(relative_max_adhesion_distance);
        swap_f(motility_speed); swap_f(motility_bias);
        swap_f(motility_dir_x); swap_f(motility_dir_y); swap_f(motility_dir_z);
        swap_f(oncoprotein);
        swap_f(simple_pressure);

        // GPU-shared uint32 fields
        auto swap_u = [&](uint32_t* arr) { if (arr) arr[index] = arr[last]; };
        swap_u(cell_type); swap_u(current_phase); swap_u(is_alive);
        swap_u(voxel_index); swap_u(mech_voxel_index);

        // CPU-only double fields
        auto swap_d = [&](double* arr) {
            if (arr) arr[index] = arr[last];
        };
        // Original
        swap_d(elapsed_time_in_phase); swap_d(phase_duration);
        swap_d(birth_rate); swap_d(death_rate);
        swap_d(target_volume); swap_d(volume_change_rate);
        // Volume
        swap_d(fluid_fraction);
        swap_d(solid_cytoplasmic); swap_d(solid_nuclear);
        swap_d(fluid);
        swap_d(cytoplasmic_biomass_change_rate);
        swap_d(nuclear_biomass_change_rate);
        swap_d(fluid_change_rate);
        swap_d(calcification_rate); swap_d(calcified_fraction);
        swap_d(target_solid_cytoplasmic); swap_d(target_solid_nuclear);
        swap_d(target_fluid_fraction);
        swap_d(rupture_volume); swap_d(relative_rupture_volume);
        // Death
        swap_d(necrosis_rate); swap_d(necrosis_threshold);
        swap_d(apoptosis_duration);
        swap_d(unlysed_fluid_change_rate); swap_d(lysed_fluid_change_rate);
        swap_d(o2_proliferation_saturation); swap_d(o2_proliferation_threshold);
        swap_d(o2_necrosis_threshold); swap_d(o2_necrosis_max);
        swap_d(max_necrosis_rate);
        swap_u(current_death_model); swap_u(lysed);
        // Cycle
        swap_d(transition_rate_01); swap_d(transition_rate_10);
        swap_d(transition_rate_20); swap_d(transition_rate_23); swap_d(transition_rate_30);
        swap_u(cycle_model_code); swap_u(num_phases); swap_u(phase_duration_fixed);
        // Interaction
        swap_d(damage); swap_d(damage_rate); swap_d(damage_repair_rate);
        swap_d(attack_elapsed); swap_u(attacking_cell);

        // Attachment fields
        if (attachment_count) {
            attachment_count[index] = attachment_count[last];
            // Also update attachment targets in other cells that reference 'last'
            // to point to 'index' instead (since 'last' is being moved to 'index')
            for (uint32_t ci = 0; ci < num_cells; ci++) {
                if (ci == index || ci == last) continue;
                if (!attachment_count || attachment_count[ci] == 0) continue;
                for (uint32_t a = 0; a < attachment_count[ci]; a++) {
                    size_t slot = static_cast<size_t>(ci) * MAX_ATTACHMENTS + a;
                    if (attachment_targets[slot] == last) {
                        attachment_targets[slot] = index;
                    }
                }
            }
        }
        if (attachment_targets) {
            for (uint32_t a = 0; a < MAX_ATTACHMENTS; a++) {
                size_t off_i = static_cast<size_t>(index) * MAX_ATTACHMENTS + a;
                size_t off_l = static_cast<size_t>(last) * MAX_ATTACHMENTS + a;
                attachment_targets[off_i] = attachment_targets[off_l];
            }
        }
        if (attachment_elastic_constant) swap_f(attachment_elastic_constant);
        if (attachment_rate)             swap_f(attachment_rate);
        if (detachment_rate)             swap_f(detachment_rate);

        // Interaction rates (float)
        auto swap_ff = [&](float* arr) { if (arr) arr[index] = arr[last]; };
        swap_ff(attack_damage_rate); swap_ff(attack_duration_field);
        swap_ff(apoptotic_phagocytosis_rate);
        swap_ff(necrotic_phagocytosis_rate);
        swap_ff(other_dead_phagocytosis_rate);

        // Per-cell-per-type arrays
        if (num_cell_types > 0) {
            auto swap_per_type = [&](float* arr) {
                if (!arr) return;
                for (uint32_t t = 0; t < num_cell_types; t++) {
                    size_t off_i = static_cast<size_t>(index) * num_cell_types + t;
                    size_t off_l = static_cast<size_t>(last)  * num_cell_types + t;
                    arr[off_i] = arr[off_l];
                }
            };
            swap_per_type(live_phagocytosis_rates);
            swap_per_type(attack_rates);
            swap_per_type(fusion_rates);
            swap_per_type(transformation_rates);
        }

        // Per-substrate arrays
        if (secretion_rate && n_substrates > 0) {
            for (uint32_t s = 0; s < n_substrates; s++) {
                size_t off_i = static_cast<size_t>(index) * n_substrates + s;
                size_t off_l = static_cast<size_t>(last)  * n_substrates + s;
                secretion_rate[off_i]     = secretion_rate[off_l];
                uptake_rate[off_i]        = uptake_rate[off_l];
                saturation_density[off_i] = saturation_density[off_l];
                net_export_rate[off_i]    = net_export_rate[off_l];
            }
        }
        // Custom data arrays
        if (custom_data && num_custom_vars > 0) {
            for (uint32_t v = 0; v < num_custom_vars; v++) {
                size_t off_i = static_cast<size_t>(index) * MAX_CUSTOM_VARS + v;
                size_t off_l = static_cast<size_t>(last)  * MAX_CUSTOM_VARS + v;
                custom_data[off_i] = custom_data[off_l];
            }
        }
    }

    num_cells--;
}

void CellData::setDefaults(uint32_t index) {
    assert(index < max_cells);

    // Position at origin
    position_x[index] = 0.0f;
    position_y[index] = 0.0f;
    position_z[index] = 0.0f;

    // Zero velocity
    velocity_x[index] = velocity_y[index] = velocity_z[index] = 0.0f;
    prev_velocity_x[index] = prev_velocity_y[index] = prev_velocity_z[index] = 0.0f;

    // ── PhysiCell MCF-7 reference defaults ──
    // Volume = 2494 µm³, Nuclear = 540 µm³, fluid_fraction = 0.75
    double V_total  = 2494.0;
    double V_nuc    = 540.0;
    double ff       = 0.75;
    double V_fluid  = ff * V_total;
    double V_nuc_fluid = ff * V_nuc;
    double V_nuc_solid = V_nuc - V_nuc_fluid;
    double V_cyto   = V_total - V_nuc;
    double V_cyto_solid = V_cyto - ff * V_cyto;

    // GPU-shared volume/geometry
    total_volume[index]   = static_cast<float>(V_total);
    nuclear_volume[index] = static_cast<float>(V_nuc);
    radius[index]         = std::cbrt(3.0f * 2494.0f / (4.0f * PI_F));
    nuclear_radius[index] = std::cbrt(3.0f * 540.0f  / (4.0f * PI_F));

    // Mechanics defaults
    cell_cell_repulsion[index] = 10.0f;
    cell_cell_adhesion[index]  = 0.4f;
    relative_max_adhesion_distance[index] = 1.25f;

    // Motility defaults (disabled by default)
    motility_speed[index] = 1.0f;
    motility_bias[index]  = 0.5f;
    motility_dir_x[index] = 0.0f;
    motility_dir_y[index] = 0.0f;
    motility_dir_z[index] = 0.0f;

    // Oncoprotein
    oncoprotein[index] = 1.0f;

    // Contact pressure (GPU-computed, reset each mechanics step)
    simple_pressure[index] = 0.0f;

    // Integer fields
    cell_type[index]     = 0;
    current_phase[index] = 0;   // phase 0
    is_alive[index]      = 1;
    voxel_index[index]   = 0;
    mech_voxel_index[index] = 0;

    // ── CPU-only: original fields ──
    if (elapsed_time_in_phase) elapsed_time_in_phase[index] = 0.0;
    if (phase_duration)        phase_duration[index] = 0.0;
    if (birth_rate)            birth_rate[index] = 0.00072;
    if (death_rate)            death_rate[index] = 5.31667e-05;
    if (target_volume)         target_volume[index] = V_total;
    if (volume_change_rate)    volume_change_rate[index] = 0.0045;

    // ── CPU-only: Volume fields (PhysiCell Volume defaults) ──
    if (fluid_fraction)                 fluid_fraction[index] = ff;
    if (solid_cytoplasmic)              solid_cytoplasmic[index] = V_cyto_solid;
    if (solid_nuclear)                  solid_nuclear[index] = V_nuc_solid;
    if (fluid)                          fluid[index] = V_fluid;
    if (cytoplasmic_biomass_change_rate) cytoplasmic_biomass_change_rate[index] = 0.27 / 60.0;
    if (nuclear_biomass_change_rate)    nuclear_biomass_change_rate[index] = 0.33 / 60.0;
    if (fluid_change_rate)             fluid_change_rate[index] = 3.0 / 60.0;
    if (calcification_rate)            calcification_rate[index] = 0.0;
    if (calcified_fraction)            calcified_fraction[index] = 0.0;
    if (target_solid_cytoplasmic)      target_solid_cytoplasmic[index] = V_cyto_solid;
    if (target_solid_nuclear)          target_solid_nuclear[index] = V_nuc_solid;
    if (target_fluid_fraction)         target_fluid_fraction[index] = ff;
    if (rupture_volume)                rupture_volume[index] = 2.0 * V_total;
    if (relative_rupture_volume)       relative_rupture_volume[index] = 2.0;

    // ── CPU-only: Death fields ──
    if (necrosis_rate)                 necrosis_rate[index] = 0.0;
    if (necrosis_threshold)            necrosis_threshold[index] = 5.0; // pO2 < 5 mmHg
    if (current_death_model)           current_death_model[index] = 0;  // 0=none
    if (lysed)                         lysed[index] = 0;
    if (apoptosis_duration)            apoptosis_duration[index] = 0.0;
    // Apoptosis death parameters (from PhysiCell defaults)
    if (unlysed_fluid_change_rate)     unlysed_fluid_change_rate[index] = 3.0 / 60.0;
    if (lysed_fluid_change_rate)       lysed_fluid_change_rate[index] = 0.05 / 60.0;

    // O2-dependent per-cell parameters
    if (o2_proliferation_saturation)   o2_proliferation_saturation[index] = 38.0;
    if (o2_proliferation_threshold)    o2_proliferation_threshold[index] = 5.0;
    if (o2_necrosis_threshold)         o2_necrosis_threshold[index] = 5.0;
    if (o2_necrosis_max)               o2_necrosis_max[index] = 2.5;
    if (max_necrosis_rate)             max_necrosis_rate[index] = 1.0 / (6.0 * 60.0);

    // ── CPU-only: Cycle fields ──
    if (cycle_model_code)              cycle_model_code[index] = 5;     // live model
    if (num_phases)                    num_phases[index] = 1;
    if (transition_rate_01)            transition_rate_01[index] = 0.00072; // birth_rate
    if (transition_rate_10)            transition_rate_10[index] = 0.0;
    if (transition_rate_20)            transition_rate_20[index] = 0.0;
    if (transition_rate_23)            transition_rate_23[index] = 0.0;
    if (transition_rate_30)            transition_rate_30[index] = 0.0;
    if (phase_duration_fixed)          phase_duration_fixed[index] = 0;  // stochastic

    // ── CPU-only: Interaction / integrity ──
    if (damage)                        damage[index] = 0.0;
    if (damage_rate)                   damage_rate[index] = 0.0;
    if (damage_repair_rate)            damage_repair_rate[index] = 0.0;
    if (attack_elapsed)                attack_elapsed[index] = 0.0;
    if (attacking_cell)                attacking_cell[index] = UINT32_MAX;  // no attacker

    // ── CPU-only: Spring attachment defaults ──
    if (attachment_count)              attachment_count[index] = 0;
    if (attachment_targets) {
        for (uint32_t a = 0; a < MAX_ATTACHMENTS; a++) {
            attachment_targets[static_cast<size_t>(index) * MAX_ATTACHMENTS + a] = UINT32_MAX;
        }
    }
    if (attachment_elastic_constant)   attachment_elastic_constant[index] = 0.01f;
    if (attachment_rate)               attachment_rate[index] = 0.0f;
    if (detachment_rate)               detachment_rate[index] = 0.0f;

    // Per-substrate defaults
    if (secretion_rate && n_substrates > 0) {
        for (uint32_t s = 0; s < n_substrates; s++) {
            size_t off = static_cast<size_t>(index) * n_substrates + s;
            secretion_rate[off]     = 0.0;
            uptake_rate[off]        = 10.0;
            saturation_density[off] = 1.0;
            net_export_rate[off]    = 0.0;
        }
    }

    // Custom data defaults
    if (custom_data && num_custom_vars > 0) {
        for (uint32_t v = 0; v < num_custom_vars; v++) {
            size_t off = static_cast<size_t>(index) * MAX_CUSTOM_VARS + v;
            custom_data[off] = (v < custom_var_defaults_.size()) ? custom_var_defaults_[v] : 0.0;
        }
    }
}

// ─── Custom variable management ─────────────────────────────────────────────────

int CellData::registerCustomVariable(const std::string& name, double default_value) {
    if (num_custom_vars >= MAX_CUSTOM_VARS) {
        std::fprintf(stderr, "CellData: MAX_CUSTOM_VARS (%u) exceeded for '%s'\n",
                     MAX_CUSTOM_VARS, name.c_str());
        return -1;
    }

    // Check for duplicate
    auto it = std::find(custom_var_names_.begin(), custom_var_names_.end(), name);
    if (it != custom_var_names_.end()) {
        return static_cast<int>(std::distance(custom_var_names_.begin(), it));
    }

    int slot = static_cast<int>(num_custom_vars);
    custom_var_names_.push_back(name);
    custom_var_defaults_.push_back(default_value);
    num_custom_vars++;
    return slot;
}

double CellData::getCustomVariable(uint32_t cell_index, int var_id) const {
    assert(cell_index < max_cells);
    assert(var_id >= 0 && static_cast<uint32_t>(var_id) < num_custom_vars);
    if (!custom_data) return 0.0;
    return custom_data[static_cast<size_t>(cell_index) * MAX_CUSTOM_VARS + var_id];
}

void CellData::setCustomVariable(uint32_t cell_index, int var_id, double value) {
    assert(cell_index < max_cells);
    assert(var_id >= 0 && static_cast<uint32_t>(var_id) < num_custom_vars);
    if (!custom_data) return;
    custom_data[static_cast<size_t>(cell_index) * MAX_CUSTOM_VARS + var_id] = value;
}

const std::vector<std::string>& CellData::getCustomVarNames() const {
    return custom_var_names_;
}

void CellData::allocateCustomDataBuffer() {
    if (custom_data_owned_) {
        free(custom_data);
    }
    size_t total = static_cast<size_t>(max_cells) * MAX_CUSTOM_VARS;
    custom_data = static_cast<double*>(calloc(total, sizeof(double)));
    custom_data_owned_ = true;

    // Apply defaults to all existing cells
    for (uint32_t i = 0; i < num_cells; i++) {
        for (uint32_t v = 0; v < num_custom_vars; v++) {
            size_t off = static_cast<size_t>(i) * MAX_CUSTOM_VARS + v;
            custom_data[off] = (v < custom_var_defaults_.size()) ? custom_var_defaults_[v] : 0.0;
        }
    }
}

void CellData::allocateInteractionRateBuffers() {
    if (num_cell_types == 0) return;
    size_t total = static_cast<size_t>(max_cells) * num_cell_types;
    live_phagocytosis_rates = static_cast<float*>(calloc(total, sizeof(float)));
    attack_rates            = static_cast<float*>(calloc(total, sizeof(float)));
    fusion_rates            = static_cast<float*>(calloc(total, sizeof(float)));
    transformation_rates    = static_cast<float*>(calloc(total, sizeof(float)));
}
