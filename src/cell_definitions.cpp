// ─────────────────────────────────────────────────────────────────────
// CellTypeRegistry — implementation
// ─────────────────────────────────────────────────────────────────────

#include "cell_definitions.h"
#include "cell_data.h"
#include "motility.h"
#include "config_parser.h"
#include "../shaders/types.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

// ─── addType ─────────────────────────────────────────────────────────

void CellTypeRegistry::addType(const CellTypeDefaults& def) {
    // Check for duplicate IDs
    for (const auto& t : types_) {
        if (t.id == def.id) {
            throw std::runtime_error(
                "CellTypeRegistry: duplicate cell type ID " + std::to_string(def.id));
        }
    }
    types_.push_back(def);
}

// ─── Lookup ──────────────────────────────────────────────────────────

const CellTypeDefaults& CellTypeRegistry::getType(int id) const {
    for (const auto& t : types_) {
        if (t.id == id) return t;
    }
    throw std::runtime_error(
        "CellTypeRegistry: unknown cell type ID " + std::to_string(id));
}

const CellTypeDefaults& CellTypeRegistry::getType(const std::string& name) const {
    for (const auto& t : types_) {
        if (t.name == name) return t;
    }
    throw std::runtime_error(
        "CellTypeRegistry: unknown cell type name '" + name + "'");
}

int CellTypeRegistry::numTypes() const {
    return static_cast<int>(types_.size());
}

// ─── syncAdhesionAffinities ──────────────────────────────────────────

void CellTypeRegistry::syncAdhesionAffinities() {
    int n = numTypes();
    for (auto& t : types_) {
        if (static_cast<int>(t.adhesion_affinities.size()) < n) {
            t.adhesion_affinities.resize(n, 1.0f);
        }
    }
}

// ─── applyDefaults ───────────────────────────────────────────────────

void CellTypeRegistry::applyDefaults(CellData& cells, uint32_t index,
                                      int type_id) const
{
    const CellTypeDefaults& def = getType(type_id);

    // Type
    cells.cell_type[index] = static_cast<uint32_t>(def.id);

    // ── Volume: full decomposition ──
    double V_total = def.total_volume;
    double V_nuc   = def.nuclear_volume;
    double ff      = def.fluid_fraction;
    double V_fluid = ff * V_total;
    double V_nuc_fluid = ff * V_nuc;
    double V_nuc_solid = V_nuc - V_nuc_fluid;
    double V_cyto  = V_total - V_nuc;
    double V_cyto_solid = V_cyto - ff * V_cyto;

    cells.total_volume[index]  = def.total_volume;
    cells.nuclear_volume[index] = def.nuclear_volume;
    cells.radius[index] = std::cbrt(3.0f * def.total_volume / (4.0f * PI_F));
    cells.nuclear_radius[index] = std::cbrt(3.0f * def.nuclear_volume / (4.0f * PI_F));

    cells.fluid_fraction[index] = ff;
    cells.solid_cytoplasmic[index] = V_cyto_solid;
    cells.solid_nuclear[index] = V_nuc_solid;
    cells.fluid[index] = V_fluid;

    // Volume ODE rates
    cells.cytoplasmic_biomass_change_rate[index] = def.cytoplasmic_biomass_change_rate;
    cells.nuclear_biomass_change_rate[index] = def.nuclear_biomass_change_rate;
    cells.fluid_change_rate[index] = def.fluid_change_rate;
    cells.calcification_rate[index] = def.calcification_rate;
    cells.calcified_fraction[index] = 0.0;

    // Volume targets
    cells.target_solid_cytoplasmic[index] = V_cyto_solid;
    cells.target_solid_nuclear[index] = V_nuc_solid;
    cells.target_fluid_fraction[index] = ff;
    cells.target_volume[index] = V_total;
    cells.volume_change_rate[index] = def.cytoplasmic_biomass_change_rate;
    cells.rupture_volume[index] = def.relative_rupture_volume * V_total;
    cells.relative_rupture_volume[index] = def.relative_rupture_volume;

    // ── Mechanics ──
    cells.cell_cell_repulsion[index] = def.cell_cell_repulsion;
    cells.cell_cell_adhesion[index]  = def.cell_cell_adhesion;
    cells.relative_max_adhesion_distance[index] = def.max_adhesion_distance;
    cells.motility_speed[index] = def.motility_speed;

    // Phase & alive
    cells.current_phase[index] = 0;
    cells.is_alive[index] = 1;

    // Zero velocity
    cells.velocity_x[index] = cells.velocity_y[index] = cells.velocity_z[index] = 0.0f;
    cells.prev_velocity_x[index] = cells.prev_velocity_y[index] = cells.prev_velocity_z[index] = 0.0f;

    // ── CPU-only: Cycle fields ──
    if (cells.elapsed_time_in_phase) cells.elapsed_time_in_phase[index] = 0.0;
    if (cells.phase_duration) {
        cells.phase_duration[index] = def.cycle_rate > 0 ? (1.0 / def.cycle_rate) : 1e10;
    }
    if (cells.birth_rate) cells.birth_rate[index] = static_cast<double>(def.cycle_rate);
    if (cells.cycle_model_code) cells.cycle_model_code[index] = static_cast<uint32_t>(def.cycle_model_code);
    if (cells.transition_rate_01) cells.transition_rate_01[index] = def.cycle_rate;
    if (cells.transition_rate_10) cells.transition_rate_10[index] = def.cycle_rate_10;
    if (cells.transition_rate_20) cells.transition_rate_20[index] = def.cycle_rate_20;
    if (cells.transition_rate_23) cells.transition_rate_23[index] = def.cycle_rate_23;
    if (cells.transition_rate_30) cells.transition_rate_30[index] = def.cycle_rate_30;
    if (cells.phase_duration_fixed) {
        cells.phase_duration_fixed[index] = (def.cycle_fixed_01 ? 1u : 0u)
                                          | (def.cycle_fixed_10 ? 2u : 0u)
                                          | (def.cycle_fixed_20 ? 4u : 0u)
                                          | (def.cycle_fixed_23 ? 8u : 0u)
                                          | (def.cycle_fixed_30 ? 16u : 0u);
    }
    if (cells.num_phases) {
        // Map model code to number of phases
        int mc = def.cycle_model_code;
        uint32_t np = 1;
        if (mc == 0) np = 3;       // Ki67 advanced
        else if (mc == 1) np = 2;  // Ki67 basic
        else if (mc == 2 || mc == 6) np = 4;  // flow cytometry
        else if (mc == 7) np = 2;  // cycling-quiescent
        cells.num_phases[index] = np;
    }

    // ── CPU-only: Death fields ──
    if (cells.death_rate) cells.death_rate[index] = static_cast<double>(def.apoptosis_rate);
    if (cells.necrosis_rate) cells.necrosis_rate[index] = static_cast<double>(def.necrosis_rate);
    if (cells.necrosis_threshold) cells.necrosis_threshold[index] = 5.0;
    if (cells.current_death_model) cells.current_death_model[index] = 0;
    if (cells.lysed) cells.lysed[index] = 0;
    if (cells.apoptosis_duration) cells.apoptosis_duration[index] = 0.0;
    if (cells.unlysed_fluid_change_rate) cells.unlysed_fluid_change_rate[index] = def.apop_unlysed_fluid_change_rate;
    if (cells.lysed_fluid_change_rate) cells.lysed_fluid_change_rate[index] = def.apop_lysed_fluid_change_rate;

    // ── CPU-only: O₂-dependent parameters ──
    if (cells.o2_proliferation_threshold)  cells.o2_proliferation_threshold[index] = 5.0;
    if (cells.o2_proliferation_saturation) cells.o2_proliferation_saturation[index] = 38.0;
    if (cells.o2_necrosis_threshold)       cells.o2_necrosis_threshold[index] = 5.0;
    if (cells.o2_necrosis_max)             cells.o2_necrosis_max[index] = 2.5;
    if (cells.max_necrosis_rate)           cells.max_necrosis_rate[index] = 1.0 / (6.0 * 60.0);

    // ── CPU-only: Interaction / integrity ──
    if (cells.damage)              cells.damage[index] = 0.0;
    if (cells.damage_rate)         cells.damage_rate[index] = static_cast<double>(def.damage_rate);
    if (cells.damage_repair_rate)  cells.damage_repair_rate[index] = static_cast<double>(def.damage_repair_rate);
    if (cells.attack_elapsed)      cells.attack_elapsed[index] = 0.0;
    if (cells.attacking_cell)      cells.attacking_cell[index] = UINT32_MAX;

    // Interaction rate scalars
    if (cells.attack_damage_rate)           cells.attack_damage_rate[index] = def.attack_damage_rate;
    if (cells.attack_duration_field)        cells.attack_duration_field[index] = def.attack_duration;
    if (cells.apoptotic_phagocytosis_rate)  cells.apoptotic_phagocytosis_rate[index] = def.apoptotic_phagocytosis_rate;
    if (cells.necrotic_phagocytosis_rate)   cells.necrotic_phagocytosis_rate[index] = def.necrotic_phagocytosis_rate;
    if (cells.other_dead_phagocytosis_rate) cells.other_dead_phagocytosis_rate[index] = def.other_dead_phagocytosis_rate;

    // ── Attachment mechanics ──
    if (cells.attachment_elastic_constant)  cells.attachment_elastic_constant[index] = def.attachment_elastic_constant;
    if (cells.attachment_rate)              cells.attachment_rate[index] = def.attachment_rate;
    if (cells.detachment_rate)             cells.detachment_rate[index] = def.detachment_rate;
    if (cells.attachment_count)             cells.attachment_count[index] = 0;
    if (cells.attachment_targets) {
        for (uint32_t a = 0; a < CellData::MAX_ATTACHMENTS; a++) {
            cells.attachment_targets[static_cast<size_t>(index) * CellData::MAX_ATTACHMENTS + a] = UINT32_MAX;
        }
    }

    // Per-cell-per-type interaction rate arrays
    if (cells.num_cell_types > 0) {
        auto setPerType = [&](float* arr, const std::vector<float>& rates) {
            if (!arr) return;
            for (uint32_t t = 0; t < cells.num_cell_types; t++) {
                size_t off = static_cast<size_t>(index) * cells.num_cell_types + t;
                arr[off] = (t < rates.size()) ? rates[t] : 0.0f;
            }
        };
        setPerType(cells.live_phagocytosis_rates, def.live_phagocytosis_rates);
        setPerType(cells.attack_rates, def.attack_rates);
        setPerType(cells.fusion_rates, def.fusion_rates);
        setPerType(cells.transformation_rates, def.transformation_rates);
    }

    // ── Per-substrate secretion/uptake ──
    if (cells.secretion_rate && cells.n_substrates > 0) {
        for (uint32_t s = 0; s < cells.n_substrates; s++) {
            size_t off = static_cast<size_t>(index) * cells.n_substrates + s;
            cells.secretion_rate[off] = 0.0;  // default off
            cells.uptake_rate[off]    = static_cast<double>(def.uptake_rate);
        }
    }
}

// ─── applyMotilityDefaults ───────────────────────────────────────────

void CellTypeRegistry::applyMotilityDefaults(MotilityData& mot, uint32_t index,
                                              int type_id, uint32_t n_substrates) const
{
    const CellTypeDefaults& def = getType(type_id);

    mot.is_motile[index]        = def.is_motile ? 1 : 0;
    mot.restrict_to_2D[index]   = def.restrict_to_2D ? 1 : 0;
    mot.persistence_time[index] = static_cast<double>(def.persistence_time);
    mot.motility_bias[index]    = static_cast<double>(def.motility_bias);
    mot.motility_elapsed[index] = 0.0;
    mot.motility_vec_x[index]   = 0.0;
    mot.motility_vec_y[index]   = 0.0;
    mot.motility_vec_z[index]   = 0.0;
    mot.chemotaxis_index[index] = 0;
    mot.chemotaxis_direction[index] = 1;

    // Set up multi-substrate chemotactic sensitivities
    if (mot.chemotactic_sensitivity && n_substrates > 0 && mot.n_substrates > 0) {
        uint32_t nsubs = mot.n_substrates;
        size_t base = static_cast<size_t>(index) * nsubs;

        if (!def.chemotactic_sensitivities.empty()) {
            // Use parsed per-substrate sensitivities
            for (uint32_t s = 0; s < nsubs; s++) {
                mot.chemotactic_sensitivity[base + s] =
                    (s < def.chemotactic_sensitivities.size()) ?
                    def.chemotactic_sensitivities[s] : 0.0;
            }
        } else {
            // Fall back: set chemotaxis_index substrate to chemotaxis_direction,
            // all others to 0. This preserves legacy single-substrate behavior
            // when multi-substrate sensitivities aren't explicitly configured.
            for (uint32_t s = 0; s < nsubs; s++) {
                mot.chemotactic_sensitivity[base + s] = 0.0;
            }
            // Use chemotaxis_index/direction to populate the single-substrate entry
            int32_t ci = mot.chemotaxis_index[index];
            if (ci >= 0 && static_cast<uint32_t>(ci) < nsubs) {
                mot.chemotactic_sensitivity[base + ci] =
                    static_cast<double>(mot.chemotaxis_direction[index]);
            }
        }
    }
}

// ─── buildFromConfig ─────────────────────────────────────────────────

void CellTypeRegistry::buildFromConfig(const std::vector<CellTypeConfig>& configs) {
    types_.clear();

    for (const auto& cfg : configs) {
        CellTypeDefaults def;
        def.name = cfg.name;
        def.id   = cfg.id;

        // Cycle
        def.cycle_model_code = cfg.cycle_model_code;
        def.cycle_rate       = cfg.cycle_rate;
        def.cycle_rate_10    = cfg.cycle_rate_10;
        def.cycle_rate_20    = cfg.cycle_rate_20;
        def.cycle_rate_23    = cfg.cycle_rate_23;
        def.cycle_rate_30    = cfg.cycle_rate_30;
        def.cycle_fixed_01   = cfg.cycle_fixed_01;
        def.cycle_fixed_10   = cfg.cycle_fixed_10;
        def.cycle_fixed_20   = cfg.cycle_fixed_20;
        def.cycle_fixed_23   = cfg.cycle_fixed_23;
        def.cycle_fixed_30   = cfg.cycle_fixed_30;

        // Death
        def.apoptosis_rate = cfg.apoptosis_rate;
        def.necrosis_rate  = cfg.necrosis_rate;

        // Apoptosis parameters
        def.apoptosis_duration           = cfg.apoptosis_duration;
        def.apop_unlysed_fluid_change_rate = cfg.apop_unlysed_fluid_change_rate;
        def.apop_lysed_fluid_change_rate   = cfg.apop_lysed_fluid_change_rate;
        def.apop_cyto_biomass_change_rate  = cfg.apop_cyto_biomass_change_rate;
        def.apop_nuc_biomass_change_rate   = cfg.apop_nuc_biomass_change_rate;

        // Necrosis parameters
        def.nec_unlysed_fluid_change_rate = cfg.nec_unlysed_fluid_change_rate;
        def.nec_lysed_fluid_change_rate   = cfg.nec_lysed_fluid_change_rate;
        def.nec_cyto_biomass_change_rate  = cfg.nec_cyto_biomass_change_rate;
        def.nec_nuc_biomass_change_rate   = cfg.nec_nuc_biomass_change_rate;

        // Volume
        def.total_volume   = cfg.total_volume;
        def.nuclear_volume = cfg.nuclear_volume;
        def.fluid_fraction = cfg.fluid_fraction;
        def.target_fluid_fraction = cfg.fluid_fraction;
        def.cytoplasmic_biomass_change_rate = cfg.cytoplasmic_biomass_change_rate;
        def.nuclear_biomass_change_rate = cfg.nuclear_biomass_change_rate;
        def.fluid_change_rate = cfg.fluid_change_rate;
        def.calcification_rate = cfg.calcification_rate;
        def.relative_rupture_volume = cfg.relative_rupture_volume;

        // Mechanics
        def.cell_cell_repulsion   = cfg.cell_cell_repulsion;
        def.cell_cell_adhesion    = cfg.cell_cell_adhesion;
        def.max_adhesion_distance = cfg.max_adhesion_distance;

        // Motility
        def.motility_speed    = cfg.motility_speed;
        def.motility_bias     = cfg.motility_bias;
        def.persistence_time  = cfg.persistence_time;
        def.is_motile         = cfg.motility_enabled;
        def.restrict_to_2D    = cfg.restrict_to_2D;

        // Adhesion affinities
        def.adhesion_affinities = cfg.adhesion_affinities;

        // Cell integrity
        def.damage_rate = cfg.damage_rate;
        def.damage_repair_rate = cfg.damage_repair_rate;

        // Secretion
        def.secretion_rate = cfg.secretion_rate;
        def.uptake_rate    = cfg.uptake_rate;

        // Oncoprotein distribution
        def.oncoprotein_mean = cfg.oncoprotein_mean;
        def.oncoprotein_sd   = cfg.oncoprotein_sd;
        def.oncoprotein_min  = cfg.oncoprotein_min;
        def.oncoprotein_max  = cfg.oncoprotein_max;

        // Cell interactions
        def.attack_damage_rate           = cfg.attack_damage_rate;
        def.attack_duration              = cfg.attack_duration;
        def.apoptotic_phagocytosis_rate  = cfg.apoptotic_phagocytosis_rate;
        def.necrotic_phagocytosis_rate   = cfg.necrotic_phagocytosis_rate;
        def.other_dead_phagocytosis_rate = cfg.other_dead_phagocytosis_rate;
        def.live_phagocytosis_rates      = cfg.phagocytosis_rates;
        def.attack_rates                 = cfg.attack_rates;
        def.fusion_rates                 = cfg.fusion_rates;

        // Cell transformations
        def.transformation_rates = cfg.transformation_rates;

        // Attachment mechanics
        def.attachment_elastic_constant = cfg.attachment_elastic_constant;
        def.attachment_rate             = cfg.attachment_rate;
        def.detachment_rate             = cfg.detachment_rate;

        // Chemotactic sensitivities (parsed from XML as per-substrate vector)
        def.chemotactic_sensitivities = cfg.chemotactic_sensitivities;

        types_.push_back(def);
    }

    // Make sure all types have properly sized adhesion affinity vectors
    syncAdhesionAffinities();
}
