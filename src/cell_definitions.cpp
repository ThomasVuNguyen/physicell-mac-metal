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

    // Volume
    cells.total_volume[index]  = def.total_volume;
    cells.nuclear_volume[index] = def.nuclear_volume;

    // Radii from volume: r = (3V / 4π)^(1/3)
    cells.radius[index] = std::cbrt(3.0f * def.total_volume / (4.0f * PI_F));
    cells.nuclear_radius[index] = std::cbrt(3.0f * def.nuclear_volume / (4.0f * PI_F));

    // Mechanics
    cells.cell_cell_repulsion[index] = def.cell_cell_repulsion;
    cells.cell_cell_adhesion[index]  = def.cell_cell_adhesion;
    cells.relative_max_adhesion_distance[index] = def.max_adhesion_distance;

    // Motility (GPU-shared float fields)
    cells.motility_speed[index] = def.motility_speed;
    cells.motility_bias[index]  = def.motility_bias;
    cells.motility_dir_x[index] = 0.0f;
    cells.motility_dir_y[index] = 0.0f;
    cells.motility_dir_z[index] = 0.0f;

    // Oncoprotein (will be re-sampled from distribution later)
    cells.oncoprotein[index] = def.oncoprotein_mean;

    // Phase & alive
    cells.current_phase[index] = 0;
    cells.is_alive[index] = 1;

    // Zero velocity
    cells.velocity_x[index] = cells.velocity_y[index] = cells.velocity_z[index] = 0.0f;
    cells.prev_velocity_x[index] = cells.prev_velocity_y[index] = cells.prev_velocity_z[index] = 0.0f;

    // CPU-only fields
    if (cells.elapsed_time_in_phase) cells.elapsed_time_in_phase[index] = 0.0;
    if (cells.phase_duration)        cells.phase_duration[index] = 0.0;
    if (cells.birth_rate)            cells.birth_rate[index] = static_cast<double>(def.cycle_rate);
    if (cells.death_rate)            cells.death_rate[index] = static_cast<double>(def.apoptosis_rate);
    if (cells.target_volume)         cells.target_volume[index] = static_cast<double>(def.total_volume);
    if (cells.volume_change_rate)    cells.volume_change_rate[index] = static_cast<double>(def.cytoplasmic_biomass_change_rate);

    // Per-substrate secretion/uptake
    if (cells.secretion_rate && cells.n_substrates > 0) {
        for (uint32_t s = 0; s < cells.n_substrates; s++) {
            size_t off = static_cast<size_t>(index) * cells.n_substrates + s;
            cells.secretion_rate[off] = static_cast<double>(def.secretion_rate);
            cells.uptake_rate[off]    = static_cast<double>(def.uptake_rate);
        }
    }
}

// ─── applyMotilityDefaults ───────────────────────────────────────────

void CellTypeRegistry::applyMotilityDefaults(MotilityData& mot, uint32_t index,
                                              int type_id) const
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
}

// ─── buildFromConfig ─────────────────────────────────────────────────

void CellTypeRegistry::buildFromConfig(const std::vector<CellTypeConfig>& configs) {
    types_.clear();

    for (const auto& cfg : configs) {
        CellTypeDefaults def;
        def.name = cfg.name;
        def.id   = cfg.id;

        // Cycle / death
        def.cycle_rate     = cfg.cycle_rate;
        def.apoptosis_rate = cfg.apoptosis_rate;
        def.necrosis_rate  = cfg.necrosis_rate;

        // Volume
        def.total_volume   = cfg.total_volume;
        def.nuclear_volume = cfg.nuclear_volume;
        def.target_fluid_fraction = cfg.fluid_fraction;

        // Mechanics
        def.cell_cell_repulsion   = cfg.cell_cell_repulsion;
        def.cell_cell_adhesion    = cfg.cell_cell_adhesion;
        def.max_adhesion_distance = cfg.max_adhesion_distance;

        // Motility
        def.motility_speed = cfg.motility_speed;
        def.motility_bias  = cfg.motility_bias;
        def.is_motile      = cfg.motility_enabled;

        // Secretion
        def.secretion_rate = cfg.secretion_rate;
        def.uptake_rate    = cfg.uptake_rate;

        // Oncoprotein distribution
        def.oncoprotein_mean = cfg.oncoprotein_mean;
        def.oncoprotein_sd   = cfg.oncoprotein_sd;
        def.oncoprotein_min  = cfg.oncoprotein_min;
        def.oncoprotein_max  = cfg.oncoprotein_max;

        types_.push_back(def);
    }

    // Make sure all types have properly sized adhesion affinity vectors
    syncAdhesionAffinities();
}
