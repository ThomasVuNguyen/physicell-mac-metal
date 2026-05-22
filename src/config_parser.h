#ifndef PHYSICELL_CONFIG_PARSER_H
#define PHYSICELL_CONFIG_PARSER_H

// ─────────────────────────────────────────────────────────────────────
// SimConfig — parsed representation of PhysiCell_settings.xml
// ─────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include "../shaders/types.h"

struct SubstrateConfig {
    std::string name;
    int id = 0;
    float diffusion_coeff = 0.0f;
    float decay_rate = 0.0f;
    float initial_value = 0.0f;
    bool dirichlet_enabled = false;
    float dirichlet_value = 0.0f;
};

struct CellTypeConfig {
    std::string name;
    int id = 0;

    // Cycle
    int cycle_model_code = 5;     // 5=live, 1=Ki67_basic
    float cycle_rate = 0.0f;      // primary transition rate (phase 0→next)
    float cycle_rate_10 = 0.0f;   // secondary transition rate (phase 1→0, for Ki67)
    bool  cycle_fixed_01 = false; // is transition 0→1 fixed duration?
    bool  cycle_fixed_10 = false; // is transition 1→0 fixed duration?

    // Death
    float apoptosis_rate = 0.0f;
    float necrosis_rate = 0.0f;

    // Apoptosis death parameters
    float apoptosis_duration = 516.0f;          // 8.6 * 60 min
    float apop_unlysed_fluid_change_rate = 0.05f; // 3.0/60.0 per min
    float apop_lysed_fluid_change_rate = 0.0f;
    float apop_cyto_biomass_change_rate = 0.01667f; // 1.0/60.0
    float apop_nuc_biomass_change_rate = 0.00583f;  // 0.35/60.0
    float apop_calcification_rate = 0.0f;
    float apop_relative_rupture_volume = 2.0f;

    // Necrosis death parameters
    float nec_unlysed_fluid_change_rate = 0.01117f; // 0.67/60.0
    float nec_lysed_fluid_change_rate = 0.000833f;  // 0.050/60.0
    float nec_cyto_biomass_change_rate = 0.0000533f; // 0.0032/60.0
    float nec_nuc_biomass_change_rate = 0.000217f;   // 0.013/60.0
    float nec_calcification_rate = 0.00007f;         // 0.0042/60.0
    float nec_relative_rupture_volume = 2.0f;

    // Volume
    float total_volume = 2494.0f;
    float nuclear_volume = 540.0f;
    float fluid_fraction = 0.75f;
    float cytoplasmic_biomass_change_rate = 0.0045f; // 0.27/60.0
    float nuclear_biomass_change_rate = 0.0055f;     // 0.33/60.0
    float fluid_change_rate = 0.05f;                 // 3.0/60.0
    float calcification_rate = 0.0f;
    float relative_rupture_volume = 2.0f;

    // Mechanics
    float cell_cell_repulsion = 10.0f;
    float cell_cell_adhesion = 0.4f;
    float max_adhesion_distance = 1.25f;

    // Motility
    float motility_speed = 1.0f;
    float motility_bias = 0.5f;
    bool motility_enabled = false;

    // Secretion (per first substrate, simplified)
    float secretion_rate = 0.0f;
    float uptake_rate = 10.0f;

    // Custom data — oncoprotein distribution parameters
    float oncoprotein_mean = 1.0f;
    float oncoprotein_sd = 0.25f;
    float oncoprotein_min = 0.0f;
    float oncoprotein_max = 2.0f;
};

struct SimConfig {
    // Domain / grid
    GridParams grid{};  // from shaders/types.h

    float x_min = 0.0f, x_max = 0.0f;
    float y_min = 0.0f, y_max = 0.0f;
    float z_min = 0.0f, z_max = 0.0f;
    bool use_2D = true;

    // Timing
    float max_time = 0.0f;
    float dt_diffusion = 0.01f;
    float dt_mechanics = 0.1f;
    float dt_phenotype = 6.0f;

    // Substrates
    std::vector<SubstrateConfig> substrates;

    // Cell types
    std::vector<CellTypeConfig> cell_types;

    // Output
    std::string output_folder = "output";
    float save_interval = 60.0f;

    // User parameters
    int initial_cell_count = 0;
    float tumor_radius = 250.0f;
    float oncoprotein_mean = 1.0f;
    float oncoprotein_sd = 0.25f;
    float oncoprotein_min = 0.0f;
    float oncoprotein_max = 2.0f;

    // Options
    bool virtual_wall = true;
    uint32_t random_seed = 0;
};

/// Parse a PhysiCell_settings.xml file and return a fully populated SimConfig.
/// Throws std::runtime_error on parse failure.
SimConfig parseConfig(const char* xml_path);

#endif // PHYSICELL_CONFIG_PARSER_H
