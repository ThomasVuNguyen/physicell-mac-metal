#ifndef PHYSICELL_CONFIG_PARSER_H
#define PHYSICELL_CONFIG_PARSER_H

// ─────────────────────────────────────────────────────────────────────
// SimConfig — parsed representation of PhysiCell_settings.xml
// ─────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <map>
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
    int cycle_model_code = 5;     // 5=live, 1=Ki67_basic, 0=Ki67_adv, 2=flow_cyto, 7=cycling_Q
    float cycle_rate = 0.0f;      // primary transition rate (phase 0→next)
    float cycle_rate_10 = 0.0f;   // secondary transition rate (phase 1→next)
    float cycle_rate_20 = 0.0f;   // rate 2→0 (Ki67 advanced)
    float cycle_rate_23 = 0.0f;   // G2→M (flow cytometry)
    float cycle_rate_30 = 0.0f;   // M→G0/G1 (flow cytometry)
    bool  cycle_fixed_01 = false; // is transition 0→1 fixed duration?
    bool  cycle_fixed_10 = false; // is transition 1→next fixed duration?
    bool  cycle_fixed_20 = false; // is transition 2→0 fixed duration?
    bool  cycle_fixed_23 = false; // is transition G2→M fixed duration?
    bool  cycle_fixed_30 = false; // is transition M→G0/G1 fixed duration?

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
    float persistence_time = 15.0f;
    bool motility_enabled = false;
    bool restrict_to_2D = true;

    // Chemotaxis
    bool chemotaxis_enabled = false;
    int chemotaxis_substrate = 0;
    int chemotaxis_direction = 1;
    std::vector<double> chemotactic_sensitivities;  // per-substrate, parsed from XML

    // Secretion defaults. Scalar fields are kept for legacy single-substrate
    // callers; vectors are indexed by substrate ID.
    float secretion_rate = 0.0f;
    float uptake_rate = 10.0f;
    std::vector<double> secretion_rates;
    std::vector<double> uptake_rates;
    std::vector<double> saturation_densities;
    std::vector<double> net_export_rates;

    // ─── Cell Interactions ───
    // Per-target-type rates (indexed by target cell type ID)
    std::vector<float> attack_rates;        // attack rate toward each type
    std::vector<float> phagocytosis_rates;  // live phagocytosis rate toward each type
    std::vector<float> fusion_rates;        // fusion rate toward each type

    float apoptotic_phagocytosis_rate = 0.0f;
    float necrotic_phagocytosis_rate = 0.0f;
    float other_dead_phagocytosis_rate = 0.0f;
    float attack_damage_rate = 1.0f;
    float attack_duration = 0.1f;  // minutes

    // ─── Cell Transformations ───
    std::vector<float> transformation_rates;  // per target type

    // ─── Cell Integrity ───
    float damage_rate = 0.0f;
    float damage_repair_rate = 0.0f;

    // ─── Adhesion Affinities ───
    std::vector<float> adhesion_affinities;  // per target cell type

    // ─── Attachment Mechanics ───
    float attachment_elastic_constant = 0.01f;
    float attachment_rate = 0.0f;
    float detachment_rate = 0.0f;
    int max_attachments = 12;

    // Custom data — oncoprotein distribution parameters
    float oncoprotein_mean = 1.0f;
    float oncoprotein_sd = 0.25f;
    float oncoprotein_min = 0.0f;
    float oncoprotein_max = 2.0f;

    // Custom data — arbitrary key-value pairs from <custom_data>
    std::map<std::string, double> custom_data;
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

    // Initial conditions
    std::string initial_conditions_type;       // "csv" or empty
    std::string initial_conditions_csv_file;   // full path to CSV file
    bool initial_conditions_enabled = false;
};

/// Parse a PhysiCell_settings.xml file and return a fully populated SimConfig.
/// Throws std::runtime_error on parse failure.
SimConfig parseConfig(const char* xml_path);

#endif // PHYSICELL_CONFIG_PARSER_H
