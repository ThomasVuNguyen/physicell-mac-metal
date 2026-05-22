#ifndef PHYSICELL_METAL_CELL_DEFINITIONS_H
#define PHYSICELL_METAL_CELL_DEFINITIONS_H

// ─────────────────────────────────────────────────────────────────────
// CellTypeRegistry — stores default parameters for each cell type
//
// Port of PhysiCell's Cell_Definition concept to the SoA architecture.
// When a cell is created, applyDefaults() stamps all its SoA fields
// with the values from the matching CellTypeDefaults.
// ─────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <cstdint>

class CellData;
struct MotilityData;

struct CellTypeDefaults {
    std::string name;
    int id = 0;

    // ─── Cycle ───
    int   cycle_model_code    = 5;           // 5=live, 1=Ki67_basic, 0=Ki67_adv, 2=flow_cyto, 7=cycling_Q
    float cycle_rate          = 0.00072f;    // primary transition rate 0→next
    float cycle_rate_10       = 0.006667f;   // secondary rate 1→next (Ki67, flow cyto S→G2)
    float cycle_rate_20       = 0.0f;        // rate 2→0 (Ki67 advanced post-M)
    float cycle_rate_23       = 0.0f;        // G2→M (flow cytometry)
    float cycle_rate_30       = 0.0f;        // M→G0/G1 (flow cytometry)
    bool  cycle_fixed_01      = false;       // fixed duration phase 0→1?
    bool  cycle_fixed_10      = false;       // fixed duration phase 1→next?
    bool  cycle_fixed_20      = false;       // fixed duration phase 2→0?
    bool  cycle_fixed_23      = false;       // fixed duration phase G2→M?
    bool  cycle_fixed_30      = false;       // fixed duration phase M→G0/G1?

    // ─── Death ───
    float apoptosis_rate      = 5.31667e-05f;
    float necrosis_rate       = 0.0f;

    // Apoptosis parameters
    float apoptosis_duration           = 516.0f;      // 8.6 * 60 min
    float apop_unlysed_fluid_change_rate = 0.05f;
    float apop_lysed_fluid_change_rate   = 0.0f;
    float apop_cyto_biomass_change_rate  = 0.01667f;
    float apop_nuc_biomass_change_rate   = 0.00583f;

    // Necrosis parameters
    float nec_unlysed_fluid_change_rate = 0.01117f;
    float nec_lysed_fluid_change_rate   = 0.000833f;
    float nec_cyto_biomass_change_rate  = 0.0000533f;
    float nec_nuc_biomass_change_rate   = 0.000217f;

    // ─── Volume ───
    float total_volume        = 2494.0f;
    float nuclear_volume      = 540.0f;
    float target_solid_cytoplasmic  = 0.0f;   // computed from volume model
    float target_solid_nuclear      = 0.0f;
    float target_fluid_fraction     = 0.75f;
    float cytoplasmic_biomass_change_rate = 0.0045f;
    float nuclear_biomass_change_rate     = 0.0055f;
    float fluid_change_rate               = 0.05f;
    float calcification_rate              = 0.0f;
    float relative_rupture_volume         = 2.0f;
    float fluid_fraction                  = 0.75f;

    // ─── Mechanics ───
    float cell_cell_repulsion      = 10.0f;
    float cell_cell_adhesion       = 0.4f;
    float max_adhesion_distance    = 1.25f;  // relative to R_sum

    // ─── Motility ───
    float motility_speed      = 1.0f;
    float motility_bias        = 0.5f;
    float persistence_time     = 15.0f;      // minutes
    bool  is_motile            = false;
    bool  restrict_to_2D       = true;

    // ─── Secretion (per first substrate, simplified) ───
    float secretion_rate       = 0.0f;
    float uptake_rate          = 10.0f;

    // ─── Oncoprotein distribution ───
    float oncoprotein_mean     = 1.0f;
    float oncoprotein_sd       = 0.25f;
    float oncoprotein_min      = 0.0f;
    float oncoprotein_max      = 2.0f;

    // ─── Adhesion affinities (per cell type) ───
    // adhesion_affinities[j] = affinity of this type toward type j.
    // Defaults to 1.0 for same type, 1.0 for others (fully heterotypic).
    std::vector<float> adhesion_affinities;

    // ─── Cell Integrity ───
    float damage_rate        = 0.0f;
    float damage_repair_rate = 0.0f;

    // ─── Attachment Mechanics ───
    float attachment_elastic_constant = 0.01f;
    float attachment_rate             = 0.0f;
    float detachment_rate             = 0.0f;

    // ─── Multi-substrate chemotactic sensitivities ───
    // If non-empty, overrides single-substrate chemotaxis_index/direction.
    // Size should match n_substrates. Positive = up-gradient, negative = down.
    std::vector<double> chemotactic_sensitivities;

    // ─── Cell Interactions ───
    float attack_damage_rate              = 1.0f;
    float attack_duration                 = 0.1f;   // minutes
    float apoptotic_phagocytosis_rate     = 0.0f;
    float necrotic_phagocytosis_rate      = 0.0f;
    float other_dead_phagocytosis_rate    = 0.0f;
    std::vector<float> live_phagocytosis_rates;   // per target type
    std::vector<float> attack_rates;               // per target type
    std::vector<float> fusion_rates;               // per target type

    // ─── Cell Transformations ───
    std::vector<float> transformation_rates;       // per target type
};

class CellTypeRegistry {
public:
    CellTypeRegistry() = default;

    /// Register a new cell type definition. The def.id must be unique.
    void addType(const CellTypeDefaults& def);

    /// Look up a cell type by numeric ID. Throws if not found.
    const CellTypeDefaults& getType(int id) const;

    /// Look up a cell type by name. Throws if not found.
    const CellTypeDefaults& getType(const std::string& name) const;

    /// Number of registered types.
    int numTypes() const;

    /// Apply default values for the given cell type to cell at `index`
    /// in the CellData SoA arrays.
    void applyDefaults(CellData& cells, uint32_t index, int type_id) const;

    /// Apply default motility values for the given cell type.
    void applyMotilityDefaults(MotilityData& mot, uint32_t index, int type_id,
                               uint32_t n_substrates = 0) const;

    /// Build the registry from a CellTypeConfig vector (parsed from XML).
    /// This is a convenience that converts the config_parser types.
    void buildFromConfig(const std::vector<struct CellTypeConfig>& configs);

    /// After all types are added, ensure adhesion_affinities vectors are
    /// sized to numTypes() with default value 1.0.
    void syncAdhesionAffinities();

    /// Get all types (read-only).
    const std::vector<CellTypeDefaults>& allTypes() const { return types_; }

private:
    std::vector<CellTypeDefaults> types_;
};

#endif // PHYSICELL_METAL_CELL_DEFINITIONS_H
