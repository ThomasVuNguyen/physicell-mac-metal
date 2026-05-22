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
    float cycle_rate          = 0.00072f;    // Ki67 default

    // ─── Death ───
    float apoptosis_rate      = 5.31667e-05f;
    float necrosis_rate       = 0.0f;

    // ─── Volume ───
    float total_volume        = 2494.0f;
    float nuclear_volume      = 540.0f;
    float target_solid_cytoplasmic  = 0.0f;   // computed from volume model
    float target_solid_nuclear      = 0.0f;
    float target_fluid_fraction     = 0.75f;
    float cytoplasmic_biomass_change_rate = 0.0045f;
    float nuclear_biomass_change_rate     = 0.0055f;
    float fluid_change_rate               = 0.05f;

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
    void applyMotilityDefaults(MotilityData& mot, uint32_t index, int type_id) const;

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
