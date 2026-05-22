// ─────────────────────────────────────────────────────────────────────
// Config parser — reads PhysiCell_settings.xml via pugixml
// ─────────────────────────────────────────────────────────────────────

#include "config_parser.h"
#include "../lib/pugixml/pugixml.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <string>

// ─── Helper: safe float parse with default ───
static float getFloat(const pugi::xml_node& node, float def = 0.0f) {
    if (!node) return def;
    const char* text = node.child_value();
    if (!text || text[0] == '\0') return def;
    return std::stof(text);
}

// ─── Helper: safe int parse with default ───
static int getInt(const pugi::xml_node& node, int def = 0) {
    if (!node) return def;
    const char* text = node.child_value();
    if (!text || text[0] == '\0') return def;
    return std::stoi(text);
}

// ─── Helper: safe bool parse ───
static bool getBool(const pugi::xml_node& node, bool def = false) {
    if (!node) return def;
    const char* text = node.child_value();
    if (!text || text[0] == '\0') return def;
    // Accept "true", "True", "TRUE", "1"
    return (strcasecmp(text, "true") == 0 || strcmp(text, "1") == 0);
}

// ─── Helper: safe string parse ───
static std::string getString(const pugi::xml_node& node, const char* def = "") {
    if (!node) return def;
    const char* text = node.child_value();
    if (!text) return def;
    return text;
}

// ─── Main parse function ─────────────────────────────────────────────

SimConfig parseConfig(const char* xml_path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xml_path);
    if (!result) {
        throw std::runtime_error(
            std::string("Failed to parse XML: ") + result.description() +
            " at offset " + std::to_string(result.offset));
    }

    SimConfig config;
    pugi::xml_node root = doc.child("PhysiCell_settings");
    if (!root) {
        throw std::runtime_error("Missing <PhysiCell_settings> root element");
    }

    // ─── Domain ───
    {
        pugi::xml_node domain = root.child("domain");
        config.x_min = getFloat(domain.child("x_min"), -1000.0f);
        config.x_max = getFloat(domain.child("x_max"),  1000.0f);
        config.y_min = getFloat(domain.child("y_min"), -1000.0f);
        config.y_max = getFloat(domain.child("y_max"),  1000.0f);
        config.z_min = getFloat(domain.child("z_min"), -10.0f);
        config.z_max = getFloat(domain.child("z_max"),  10.0f);
        config.use_2D = getBool(domain.child("use_2D"), true);

        float dx = getFloat(domain.child("dx"), 20.0f);
        float dy = getFloat(domain.child("dy"), 20.0f);
        float dz = getFloat(domain.child("dz"), 20.0f);

        // Compute grid dimensions
        config.grid.dx = dx;
        config.grid.dy = dy;
        config.grid.dz = dz;
        config.grid.x_min = config.x_min;
        config.grid.y_min = config.y_min;
        config.grid.z_min = config.z_min;
        config.grid.nx = static_cast<uint32_t>(std::ceil((config.x_max - config.x_min) / dx));
        config.grid.ny = static_cast<uint32_t>(std::ceil((config.y_max - config.y_min) / dy));
        config.grid.nz = static_cast<uint32_t>(std::ceil((config.z_max - config.z_min) / dz));
    }

    // ─── Overall timing ───
    {
        pugi::xml_node overall = root.child("overall");
        config.max_time     = getFloat(overall.child("max_time"), 64800.0f);
        config.dt_diffusion = getFloat(overall.child("dt_diffusion"), 0.01f);
        config.dt_mechanics = getFloat(overall.child("dt_mechanics"), 0.1f);
        config.dt_phenotype = getFloat(overall.child("dt_phenotype"), 6.0f);

        // Also store dt_diffusion in grid params (used by GPU diffusion solver)
        config.grid.dt = config.dt_diffusion;
    }

    // ─── Save ───
    {
        pugi::xml_node save = root.child("save");
        config.output_folder = getString(save.child("folder"), "output");

        pugi::xml_node full_data = save.child("full_data");
        config.save_interval = getFloat(full_data.child("interval"), 60.0f);
    }

    // ─── Options ───
    {
        pugi::xml_node options = root.child("options");
        config.virtual_wall = getBool(options.child("virtual_wall_at_domain_edge"), true);
        config.random_seed  = static_cast<uint32_t>(getInt(options.child("random_seed"), 0));
    }

    // ─── Microenvironment ───
    {
        pugi::xml_node me = root.child("microenvironment_setup");
        uint32_t substrate_count = 0;

        for (pugi::xml_node var = me.child("variable"); var;
             var = var.next_sibling("variable"))
        {
            SubstrateConfig sub;
            sub.name = var.attribute("name").as_string("unnamed");
            sub.id   = var.attribute("ID").as_int(static_cast<int>(substrate_count));

            pugi::xml_node phys = var.child("physical_parameter_set");
            sub.diffusion_coeff = getFloat(phys.child("diffusion_coefficient"), 100000.0f);
            sub.decay_rate      = getFloat(phys.child("decay_rate"), 0.1f);
            sub.initial_value   = getFloat(var.child("initial_condition"), 38.0f);

            pugi::xml_node dirichlet = var.child("Dirichlet_boundary_condition");
            if (dirichlet) {
                const char* enabled = dirichlet.attribute("enabled").as_string("False");
                sub.dirichlet_enabled = (strcasecmp(enabled, "true") == 0 ||
                                         strcmp(enabled, "1") == 0);
                sub.dirichlet_value = std::stof(dirichlet.child_value());
            }

            config.substrates.push_back(sub);
            substrate_count++;
        }

        config.grid.n_substrates = substrate_count;
    }

    // ─── Cell definitions ───
    {
        pugi::xml_node cell_defs = root.child("cell_definitions");
        for (pugi::xml_node cd = cell_defs.child("cell_definition"); cd;
             cd = cd.next_sibling("cell_definition"))
        {
            CellTypeConfig ct;
            ct.name = cd.attribute("name").as_string("unnamed");
            ct.id   = cd.attribute("ID").as_int(0);

            pugi::xml_node phenotype = cd.child("phenotype");

            // ── Cycle ──
            {
                pugi::xml_node cycle = phenotype.child("cycle");
                // Read cycle model code from XML attribute
                ct.cycle_model_code = cycle.attribute("code").as_int(5);

                pugi::xml_node rates = cycle.child("phase_transition_rates");
                if (!rates) rates = cycle.child("phase_durations");

                // Parse all rate elements
                int rate_idx = 0;
                for (pugi::xml_node rate = rates.child("rate"); rate;
                     rate = rate.next_sibling("rate"), rate_idx++)
                {
                    float rval = getFloat(rate, 0.0f);
                    bool fixed = false;
                    const char* fd = rate.attribute("fixed_duration").as_string("false");
                    if (strcasecmp(fd, "true") == 0 || strcmp(fd, "1") == 0) fixed = true;

                    if (rate_idx == 0) {
                        ct.cycle_rate = rval;
                        ct.cycle_fixed_01 = fixed;
                    } else if (rate_idx == 1) {
                        ct.cycle_rate_10 = rval;
                        ct.cycle_fixed_10 = fixed;
                    }
                }
            }

            // ── Death ──
            {
                pugi::xml_node death = phenotype.child("death");
                for (pugi::xml_node model = death.child("model"); model;
                     model = model.next_sibling("model"))
                {
                    int code = model.attribute("code").as_int(0);
                    float dr = getFloat(model.child("death_rate"), 0.0f);

                    if (code == 100) {        // apoptosis
                        ct.apoptosis_rate = dr;

                        // Parse apoptosis death parameters
                        pugi::xml_node params = model.child("parameters");
                        if (params) {
                            ct.apop_unlysed_fluid_change_rate =
                                getFloat(params.child("unlysed_fluid_change_rate"), 3.0f/60.0f);
                            ct.apop_lysed_fluid_change_rate =
                                getFloat(params.child("lysed_fluid_change_rate"), 0.0f);
                            ct.apop_cyto_biomass_change_rate =
                                getFloat(params.child("cytoplasmic_biomass_change_rate"), 1.0f/60.0f);
                            ct.apop_nuc_biomass_change_rate =
                                getFloat(params.child("nuclear_biomass_change_rate"), 0.35f/60.0f);
                            ct.apop_calcification_rate =
                                getFloat(params.child("calcification_rate"), 0.0f);
                            ct.apop_relative_rupture_volume =
                                getFloat(params.child("relative_rupture_volume"), 2.0f);
                        }

                        // Parse apoptosis duration from phase_durations
                        pugi::xml_node ptr = model.child("phase_transition_rates");
                        if (!ptr) ptr = model.child("phase_durations");
                        pugi::xml_node dur = ptr.child("duration");
                        if (!dur) dur = ptr.child("rate");
                        if (dur) {
                            float val = getFloat(dur, 0.0f);
                            // If it's a rate, convert to duration
                            const char* fixed = dur.attribute("fixed_duration").as_string("true");
                            if (strcasecmp(fixed, "true") == 0 || strcmp(fixed, "1") == 0) {
                                ct.apoptosis_duration = val;  // direct duration
                            } else if (val > 0) {
                                ct.apoptosis_duration = 1.0f / val;  // rate to duration
                            }
                        }

                    } else if (code == 101) { // necrosis
                        ct.necrosis_rate = dr;

                        // Parse necrosis death parameters
                        pugi::xml_node params = model.child("parameters");
                        if (params) {
                            ct.nec_unlysed_fluid_change_rate =
                                getFloat(params.child("unlysed_fluid_change_rate"), 0.67f/60.0f);
                            ct.nec_lysed_fluid_change_rate =
                                getFloat(params.child("lysed_fluid_change_rate"), 0.050f/60.0f);
                            ct.nec_cyto_biomass_change_rate =
                                getFloat(params.child("cytoplasmic_biomass_change_rate"), 0.0032f/60.0f);
                            ct.nec_nuc_biomass_change_rate =
                                getFloat(params.child("nuclear_biomass_change_rate"), 0.013f/60.0f);
                            ct.nec_calcification_rate =
                                getFloat(params.child("calcification_rate"), 0.0042f/60.0f);
                            ct.nec_relative_rupture_volume =
                                getFloat(params.child("relative_rupture_volume"), 2.0f);
                        }
                    }
                }
            }

            // ── Volume ──
            {
                pugi::xml_node vol = phenotype.child("volume");
                ct.total_volume   = getFloat(vol.child("total"), 2494.0f);
                ct.nuclear_volume = getFloat(vol.child("nuclear"), 540.0f);
                ct.fluid_fraction = getFloat(vol.child("fluid_fraction"), 0.75f);
                ct.cytoplasmic_biomass_change_rate =
                    getFloat(vol.child("cytoplasmic_biomass_change_rate"), 0.27f/60.0f);
                ct.nuclear_biomass_change_rate =
                    getFloat(vol.child("nuclear_biomass_change_rate"), 0.33f/60.0f);
                ct.fluid_change_rate =
                    getFloat(vol.child("fluid_change_rate"), 3.0f/60.0f);
                ct.calcification_rate =
                    getFloat(vol.child("calcification_rate"), 0.0f);
                ct.relative_rupture_volume =
                    getFloat(vol.child("relative_rupture_volume"), 2.0f);
            }

            // ── Mechanics ──
            {
                pugi::xml_node mech = phenotype.child("mechanics");
                ct.cell_cell_adhesion  = getFloat(mech.child("cell_cell_adhesion_strength"), 0.4f);
                ct.cell_cell_repulsion = getFloat(mech.child("cell_cell_repulsion_strength"), 10.0f);
                ct.max_adhesion_distance = getFloat(
                    mech.child("relative_maximum_adhesion_distance"), 1.25f);
            }

            // ── Motility ──
            {
                pugi::xml_node mot = phenotype.child("motility");
                ct.motility_speed = getFloat(mot.child("speed"), 1.0f);
                ct.motility_bias  = getFloat(mot.child("migration_bias"), 0.5f);
                pugi::xml_node opts = mot.child("options");
                ct.motility_enabled = getBool(opts.child("enabled"), false);
            }

            // ── Secretion (first substrate only for simplified model) ──
            {
                pugi::xml_node sec = phenotype.child("secretion");
                pugi::xml_node sub = sec.child("substrate");
                if (sub) {
                    ct.secretion_rate = getFloat(sub.child("secretion_rate"), 0.0f);
                    ct.uptake_rate    = getFloat(sub.child("uptake_rate"), 10.0f);
                }
            }

            // ── Custom data (oncoprotein) ──
            {
                pugi::xml_node custom = cd.child("custom_data");
                pugi::xml_node onco   = custom.child("oncoprotein");
                // The base value in the cell definition; distribution params
                // come from user_parameters
                if (onco) {
                    ct.oncoprotein_mean = getFloat(onco, 1.0f);
                }
            }

            config.cell_types.push_back(ct);
        }
    }

    // ─── User parameters ───
    {
        pugi::xml_node user = root.child("user_parameters");
        config.initial_cell_count = getInt(user.child("number_of_cells"), 0);
        config.tumor_radius       = getFloat(user.child("tumor_radius"), 250.0f);
        config.oncoprotein_mean   = getFloat(user.child("oncoprotein_mean"), 1.0f);
        config.oncoprotein_sd     = getFloat(user.child("oncoprotein_sd"), 0.25f);
        config.oncoprotein_min    = getFloat(user.child("oncoprotein_min"), 0.0f);
        config.oncoprotein_max    = getFloat(user.child("oncoprotein_max"), 2.0f);

        // Propagate distribution params to cell types
        for (auto& ct : config.cell_types) {
            ct.oncoprotein_mean = config.oncoprotein_mean;
            ct.oncoprotein_sd   = config.oncoprotein_sd;
            ct.oncoprotein_min  = config.oncoprotein_min;
            ct.oncoprotein_max  = config.oncoprotein_max;
        }
    }

    return config;
}
