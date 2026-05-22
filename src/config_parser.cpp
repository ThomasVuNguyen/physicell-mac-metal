// ─────────────────────────────────────────────────────────────────────
// Config parser — reads PhysiCell_settings.xml via pugixml
// ─────────────────────────────────────────────────────────────────────

#include "config_parser.h"
#include "../lib/pugixml/pugixml.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>

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

static int substrateIndexByNameOrId(const std::vector<SubstrateConfig>& substrates,
                                    const char* name, int fallback_id) {
    if (name && name[0] != '\0') {
        for (size_t i = 0; i < substrates.size(); i++) {
            if (substrates[i].name == name) {
                return static_cast<int>(i);
            }
        }
    }
    if (fallback_id >= 0 && fallback_id < static_cast<int>(substrates.size())) {
        return fallback_id;
    }
    return -1;
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
                    } else if (rate_idx == 2) {
                        ct.cycle_rate_20 = rval;
                        ct.cycle_fixed_20 = fixed;
                    } else if (rate_idx == 3) {
                        ct.cycle_rate_23 = rval;
                        ct.cycle_fixed_23 = fixed;
                    } else if (rate_idx == 4) {
                        ct.cycle_rate_30 = rval;
                        ct.cycle_fixed_30 = fixed;
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

                // Adhesion affinities (per target cell type)
                pugi::xml_node affinities = mech.child("cell_adhesion_affinities");
                for (pugi::xml_node aff = affinities.child("cell_adhesion_affinity"); aff;
                     aff = aff.next_sibling("cell_adhesion_affinity"))
                {
                    ct.adhesion_affinities.push_back(getFloat(aff, 1.0f));
                }

                // Attachment mechanics
                ct.attachment_elastic_constant =
                    getFloat(mech.child("attachment_elastic_constant"), 0.01f);
                ct.attachment_rate = getFloat(mech.child("attachment_rate"), 0.0f);
                ct.detachment_rate = getFloat(mech.child("detachment_rate"), 0.0f);
                ct.max_attachments =
                    mech.child("maximum_number_of_attachments").text().as_int(12);
            }

            // ── Motility ──
            {
                pugi::xml_node mot = phenotype.child("motility");
                ct.motility_speed = getFloat(mot.child("speed"), 1.0f);
                ct.motility_bias  = getFloat(mot.child("migration_bias"), 0.5f);
                ct.persistence_time = getFloat(mot.child("persistence_time"), 15.0f);

                pugi::xml_node opts = mot.child("options");
                ct.motility_enabled = getBool(opts.child("enabled"), false);
                ct.restrict_to_2D = getBool(opts.child("use_2D"), true);

                // Chemotaxis
                pugi::xml_node chemo = opts.child("chemotaxis");
                ct.chemotaxis_enabled = getBool(chemo.child("enabled"), false);
                ct.chemotaxis_direction = chemo.child("direction").text().as_int(1);

                // Parse substrate attribute for legacy single-substrate chemotaxis
                ct.chemotaxis_substrate = chemo.child("substrate").text().as_int(0);

                // Parse chemotactic_sensitivities: comma-separated or
                // child elements <substrate_sensitivity> per substrate
                pugi::xml_node sens_node = opts.child("chemotactic_sensitivities");
                if (sens_node) {
                    // Try as child elements first
                    for (pugi::xml_node s = sens_node.child("chemotactic_sensitivity"); s;
                         s = s.next_sibling("chemotactic_sensitivity")) {
                        ct.chemotactic_sensitivities.push_back(
                            static_cast<double>(getFloat(s, 0.0f)));
                    }
                    // If no child elements, try as comma-separated text
                    if (ct.chemotactic_sensitivities.empty()) {
                        std::string text = getString(sens_node, "");
                        if (!text.empty()) {
                            // Parse comma-separated values
                            size_t pos = 0;
                            while (pos < text.size()) {
                                size_t next = text.find(',', pos);
                                if (next == std::string::npos) next = text.size();
                                std::string val = text.substr(pos, next - pos);
                                // Trim whitespace
                                size_t start = val.find_first_not_of(" \t\n\r");
                                if (start != std::string::npos) {
                                    val = val.substr(start);
                                    size_t end = val.find_last_not_of(" \t\n\r");
                                    if (end != std::string::npos) val = val.substr(0, end + 1);
                                    ct.chemotactic_sensitivities.push_back(std::stod(val));
                                }
                                pos = next + 1;
                            }
                        }
                    }
                }
            }

            // ── Secretion ──
            {
                size_t nsubs = config.substrates.size();
                ct.secretion_rates.assign(nsubs, 0.0);
                ct.uptake_rates.assign(nsubs, 0.0);
                ct.saturation_densities.assign(nsubs, 1.0);
                ct.net_export_rates.assign(nsubs, 0.0);

                pugi::xml_node sec = phenotype.child("secretion");
                for (pugi::xml_node sub = sec.child("substrate"); sub;
                     sub = sub.next_sibling("substrate"))
                {
                    int sid = substrateIndexByNameOrId(
                        config.substrates,
                        sub.attribute("name").as_string(""),
                        sub.attribute("ID").as_int(-1));
                    if (sid < 0) {
                        continue;
                    }

                    ct.secretion_rates[static_cast<size_t>(sid)] =
                        getFloat(sub.child("secretion_rate"), 0.0f);
                    ct.saturation_densities[static_cast<size_t>(sid)] =
                        getFloat(sub.child("secretion_target"), 1.0f);
                    ct.uptake_rates[static_cast<size_t>(sid)] =
                        getFloat(sub.child("uptake_rate"), 0.0f);
                    ct.net_export_rates[static_cast<size_t>(sid)] =
                        getFloat(sub.child("net_export_rate"), 0.0f);
                }

                if (!ct.secretion_rates.empty()) {
                    ct.secretion_rate = static_cast<float>(ct.secretion_rates[0]);
                    ct.uptake_rate = static_cast<float>(ct.uptake_rates[0]);
                }
            }

            // ── Cell Interactions ──
            {
                pugi::xml_node interactions = phenotype.child("cell_interactions");
                if (interactions) {
                    ct.apoptotic_phagocytosis_rate =
                        getFloat(interactions.child("apoptotic_phagocytosis_rate"), 0.0f);
                    ct.necrotic_phagocytosis_rate =
                        getFloat(interactions.child("necrotic_phagocytosis_rate"), 0.0f);
                    ct.other_dead_phagocytosis_rate =
                        getFloat(interactions.child("other_dead_phagocytosis_rate"), 0.0f);
                    ct.attack_damage_rate =
                        getFloat(interactions.child("attack_damage_rate"), 1.0f);
                    ct.attack_duration =
                        getFloat(interactions.child("attack_duration"), 0.1f);

                    // Live phagocytosis rates (per target type)
                    pugi::xml_node phago = interactions.child("live_phagocytosis_rates");
                    for (pugi::xml_node pr = phago.child("phagocytosis_rate"); pr;
                         pr = pr.next_sibling("phagocytosis_rate"))
                    {
                        ct.phagocytosis_rates.push_back(getFloat(pr, 0.0f));
                    }

                    // Attack rates (per target type)
                    pugi::xml_node attacks = interactions.child("attack_rates");
                    for (pugi::xml_node ar = attacks.child("attack_rate"); ar;
                         ar = ar.next_sibling("attack_rate"))
                    {
                        ct.attack_rates.push_back(getFloat(ar, 0.0f));
                    }

                    // Fusion rates (per target type)
                    pugi::xml_node fusions = interactions.child("fusion_rates");
                    for (pugi::xml_node fr = fusions.child("fusion_rate"); fr;
                         fr = fr.next_sibling("fusion_rate"))
                    {
                        ct.fusion_rates.push_back(getFloat(fr, 0.0f));
                    }
                }
            }

            // ── Cell Transformations ──
            {
                pugi::xml_node transformations = phenotype.child("cell_transformations");
                if (transformations) {
                    pugi::xml_node rates = transformations.child("transformation_rates");
                    for (pugi::xml_node tr = rates.child("transformation_rate"); tr;
                         tr = tr.next_sibling("transformation_rate"))
                    {
                        ct.transformation_rates.push_back(getFloat(tr, 0.0f));
                    }
                }
            }

            // ── Cell Integrity ──
            {
                pugi::xml_node integrity = phenotype.child("cell_integrity");
                if (integrity) {
                    ct.damage_rate = getFloat(integrity.child("damage_rate"), 0.0f);
                    ct.damage_repair_rate = getFloat(integrity.child("damage_repair_rate"), 0.0f);
                }
            }

            // ── Custom data (arbitrary key-value pairs) ──
            {
                pugi::xml_node custom = cd.child("custom_data");
                // Legacy: oncoprotein distribution parameters
                pugi::xml_node onco = custom.child("oncoprotein");
                if (onco) {
                    ct.oncoprotein_mean = getFloat(onco, 1.0f);
                }

                // Parse ALL custom_data children as key-value doubles
                for (pugi::xml_node child = custom.first_child(); child;
                     child = child.next_sibling())
                {
                    std::string key = child.name();
                    double val = 0.0;
                    const char* text = child.child_value();
                    if (text && text[0] != '\0') {
                        try { val = std::stod(text); } catch (...) { val = 0.0; }
                    }
                    ct.custom_data[key] = val;
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

    // ─── Initial conditions ───
    {
        pugi::xml_node ic = root.child("initial_conditions");
        if (ic) {
            pugi::xml_node cell_pos = ic.child("cell_positions");
            if (cell_pos) {
                std::string type = cell_pos.attribute("type").as_string("");
                const char* en_str = cell_pos.attribute("enabled").as_string("false");
                bool enabled = (strcasecmp(en_str, "true") == 0 || strcmp(en_str, "1") == 0);

                if (enabled && type == "csv") {
                    config.initial_conditions_type = "csv";
                    config.initial_conditions_enabled = true;

                    std::string folder = getString(cell_pos.child("folder"), "./config");
                    std::string filename = getString(cell_pos.child("filename"), "cells.csv");

                    // Build full path
                    if (!folder.empty() && folder.back() != '/') folder += '/';
                    config.initial_conditions_csv_file = folder + filename;
                }
            }
        }
    }

    return config;
}
