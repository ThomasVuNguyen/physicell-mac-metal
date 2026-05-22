// ─────────────────────────────────────────────────────────────────────
// OutputWriter — writes simulation frames as JSON for the 3D viewer
// ─────────────────────────────────────────────────────────────────────

#include "output_writer.h"
#include "matlab_writer.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <sys/stat.h>
#include <algorithm>

// ─── Helper: ensure directory exists ─────────────────────────────────

static void ensureDirectory(const std::string& path) {
    // Use mkdir -p equivalent (recursive)
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        // Try to create. For simplicity, just create the leaf dir.
        // In production, would do recursive mkdir.
        mkdir(path.c_str(), 0755);
    }
}

// ─── Constructor ─────────────────────────────────────────────────────

OutputWriter::OutputWriter(const std::string& folder)
    : folder_(folder)
{
    ensureDirectory(folder_);
}

// ─── Write a single frame ────────────────────────────────────────────

void OutputWriter::writeFrame(const CellData& cells, double time, int frame_index) {
    // Build filename: output/frame_000042.json
    std::ostringstream fname;
    fname << folder_ << "/frame_" << std::setw(6) << std::setfill('0')
          << frame_index << ".json";
    std::string path = fname.str();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "OutputWriter: failed to open %s\n", path.c_str());
        return;
    }

    // Use a string buffer for performance
    std::ostringstream buf;
    buf << std::fixed << std::setprecision(4);

    buf << "{\"time\":" << time
        << ",\"num_cells\":" << cells.num_cells
        << ",\"cells\":[\n";

    for (uint32_t i = 0; i < cells.num_cells; i++) {
        if (i > 0) buf << ",\n";
        buf << "{\"x\":" << cells.position_x[i]
            << ",\"y\":" << cells.position_y[i]
            << ",\"z\":" << cells.position_z[i]
            << ",\"radius\":" << cells.radius[i]
            << ",\"type\":" << cells.cell_type[i]
            << ",\"phase\":" << cells.current_phase[i]
            << ",\"alive\":" << cells.is_alive[i]
            << ",\"oncoprotein\":" << cells.oncoprotein[i]
            << "}";
    }

    buf << "\n]}";

    ofs << buf.str();
    ofs.close();

    frame_files_.push_back(path);
}

// ─── Write combined multi-frame file ─────────────────────────────────

void OutputWriter::writeAllFrames() {
    if (frame_files_.empty()) return;

    std::string path = folder_ + "/all_frames.json";
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "OutputWriter: failed to open %s\n", path.c_str());
        return;
    }

    ofs << "{\"num_frames\":" << frame_files_.size() << ",\"frames\":[\n";

    for (size_t f = 0; f < frame_files_.size(); f++) {
        // Read each frame file and embed it
        std::ifstream ifs(frame_files_[f]);
        if (!ifs.is_open()) {
            ofs << "null";
        } else {
            // Stream the entire frame file content
            ofs << ifs.rdbuf();
            ifs.close();
        }

        if (f + 1 < frame_files_.size()) {
            ofs << ",\n";
        }
    }

    ofs << "\n]}";
    ofs.close();

    std::printf("OutputWriter: wrote %zu frames to %s\n",
                frame_files_.size(), path.c_str());
}

// ─── MultiCellDS XML Snapshot ────────────────────────────────────────

void writeMultiCellDSSnapshot(const CellData& cells, const float* density,
                              const GridParams& grid, double current_time,
                              double elapsed_sec, int frame_index,
                              const std::string& output_folder) {
    // Build filename: output/output00000000.xml
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/output%08d.xml",
                  output_folder.c_str(), frame_index);

    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "writeMultiCellDSSnapshot: failed to open %s\n", fname);
        return;
    }

    ofs << std::fixed;

    // ─── XML header + root ───
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<MultiCellDS version=\"2\" type=\"snapshot/simulation\">\n";

    // ─── Metadata ───
    ofs << "  <metadata>\n";
    ofs << "    <current_time units=\"min\">" << std::setprecision(4) << current_time << "</current_time>\n";
    ofs << "    <current_runtime units=\"sec\">" << std::setprecision(4) << elapsed_sec << "</current_runtime>\n";
    ofs << "    <labels>\n";
    ofs << "      <label>is_motile</label>\n";
    ofs << "      <label>secretion_rates</label>\n";
    ofs << "      <label>cell_cell_repulsion_strength</label>\n";
    ofs << "      <label>death_rates</label>\n";
    ofs << "    </labels>\n";
    ofs << "  </metadata>\n";

    // ─── Microenvironment ───
    ofs << "  <microenvironment>\n";
    ofs << "    <domain>\n";
    ofs << "      <mesh type=\"Cartesian\" uniform=\"true\">\n";

    // X coordinates
    ofs << "        <x_coordinates delimiter=\" \">";
    for (uint32_t i = 0; i < grid.nx; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(2) << (grid.x_min + (i + 0.5f) * grid.dx);
    }
    ofs << "</x_coordinates>\n";

    // Y coordinates
    ofs << "        <y_coordinates delimiter=\" \">";
    for (uint32_t j = 0; j < grid.ny; j++) {
        if (j > 0) ofs << " ";
        ofs << std::setprecision(2) << (grid.y_min + (j + 0.5f) * grid.dy);
    }
    ofs << "</y_coordinates>\n";

    ofs << "      </mesh>\n";
    ofs << "    </domain>\n";

    // Substrate densities
    ofs << "    <variables>\n";
    uint32_t total_voxels = grid.nx * grid.ny * std::max(grid.nz, 1u);

    for (uint32_t s = 0; s < grid.n_substrates; s++) {
        ofs << "      <variable name=\"substrate_" << s << "\" units=\"mmHg\" ID=\"" << s << "\">\n";
        ofs << "        <data type=\"list\" delimiter=\" \">";
        for (uint32_t v = 0; v < total_voxels; v++) {
            if (v > 0) ofs << " ";
            ofs << std::setprecision(4) << density[s * total_voxels + v];
        }
        ofs << "</data>\n";
        ofs << "      </variable>\n";
    }

    ofs << "    </variables>\n";
    ofs << "  </microenvironment>\n";

    // ─── Cellular information ───
    ofs << "  <cellular_information>\n";
    ofs << "    <cell_populations>\n";
    ofs << "      <cell_population type=\"individual\">\n";
    ofs << "        <custom_cell_data>\n";

    uint32_t n = cells.num_cells;

    // ID
    ofs << "          <simple_variable name=\"ID\" type=\"int\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << i;
    }
    ofs << "</simple_variable>\n";

    // position_x
    ofs << "          <simple_variable name=\"position_x\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.position_x[i];
    }
    ofs << "</simple_variable>\n";

    // position_y
    ofs << "          <simple_variable name=\"position_y\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.position_y[i];
    }
    ofs << "</simple_variable>\n";

    // position_z
    ofs << "          <simple_variable name=\"position_z\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.position_z[i];
    }
    ofs << "</simple_variable>\n";

    // total_volume
    ofs << "          <simple_variable name=\"total_volume\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.total_volume[i];
    }
    ofs << "</simple_variable>\n";

    // cell_type
    ofs << "          <simple_variable name=\"cell_type\" type=\"int\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << cells.cell_type[i];
    }
    ofs << "</simple_variable>\n";

    // current_phase
    ofs << "          <simple_variable name=\"current_phase\" type=\"int\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << cells.current_phase[i];
    }
    ofs << "</simple_variable>\n";

    // elapsed_time_in_phase
    ofs << "          <simple_variable name=\"elapsed_time_in_phase\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.elapsed_time_in_phase[i];
    }
    ofs << "</simple_variable>\n";

    // oncoprotein
    ofs << "          <simple_variable name=\"oncoprotein\" type=\"double\">";
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) ofs << " ";
        ofs << std::setprecision(4) << cells.oncoprotein[i];
    }
    ofs << "</simple_variable>\n";

    // Custom variables
    const auto& customNames = cells.getCustomVarNames();
    for (uint32_t v = 0; v < cells.num_custom_vars; v++) {
        ofs << "          <simple_variable name=\"" << customNames[v] << "\" type=\"double\">";
        for (uint32_t i = 0; i < n; i++) {
            if (i > 0) ofs << " ";
            ofs << std::setprecision(4) << cells.getCustomVariable(i, v);
        }
        ofs << "</simple_variable>\n";
    }

    ofs << "        </custom_cell_data>\n";
    ofs << "      </cell_population>\n";
    ofs << "    </cell_populations>\n";
    ofs << "  </cellular_information>\n";
    ofs << "</MultiCellDS>\n";

    ofs.close();
}

// ─── MATLAB .mat Snapshot ────────────────────────────────────────────

void writeMatlabSnapshots(const CellData& cells, const float* density,
                          const GridParams& grid, int frame_index,
                          const std::string& output_folder) {
    char fname_cells[512];
    std::snprintf(fname_cells, sizeof(fname_cells), "%s/output%08d_cells.mat", output_folder.c_str(), frame_index);

    uint32_t n_cells = cells.num_cells;
    if (n_cells > 0) {
        std::vector<std::vector<double>> cell_mat;
        cell_mat.reserve(n_cells);
        
        uint32_t n_subs = grid.n_substrates;
        uint32_t n_types = std::max(1u, cells.num_cell_types);
        
        for (uint32_t i = 0; i < n_cells; i++) {
            std::vector<double> row;
            row.reserve(100);
            
            // 17 compatibility entries
            row.push_back(i); // ID
            row.push_back(cells.position_x[i]);
            row.push_back(cells.position_y[i]);
            row.push_back(cells.position_z[i]);
            row.push_back(cells.total_volume[i]);
            row.push_back(cells.cell_type ? cells.cell_type[i] : 0.0);
            row.push_back(cells.cycle_model_code ? cells.cycle_model_code[i] : 5.0);
            row.push_back(cells.current_phase ? cells.current_phase[i] : 0.0);
            row.push_back(cells.elapsed_time_in_phase ? cells.elapsed_time_in_phase[i] : 0.0);
            row.push_back(cells.nuclear_volume[i]);
            row.push_back(cells.total_volume[i] - cells.nuclear_volume[i]);
            row.push_back(cells.fluid_fraction ? cells.fluid_fraction[i] : 0.75);
            row.push_back(cells.calcified_fraction ? cells.calcified_fraction[i] : 0.0);
            row.push_back(0.0); row.push_back(0.0); row.push_back(0.0); // orientation
            row.push_back(0.0); // polarity
            
            // state
            row.push_back(cells.velocity_x[i]);
            row.push_back(cells.velocity_y[i]);
            row.push_back(cells.velocity_z[i]);
            row.push_back(cells.simple_pressure ? cells.simple_pressure[i] : 0.0);
            row.push_back(1.0); // number_of_nuclei
            row.push_back(0.0); // total_attack_time
            row.push_back(0.0); // contact_BM
            
            // cycle & death
            row.push_back(cells.transition_rate_01 ? cells.transition_rate_01[i] : 0.0);
            row.push_back(cells.elapsed_time_in_phase ? cells.elapsed_time_in_phase[i] : 0.0);
            row.push_back(cells.is_alive[i] ? 0.0 : 1.0);
            row.push_back(cells.current_death_model ? cells.current_death_model[i] : 0.0);
            row.push_back(0.0); // apoptosis rate
            row.push_back(cells.necrosis_rate ? cells.necrosis_rate[i] : 0.0);
            
            // volume
            row.push_back(cells.cytoplasmic_biomass_change_rate ? cells.cytoplasmic_biomass_change_rate[i] : 0.0);
            row.push_back(cells.nuclear_biomass_change_rate ? cells.nuclear_biomass_change_rate[i] : 0.0);
            row.push_back(cells.fluid_change_rate ? cells.fluid_change_rate[i] : 0.0);
            row.push_back(cells.calcification_rate ? cells.calcification_rate[i] : 0.0);
            row.push_back(cells.target_solid_cytoplasmic ? cells.target_solid_cytoplasmic[i] : 0.0);
            row.push_back(cells.target_solid_nuclear ? cells.target_solid_nuclear[i] : 0.0);
            row.push_back(cells.target_fluid_fraction ? cells.target_fluid_fraction[i] : 0.0);
            
            // geometry
            row.push_back(cells.radius[i]);
            row.push_back(cells.nuclear_radius[i]);
            row.push_back(4.0f * 3.14159f * cells.radius[i] * cells.radius[i]);
            
            // mechanics
            row.push_back(cells.cell_cell_adhesion ? cells.cell_cell_adhesion[i] : 0.0);
            row.push_back(0.0); // cBM_adhesion
            row.push_back(cells.cell_cell_repulsion ? cells.cell_cell_repulsion[i] : 0.0);
            row.push_back(0.0); // cBM_repulsion
            for (uint32_t t = 0; t < n_types; t++) row.push_back(1.0); // adhesion affinities
            row.push_back(cells.relative_max_adhesion_distance ? cells.relative_max_adhesion_distance[i] : 0.0);
            row.push_back(12.0); // max_attachments
            row.push_back(cells.attachment_elastic_constant ? cells.attachment_elastic_constant[i] : 0.0);
            row.push_back(cells.attachment_rate ? cells.attachment_rate[i] : 0.0);
            row.push_back(cells.detachment_rate ? cells.detachment_rate[i] : 0.0);
            
            // motility
            row.push_back(cells.motility_speed && cells.motility_speed[i] > 0.0 ? 1.0 : 0.0);
            row.push_back(15.0); // persistence_time
            row.push_back(cells.motility_speed ? cells.motility_speed[i] : 0.0);
            row.push_back(cells.motility_dir_x ? cells.motility_dir_x[i] : 0.0);
            row.push_back(cells.motility_dir_y ? cells.motility_dir_y[i] : 0.0);
            row.push_back(cells.motility_dir_z ? cells.motility_dir_z[i] : 0.0);
            row.push_back(cells.motility_bias ? cells.motility_bias[i] : 0.0);
            row.push_back(0.0); row.push_back(0.0); row.push_back(0.0); // motility_vector
            row.push_back(0.0); // chemotaxis_index
            row.push_back(1.0); // chemotaxis_dir
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(0.0); // chemotactic_sens
            
            // secretion
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(cells.secretion_rate ? cells.secretion_rate[i * n_subs + s] : 0.0);
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(cells.uptake_rate ? cells.uptake_rate[i * n_subs + s] : 0.0);
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(cells.saturation_density ? cells.saturation_density[i * n_subs + s] : 1.0);
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(cells.net_export_rate ? cells.net_export_rate[i * n_subs + s] : 0.0);
            
            // molecular
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(0.0);
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(0.0);
            for (uint32_t s = 0; s < n_subs; s++) row.push_back(0.0);
            
            // interactions
            row.push_back(cells.apoptotic_phagocytosis_rate ? cells.apoptotic_phagocytosis_rate[i] : 0.0);
            row.push_back(cells.necrotic_phagocytosis_rate ? cells.necrotic_phagocytosis_rate[i] : 0.0);
            row.push_back(cells.other_dead_phagocytosis_rate ? cells.other_dead_phagocytosis_rate[i] : 0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(cells.live_phagocytosis_rates ? cells.live_phagocytosis_rates[i * n_types + t] : 0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(cells.attack_rates ? cells.attack_rates[i * n_types + t] : 0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(1.0); // immunogenicities
            row.push_back(cells.attacking_cell ? static_cast<double>(cells.attacking_cell[i]) : 0.0);
            row.push_back(cells.attack_damage_rate ? cells.attack_damage_rate[i] : 0.0);
            row.push_back(cells.attack_duration_field ? cells.attack_duration_field[i] : 0.0);
            row.push_back(0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(cells.fusion_rates ? cells.fusion_rates[i * n_types + t] : 0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(cells.transformation_rates ? cells.transformation_rates[i * n_types + t] : 0.0);
            for (uint32_t t = 0; t < n_types; t++) row.push_back(0.0);
            
            // integrity
            row.push_back(cells.damage ? cells.damage[i] : 0.0);
            row.push_back(cells.damage_rate ? cells.damage_rate[i] : 0.0);
            row.push_back(cells.damage_repair_rate ? cells.damage_repair_rate[i] : 0.0);
            
            // custom
            for (uint32_t v = 0; v < cells.num_custom_vars; v++) {
                row.push_back(cells.getCustomVariable(i, v));
            }
            
            cell_mat.push_back(row);
        }
        write_matlab4(cell_mat, fname_cells, "cells");
    }

    // Microenvironment
    char fname_microenv[512];
    std::snprintf(fname_microenv, sizeof(fname_microenv), "%s/output%08d_microenvironment0.mat", output_folder.c_str(), frame_index);

    uint32_t n_voxels = grid.nx * grid.ny * std::max(grid.nz, 1u);
    if (n_voxels > 0 && density != nullptr) {
        size_t n_rows = 4 + grid.n_substrates;
        std::vector<std::vector<double>> microenv_mat(n_voxels, std::vector<double>(n_rows, 0.0));
        
        float v_vol = grid.dx * grid.dy * grid.dz;
        for (uint32_t v = 0; v < n_voxels; v++) {
            uint32_t iz = v / (grid.nx * grid.ny);
            uint32_t rem = v % (grid.nx * grid.ny);
            uint32_t iy = rem / grid.nx;
            uint32_t ix = rem % grid.nx;
            
            microenv_mat[v][0] = grid.x_min + (ix + 0.5f) * grid.dx;
            microenv_mat[v][1] = grid.y_min + (iy + 0.5f) * grid.dy;
            microenv_mat[v][2] = grid.z_min + (iz + 0.5f) * grid.dz;
            microenv_mat[v][3] = v_vol;
            for (uint32_t s = 0; s < grid.n_substrates; s++) {
                microenv_mat[v][4 + s] = density[s * n_voxels + v];
            }
        }
        write_matlab4(microenv_mat, fname_microenv, "multiscale_microenvironment");
    }
}

// ─── Neighbor Graph Export ───────────────────────────────────────────

void writeCellNeighborGraph(const CellData& cells, const std::string& output_folder) {
    std::string path = output_folder + "/final_cell_neighbor_graph.txt";
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    // PhysiCell's neighbor graph is based on nearby interacting cells, not
    // only spring attachments. Use the same contact-style radius criterion
    // used by rules/contact signals so default non-attachment simulations
    // still export their physical neighbor graph.
    for (uint32_t i = 0; i < cells.num_cells; i++) {
        if (cells.is_alive && !cells.is_alive[i]) continue;
        if (cells.total_volume[i] <= 1e-15f) continue;

        for (uint32_t j = i + 1; j < cells.num_cells; j++) {
            if (cells.is_alive && !cells.is_alive[j]) continue;
            if (cells.total_volume[j] <= 1e-15f) continue;

            float dx = cells.position_x[i] - cells.position_x[j];
            float dy = cells.position_y[i] - cells.position_y[j];
            float dz = cells.position_z[i] - cells.position_z[j];
            float rsum = cells.radius[i] + cells.radius[j];
            float scale_i = cells.relative_max_adhesion_distance
                ? cells.relative_max_adhesion_distance[i] : 1.25f;
            float scale_j = cells.relative_max_adhesion_distance
                ? cells.relative_max_adhesion_distance[j] : 1.25f;
            float max_dist = std::max(scale_i, scale_j) * rsum;

            if (dx * dx + dy * dy + dz * dz <= max_dist * max_dist) {
                ofs << i << " " << j << "\n";
            }
        }
    }
    ofs.close();
}
