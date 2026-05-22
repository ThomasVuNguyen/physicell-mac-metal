// ─────────────────────────────────────────────────────────────────────
// OutputWriter — writes simulation frames as JSON for the 3D viewer
// ─────────────────────────────────────────────────────────────────────

#include "output_writer.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <sys/stat.h>

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
