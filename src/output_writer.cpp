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
