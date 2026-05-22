// ─────────────────────────────────────────────────────────────────────
// SVGWriter — PhysiCell-compatible SVG snapshot renderer
//
// Tier 5A: SVG Output
//
// Renders each simulation frame as a 2000×2000 SVG:
//   - Each cell drawn as a circle at (x,y) with radius proportional to
//     cell radius, using the domain-to-viewport coordinate mapping.
//   - Cell coloring follows PhysiCell pathology conventions:
//       Alive cycling:    Green (brightness modulated by oncoprotein)
//       Alive quiescent:  Blue
//       Apoptotic:        Red
//       Necrotic swelling: Brown
//       Necrotic lysed:    Dark brown
//   - Time label in upper-left corner
//   - 100 µm scale bar in lower-right corner
//   - File saved as snapshot_NNNNNN.svg
//
// Reference: PhysiCell modules/PhysiCell_SVG.cpp,
//            modules/PhysiCell_pathology.cpp
// ─────────────────────────────────────────────────────────────────────

#include "svg_writer.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

// ─── Helper: ensure directory exists ─────────────────────────────────

static void ensureDir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

// ─── Cell color determination ────────────────────────────────────────
// Returns an SVG-compatible fill color string based on cell state.
//
// We look at:
//   current_death_model[i]:  DEATH_NONE (0), DEATH_APOPTOSIS (1), DEATH_NECROSIS (2)
//   lysed[i]:                0 = swelling phase, 1 = lysed phase
//   is_alive[i]:             1 = alive, 0 = dead
//   oncoprotein[i]:          modulates green brightness for cycling cells

static std::string cellFillColor(const CellData& cells, uint32_t i) {
    uint32_t death_model = cells.current_death_model[i];

    if (death_model == 1) {
        // Apoptotic — red
        return "rgb(255,0,0)";
    }

    if (death_model == 2) {
        if (cells.lysed[i] == 0) {
            // Necrotic swelling — brown
            return "rgb(139,90,43)";
        } else {
            // Necrotic lysed — dark brown
            return "rgb(101,67,33)";
        }
    }

    // Alive cell
    if (cells.is_alive[i] == 1) {
        // Use oncoprotein to modulate green channel brightness.
        // Base: green (0, G, 0) where G varies with oncoprotein.
        // oncoprotein ~1.0 → bright green, ~0 → dark green, >1 → lighter/yellow-ish
        double onco = static_cast<double>(cells.oncoprotein[i]);
        // Clamp to [0, 2]
        onco = std::max(0.0, std::min(onco, 2.0));

        // Map oncoprotein to green intensity: [0,2] → [80, 255]
        int g = static_cast<int>(80.0 + (255.0 - 80.0) * (onco / 2.0));
        g = std::min(255, std::max(0, g));

        // Small red component for warm tone when oncoprotein is high
        int r = static_cast<int>(onco > 1.0 ? 40.0 * (onco - 1.0) : 0.0);
        r = std::min(80, std::max(0, r));

        return "rgb(" + std::to_string(r) + "," + std::to_string(g) + ",0)";
    }

    // Fallback: grey (shouldn't normally reach here)
    return "rgb(128,128,128)";
}

static std::string cellStrokeColor(const CellData& cells, uint32_t i) {
    // Darker outline matching PhysiCell conventions
    uint32_t death_model = cells.current_death_model[i];
    if (death_model == 1) return "rgb(180,0,0)";      // dark red for apoptotic
    if (death_model == 2) return "rgb(80,40,20)";      // dark brown for necrotic
    return "rgb(0,80,0)";                               // dark green for alive
}

// ─── SVGWriter::writeFrame ───────────────────────────────────────────

void SVGWriter::writeFrame(const CellData& cells, double time, int frame_index,
                           const std::string& output_folder,
                           float x_min, float x_max, float y_min, float y_max)
{
    ensureDir(output_folder);

    // Build filename: snapshot_000042.svg
    std::ostringstream fname;
    fname << output_folder << "/snapshot_"
          << std::setw(6) << std::setfill('0') << frame_index << ".svg";
    std::string path = fname.str();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "SVGWriter: failed to open %s\n", path.c_str());
        return;
    }

    // Domain dimensions in microns
    float domain_w = x_max - x_min;
    float domain_h = y_max - y_min;

    // SVG pixel dimensions
    constexpr int SVG_W = 2000;
    constexpr int SVG_H = 2000;

    // ─── SVG header ───
    // We use a viewBox matching the domain coordinates so cell positions
    // map directly into the SVG coordinate space. This avoids manual
    // coordinate transforms for each circle.
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        << "<!-- Created with PhysiCell-Metal (https://github.com/physicell-metal) -->\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\""
        << " viewBox=\"" << x_min << " " << y_min << " " << domain_w << " " << domain_h << "\""
        << " width=\"" << SVG_W << "\" height=\"" << SVG_H << "\">\n";

    // ─── Background rectangle ───
    ofs << "  <rect x=\"" << x_min << "\" y=\"" << y_min
        << "\" width=\"" << domain_w << "\" height=\"" << domain_h
        << "\" fill=\"white\" stroke=\"black\" stroke-width=\""
        << (domain_w / SVG_W) * 2.0f << "\"/>\n";

    // ─── Draw cells ───
    // Draw dead cells first (below), then alive cells on top.
    // This matches PhysiCell's z-ordering convention.

    // Pass 1: dead cells
    for (uint32_t i = 0; i < cells.num_cells; i++) {
        if (cells.is_alive[i] == 1) continue;

        float cx = cells.position_x[i];
        float cy = cells.position_y[i];
        float r  = cells.radius[i];

        // Stroke width scales with domain for consistent appearance
        float sw = (domain_w / SVG_W) * 0.5f;

        ofs << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
            << "\" fill=\"" << cellFillColor(cells, i)
            << "\" stroke=\"" << cellStrokeColor(cells, i)
            << "\" stroke-width=\"" << sw << "\"/>\n";
    }

    // Pass 2: alive cells
    for (uint32_t i = 0; i < cells.num_cells; i++) {
        if (cells.is_alive[i] != 1) continue;

        float cx = cells.position_x[i];
        float cy = cells.position_y[i];
        float r  = cells.radius[i];

        float sw = (domain_w / SVG_W) * 0.5f;

        ofs << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
            << "\" fill=\"" << cellFillColor(cells, i)
            << "\" stroke=\"" << cellStrokeColor(cells, i)
            << "\" stroke-width=\"" << sw << "\"/>\n";
    }

    // ─── Time label ───
    // Font size in domain coordinates so it's readable
    float font_size = domain_h * 0.025f;  // 2.5% of domain height
    float text_x = x_min + domain_w * 0.02f;
    float text_y = y_min + font_size * 1.5f;

    // Format time as hours + minutes
    char time_str[128];
    if (time < 60.0) {
        std::snprintf(time_str, sizeof(time_str), "t = %.1f min", time);
    } else {
        double hours = time / 60.0;
        std::snprintf(time_str, sizeof(time_str), "t = %.2f h (%.0f min)", hours, time);
    }

    ofs << "  <text x=\"" << text_x << "\" y=\"" << text_y
        << "\" font-family=\"Arial, sans-serif\" font-size=\"" << font_size
        << "\" fill=\"black\">" << time_str << "</text>\n";

    // Cell count label
    char count_str[128];
    std::snprintf(count_str, sizeof(count_str), "cells: %u", cells.num_cells);
    ofs << "  <text x=\"" << text_x << "\" y=\"" << (text_y + font_size * 1.2f)
        << "\" font-family=\"Arial, sans-serif\" font-size=\"" << font_size * 0.8f
        << "\" fill=\"black\">" << count_str << "</text>\n";

    // ─── Scale bar (100 µm) ───
    float bar_length = 100.0f;  // 100 µm
    float bar_height = domain_h * 0.005f;
    float bar_x = x_max - domain_w * 0.05f - bar_length;
    float bar_y = y_max - domain_h * 0.04f;

    // Bar rectangle
    ofs << "  <rect x=\"" << bar_x << "\" y=\"" << bar_y
        << "\" width=\"" << bar_length << "\" height=\"" << bar_height
        << "\" fill=\"black\" stroke=\"none\"/>\n";

    // Scale label
    float label_font = font_size * 0.7f;
    ofs << "  <text x=\"" << (bar_x + bar_length * 0.5f) << "\" y=\"" << (bar_y - bar_height * 2.0f)
        << "\" font-family=\"Arial, sans-serif\" font-size=\"" << label_font
        << "\" fill=\"black\" text-anchor=\"middle\">100 &#181;m</text>\n";

    // ─── Close SVG ───
    ofs << "</svg>\n";
    ofs.close();
}
