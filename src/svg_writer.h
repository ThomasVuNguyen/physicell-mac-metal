#ifndef PHYSICELL_SVG_WRITER_H
#define PHYSICELL_SVG_WRITER_H

// ─────────────────────────────────────────────────────────────────────
// SVGWriter — renders simulation snapshots as PhysiCell-style SVG files
//
// Tier 5A: SVG Output
// Produces 2000×2000 px SVG images with cells colored by phenotypic state,
// time labels, and a 100 µm scale bar.
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"
#include "cell_phenotype.h"

#include <string>

class SVGWriter {
public:
    /// Write a single SVG frame of all cells in the simulation.
    ///
    /// @param cells         SoA cell data
    /// @param time          Current simulation time in minutes
    /// @param frame_index   Frame number (used for filename)
    /// @param output_folder Path to output directory (created if needed)
    /// @param x_min, x_max  Domain bounds in x (microns)
    /// @param y_min, y_max  Domain bounds in y (microns)
    void writeFrame(const CellData& cells, double time, int frame_index,
                    const std::string& output_folder,
                    float x_min, float x_max, float y_min, float y_max);
};

#endif // PHYSICELL_SVG_WRITER_H
