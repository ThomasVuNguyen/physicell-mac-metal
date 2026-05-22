// ─────────────────────────────────────────────────────────────────────
// Geometry — cell placement utilities (SoA port)
//
// Ported from PhysiCell modules/PhysiCell_geometry.cpp.
// Uses hexagonal close-packing (offset rows) matching PhysiCell's layout.
//
// Reference: PhysiCell modules/PhysiCell_geometry.cpp:
//   fill_circle, fill_rectangle, fill_annulus, load_cells_csv_v1
// ─────────────────────────────────────────────────────────────────────

#include "geometry.h"
#include "../shaders/types.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

// ─── Constants ───────────────────────────────────────────────────────

static constexpr float SQRT3_OVER_2 = 0.8660254037844386f; // sqrt(3)/2

// ─── Helper: place a single cell at (x, y, z) with type ─────────────

static uint32_t placeCell(CellData& cells, Microenvironment& microenv,
                          float x, float y, float z, int cell_type) {
    uint32_t idx = cells.addCell();
    if (idx == UINT32_MAX) return UINT32_MAX;

    cells.position_x[idx] = x;
    cells.position_y[idx] = y;
    cells.position_z[idx] = z;
    cells.cell_type[idx]  = static_cast<uint32_t>(cell_type);

    // Compute voxel index from position
    cells.voxel_index[idx] = microenv.voxelIndex(x, y, z);

    return idx;
}

// ─── fillCircle ──────────────────────────────────────────────────────
//
// PhysiCell fill_circle (PhysiCell_geometry.cpp:147):
//   Hex-packed rows, alternating x offset.
//   Accept cell if dist² from center ≤ (radius - cell_radius)².

uint32_t fillCircle(CellData& cells, Microenvironment& microenv,
                    float center_x, float center_y,
                    float radius, float spacing,
                    int cell_type) {
    float half_space = 0.5f * spacing;
    float y_offset   = SQRT3_OVER_2 * spacing; // sqrt(3) * half_space

    // Use the spacing as a proxy for cell_radius (spacing ≈ 2 * cell_radius)
    float cell_radius = spacing * 0.5f;
    float r_m_cr_2 = (radius - cell_radius) * (radius - cell_radius);

    float x_min = center_x - radius;
    float x_max = center_x + radius;
    float y_min = center_y - radius;
    float y_max = center_y + radius;

    uint32_t count = 0;
    int row = 0;

    for (float y = y_min + cell_radius; y <= y_max - cell_radius; y += y_offset) {
        float x_start = x_min + cell_radius;
        if (row % 2 == 1) {
            x_start += half_space;
        }

        for (float x = x_start; x <= x_max - cell_radius; x += spacing) {
            float dx = x - center_x;
            float dy = y - center_y;
            float d2 = dx * dx + dy * dy;

            if (d2 <= r_m_cr_2) {
                uint32_t idx = placeCell(cells, microenv, x, y, 0.0f, cell_type);
                if (idx == UINT32_MAX) return count; // full
                count++;
            }
        }
        row++;
    }

    return count;
}

// ─── fillRectangle ───────────────────────────────────────────────────
//
// PhysiCell fill_rectangle (PhysiCell_geometry.cpp:74):
//   Hex-packed rows within the bounding box.

uint32_t fillRectangle(CellData& cells, Microenvironment& microenv,
                       float x_min, float y_min,
                       float x_max, float y_max,
                       float spacing, int cell_type) {
    float half_space = 0.5f * spacing;
    float y_offset   = SQRT3_OVER_2 * spacing;
    float cell_radius = spacing * 0.5f;

    uint32_t count = 0;
    int row = 0;

    for (float y = y_min + cell_radius; y <= y_max - cell_radius; y += y_offset) {
        float x_start = x_min + cell_radius;
        if (row % 2 == 1) {
            x_start += half_space;
        }

        for (float x = x_start; x <= x_max - cell_radius; x += spacing) {
            uint32_t idx = placeCell(cells, microenv, x, y, 0.0f, cell_type);
            if (idx == UINT32_MAX) return count;
            count++;
        }
        row++;
    }

    return count;
}

// ─── fillSphere ──────────────────────────────────────────────────────
//
// 3D extension: hex-packed layers with z-offset = sqrt(6)/3 * spacing.

uint32_t fillSphere(CellData& cells, Microenvironment& microenv,
                    float cx, float cy, float cz,
                    float radius, float spacing,
                    int cell_type) {
    float half_space = 0.5f * spacing;
    float y_offset   = SQRT3_OVER_2 * spacing;
    float z_offset   = std::sqrt(6.0f) / 3.0f * spacing; // FCC z-layer spacing

    float cell_radius = spacing * 0.5f;
    float r_m_cr_2 = (radius - cell_radius) * (radius - cell_radius);

    uint32_t count = 0;
    int layer = 0;

    for (float z = cz - radius + cell_radius; z <= cz + radius - cell_radius; z += z_offset) {
        int row = 0;

        // Offset even/odd layers
        float layer_x_off = (layer % 2 == 1) ? half_space * 0.5f : 0.0f;
        float layer_y_off = (layer % 2 == 1) ? y_offset / 3.0f : 0.0f;

        for (float y = cy - radius + cell_radius + layer_y_off;
             y <= cy + radius - cell_radius; y += y_offset) {

            float x_start = cx - radius + cell_radius + layer_x_off;
            if (row % 2 == 1) {
                x_start += half_space;
            }

            for (float x = x_start; x <= cx + radius - cell_radius; x += spacing) {
                float dx = x - cx;
                float dy = y - cy;
                float dz = z - cz;
                float d2 = dx * dx + dy * dy + dz * dz;

                if (d2 <= r_m_cr_2) {
                    uint32_t idx = placeCell(cells, microenv, x, y, z, cell_type);
                    if (idx == UINT32_MAX) return count;
                    count++;
                }
            }
            row++;
        }
        layer++;
    }

    return count;
}

// ─── loadCellsCSV ────────────────────────────────────────────────────
//
// PhysiCell load_cells_csv_v1 (PhysiCell_geometry.cpp:295):
//   Format: x, y, z, type_id
//   Lines starting with '#' or empty lines are skipped.

uint32_t loadCellsCSV(CellData& cells, Microenvironment& microenv,
                      const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        printf("  ERROR: Could not open CSV file: %s\n", filepath.c_str());
        return 0;
    }

    printf("  Loading cells from CSV: %s ...\n", filepath.c_str());

    uint32_t count = 0;
    std::string line;

    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Skip comment lines
        if (line[0] == '#') continue;

        // Skip lines that start with non-numeric characters (headers)
        char first = line[0];
        if (!std::isdigit(first) && first != '-' && first != '+' && first != '.') {
            continue;
        }

        // Parse CSV: x, y, z, type_id
        std::stringstream ss(line);
        std::string token;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        int type_id = 0;

        if (std::getline(ss, token, ',')) x = std::stof(token);
        else continue;
        if (std::getline(ss, token, ',')) y = std::stof(token);
        else continue;
        if (std::getline(ss, token, ',')) z = std::stof(token);
        else continue;
        if (std::getline(ss, token, ',')) type_id = std::stoi(token);
        else continue;

        uint32_t idx = placeCell(cells, microenv, x, y, z, type_id);
        if (idx == UINT32_MAX) {
            printf("  WARNING: Max cells reached while loading CSV (loaded %u)\n", count);
            break;
        }

        count++;
    }

    file.close();
    printf("  Loaded %u cells from CSV\n", count);
    return count;
}

// ─── fillAnnulus ─────────────────────────────────────────────────────
//
// PhysiCell fill_annulus (PhysiCell_geometry.cpp:203):
//   Like fillCircle but with inner and outer radius constraint.
//   Accept cell if (inner_radius + cell_radius)² ≤ dist² ≤ (outer_radius - cell_radius)².

uint32_t fillAnnulus(CellData& cells, Microenvironment& microenv,
                     float center_x, float center_y,
                     float inner_radius, float outer_radius,
                     float spacing, int cell_type) {
    float half_space = 0.5f * spacing;
    float y_offset   = SQRT3_OVER_2 * spacing;
    float cell_radius = spacing * 0.5f;

    float ro_m_cr_2 = (outer_radius - cell_radius) * (outer_radius - cell_radius);
    float ri_p_cr_2 = (inner_radius + cell_radius) * (inner_radius + cell_radius);

    float x_min = center_x - outer_radius;
    float x_max = center_x + outer_radius;
    float y_min = center_y - outer_radius;
    float y_max = center_y + outer_radius;

    uint32_t count = 0;
    int row = 0;

    for (float y = y_min + cell_radius; y <= y_max - cell_radius; y += y_offset) {
        float x_start = x_min + cell_radius;
        if (row % 2 == 1) {
            x_start += half_space;
        }

        for (float x = x_start; x <= x_max - cell_radius; x += spacing) {
            float dx = x - center_x;
            float dy = y - center_y;
            float d2 = dx * dx + dy * dy;

            if (d2 <= ro_m_cr_2 && d2 >= ri_p_cr_2) {
                uint32_t idx = placeCell(cells, microenv, x, y, 0.0f, cell_type);
                if (idx == UINT32_MAX) return count;
                count++;
            }
        }
        row++;
    }

    return count;
}
