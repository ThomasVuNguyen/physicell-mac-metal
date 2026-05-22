#ifndef PHYSICELL_GEOMETRY_H
#define PHYSICELL_GEOMETRY_H

// ─────────────────────────────────────────────────────────────────────
// Geometry — cell placement utilities
//
// Ported from PhysiCell modules/PhysiCell_geometry.cpp.
// Uses hexagonal close-packing (offset rows) matching PhysiCell's layout.
//
// All functions return the number of cells placed.
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"
#include "config_parser.h"
#include "microenvironment.h"
#include <string>

/// Fill a circle (2D) with cells at regular hexagonal spacing.
///
/// Matches PhysiCell's fill_circle: hex-packed rows with y_offset = sqrt(3)/2 * spacing,
/// alternating x offsets. Cells are only placed if distance from center < radius - cell_radius.
///
/// @param cells       SoA cell storage
/// @param microenv    Microenvironment (for voxel index computation)
/// @param center_x    X center of the circle
/// @param center_y    Y center of the circle
/// @param radius      Radius of the circle (microns)
/// @param spacing     Distance between cell centers (microns). Typically 0.95 * 2 * cell_radius.
/// @param cell_type   Cell type ID to assign
/// @return Number of cells placed
uint32_t fillCircle(CellData& cells, Microenvironment& microenv,
                    float center_x, float center_y,
                    float radius, float spacing,
                    int cell_type);

/// Fill a rectangle (2D) with cells at regular hexagonal spacing.
///
/// @param cells       SoA cell storage
/// @param microenv    Microenvironment (for voxel index computation)
/// @param x_min       Left bound
/// @param y_min       Bottom bound
/// @param x_max       Right bound
/// @param y_max       Top bound
/// @param spacing     Distance between cell centers
/// @param cell_type   Cell type ID to assign
/// @return Number of cells placed
uint32_t fillRectangle(CellData& cells, Microenvironment& microenv,
                       float x_min, float y_min,
                       float x_max, float y_max,
                       float spacing, int cell_type);

/// Fill a sphere (3D) with cells at regular spacing.
///
/// Extends the 2D hex-packing to 3D layers with z-offset.
///
/// @param cells       SoA cell storage
/// @param microenv    Microenvironment (for voxel index computation)
/// @param cx, cy, cz  Center of the sphere
/// @param radius      Radius of the sphere (microns)
/// @param spacing     Distance between cell centers
/// @param cell_type   Cell type ID to assign
/// @return Number of cells placed
uint32_t fillSphere(CellData& cells, Microenvironment& microenv,
                    float cx, float cy, float cz,
                    float radius, float spacing,
                    int cell_type);

/// Load cells from CSV file.
///
/// Expected format: x, y, z, type_id (one cell per line).
/// Lines starting with '#' or empty lines are skipped.
///
/// @param cells       SoA cell storage
/// @param microenv    Microenvironment (for voxel index computation)
/// @param filepath    Path to CSV file
/// @return Number of cells loaded
uint32_t loadCellsCSV(CellData& cells, Microenvironment& microenv,
                      const std::string& filepath);

/// Fill an annulus (2D ring) with cells at regular hexagonal spacing.
///
/// Matches PhysiCell's fill_annulus: cells placed where
/// inner_radius + cell_radius <= distance <= outer_radius - cell_radius.
///
/// @param cells          SoA cell storage
/// @param microenv       Microenvironment
/// @param center_x       X center
/// @param center_y       Y center
/// @param inner_radius   Inner radius of the annulus
/// @param outer_radius   Outer radius of the annulus
/// @param spacing        Distance between cell centers
/// @param cell_type      Cell type ID to assign
/// @return Number of cells placed
uint32_t fillAnnulus(CellData& cells, Microenvironment& microenv,
                     float center_x, float center_y,
                     float inner_radius, float outer_radius,
                     float spacing, int cell_type);

#endif // PHYSICELL_GEOMETRY_H
