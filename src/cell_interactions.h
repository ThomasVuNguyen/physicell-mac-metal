#ifndef PHYSICELL_CELL_INTERACTIONS_H
#define PHYSICELL_CELL_INTERACTIONS_H

// ─────────────────────────────────────────────────────────────────────
// Cell Interactions — attack, ingest (phagocytose), and fuse
//
// Ported from PhysiCell core/PhysiCell_cell.cpp:
//   Cell::attack_cell, Cell::ingest_cell, Cell::fuse_cell
//
// Operates on SoA CellData arrays. All functions are CPU-only.
//
// REQUIRED CellData fields (to be added to cell_data.h):
//   double* damage            — accumulated damage from attacks
//   double* damage_rate       — rate of damage this cell inflicts (per min)
//   double* damage_threshold  — damage level that triggers apoptosis
//   double* attack_duration   — elapsed time this cell has been attacking
// ─────────────────────────────────────────────────────────────────────

#include "cell_data.h"

/// Attack another cell (e.g., T-cell attacking cancer cell).
///
/// Increases damage on the target by damage_rate[attacker] * dt.
/// Accumulates attack_duration on the attacker.
/// If target's damage exceeds damage_threshold, triggers apoptosis.
///
/// Modeled after PhysiCell Cell::attack_cell():
///   new_damage = phenotype.cell_interactions.attack_damage_rate * dt
///   target.cell_integrity.damage += new_damage
void attackCell(CellData& cells, uint32_t attacker, uint32_t target, double dt);

/// Ingest (phagocytose) another cell (e.g., macrophage eating dead cell).
///
/// Absorbs target's volume compartments into the eater's cytoplasm
/// (fluid + solid), then removes the target from the simulation.
///
/// Modeled after PhysiCell Cell::ingest_cell():
///   - Absorbs fluid, cytoplasmic solid, and nuclear solid into eater
///   - Zeros out target volumes
///   - Recomputes eater geometry (radii from volume)
///   - Removes target via cells.removeCell()
void ingestCell(CellData& cells, uint32_t eater, uint32_t target);

/// Fuse two cells into one (e.g., viral fusion, myoblast fusion).
///
/// Combines all volume compartments. The surviving cell is placed at
/// the volume-weighted centroid of the two cells. The smaller cell
/// is removed from the simulation.
///
/// Modeled after PhysiCell Cell::fuse_cell():
///   - Position = (V_a * pos_a + V_b * pos_b) / (V_a + V_b)
///   - Absorbs all volume compartments (fluid + solid, nuclear + cyto)
///   - Recomputes geometry
///   - Removes the smaller cell
void fuseCells(CellData& cells, uint32_t cell_a, uint32_t cell_b);

// ─── Spring Attachment Forces ────────────────────────────────────────

/// Update spring attachment forces for all cells.
///
/// For each cell with attachments, computes elastic spring forces:
///   F = -k * (distance - rest_length) * unit_direction
/// where rest_length = sum of radii at attachment time (approximated
/// as current R_i + R_j).
///
/// Also handles stochastic detachment: if uniformRandom() < detachment_rate * dt,
/// the attachment is broken and removed from both cells.
///
/// Spring forces are added directly to cell velocities.
void updateAttachmentForces(CellData& cells, double dt);

/// Try to form a spring attachment between two cells.
///
/// Forms an attachment if:
///   1. distance < interaction_distance (relative_max_adhesion_distance * R_sum)
///   2. Both cells have room (attachment_count < MAX_ATTACHMENTS)
///   3. They are not already attached
///   4. Stochastic check: uniformRandom() < attachment_rate * dt
///
/// @return true if attachment was formed
bool tryFormAttachment(CellData& cells, uint32_t cell_a, uint32_t cell_b, double dt);

/// Scan all cell pairs and attempt to form attachments (brute-force O(n²)).
/// For production use, this should be replaced with a spatial hash.
void tryFormAllAttachments(CellData& cells, double dt);

#endif // PHYSICELL_CELL_INTERACTIONS_H
