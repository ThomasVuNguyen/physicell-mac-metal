// ─────────────────────────────────────────────────────────────────────
// Cell Interactions — attack, ingest, fuse (SoA port)
//
// Ported from PhysiCell core/PhysiCell_cell.cpp.
// All operations use CellData SoA arrays directly.
//
// NOTE: attackCell requires four additional fields in CellData that
// are not yet present (damage, damage_rate, damage_threshold,
// attack_duration). Until those fields are added, attackCell stores
// damage information in local/external tracking. The function
// signature is ready; the caller must provide the damage arrays.
// ─────────────────────────────────────────────────────────────────────

#include "cell_interactions.h"
#include "cell_phenotype.h"  // DEATH_APOPTOSIS, DEATH_NONE
#include "../shaders/types.h"

#include <cmath>
#include <cstdio>
#include <cassert>
#include <algorithm>

// ─── Constants ───────────────────────────────────────────────────────

static constexpr double FOUR_THIRDS_PI_I = 4.188790204786391;

// ─── Helper: recompute radii from total/nuclear volumes ─────────────

static inline void recomputeGeometry(CellData& cells, uint32_t i) {
    double V  = static_cast<double>(cells.total_volume[i]);
    double nV = static_cast<double>(cells.nuclear_volume[i]);
    cells.radius[i]         = static_cast<float>(std::cbrt(V  / FOUR_THIRDS_PI_I));
    cells.nuclear_radius[i] = static_cast<float>(std::cbrt(nV / FOUR_THIRDS_PI_I));
}

// ─── Helper: trigger apoptosis on a cell ─────────────────────────────

static void triggerApoptosis(CellData& cells, uint32_t idx) {
    if (cells.current_death_model[idx] != DEATH_NONE) return;

    cells.current_death_model[idx] = DEATH_APOPTOSIS;
    cells.is_alive[idx] = 0;
    if (cells.apoptosis_duration)
        cells.apoptosis_duration[idx] = 0.0;

    // Apoptosis entry function (matching cell_phenotype.cpp)
    cells.target_fluid_fraction[idx] = 0.0;
    cells.target_solid_cytoplasmic[idx] = 0.0;
    cells.target_solid_nuclear[idx] = 0.0;

    cells.cytoplasmic_biomass_change_rate[idx] = 1.0 / 60.0;
    cells.nuclear_biomass_change_rate[idx] = 0.35 / 60.0;
    cells.fluid_change_rate[idx] =
        cells.unlysed_fluid_change_rate[idx];
    cells.calcification_rate[idx] = 0.0;

    cells.relative_rupture_volume[idx] = 2.0;
    cells.rupture_volume[idx] =
        static_cast<double>(cells.total_volume[idx]) *
        cells.relative_rupture_volume[idx];

    cells.motility_speed[idx] = 0.0f;
}

// ─── attackCell ──────────────────────────────────────────────────────
//
// PhysiCell logic (core/PhysiCell_cell.cpp:1468):
//   new_damage = attack_damage_rate * dt
//   target.damage += new_damage
//   attacker.total_damage_delivered += new_damage
//
// REQUIRED CellData fields (not yet in cell_data.h):
//   double* damage            — accumulated damage on this cell
//   double* damage_rate       — per-minute damage this cell inflicts
//   double* damage_threshold  — damage that triggers death
//   double* attack_duration   — how long this cell has been attacking
//
// Until those fields are added, this function takes external arrays
// as parameters via the CellData struct. If the fields are nullptr
// (i.e. not yet allocated), the function is a no-op but compiles fine.

void attackCell(CellData& cells, uint32_t attacker, uint32_t target, double dt) {
    // Safety checks
    if (attacker == target) return;
    if (attacker >= cells.num_cells || target >= cells.num_cells) return;

    // Don't attack dead or negligible cells
    if (cells.is_alive[target] == 0) return;
    if (cells.total_volume[target] < 1e-15f) return;

    // ── Use death_rate as a proxy for damage rate until proper fields exist ──
    // When the damage/damage_rate/damage_threshold fields are added to CellData,
    // replace this section with proper damage accumulation.
    //
    // Proxy logic: use death_rate[attacker] as attack damage rate.
    // Accumulated "damage" tracked via a simple immediate-kill check:
    //   If the attacker's death_rate * dt roll succeeds, kill the target.
    // This is a simplification; the proper implementation with the
    // dedicated damage fields will use cumulative damage tracking.

    double damage_rate_val = cells.death_rate[attacker]; // proxy
    double new_damage = damage_rate_val * dt;

    // For now, use a probabilistic model: if damage exceeds a single-step
    // threshold (proxy: death_rate is high enough), trigger apoptosis
    // This will be replaced by cumulative damage tracking once the fields exist.
    if (new_damage > 0.0 && cells.death_rate[attacker] > 0.001) {
        // Cumulative damage model placeholder:
        // In the full implementation with damage[] array:
        //   cells.damage[target] += new_damage;
        //   cells.attack_duration[attacker] += dt;
        //   if (cells.damage[target] >= cells.damage_threshold[target])
        //       triggerApoptosis(cells, target);
        triggerApoptosis(cells, target);
    }
}

// ─── ingestCell ──────────────────────────────────────────────────────
//
// PhysiCell logic (core/PhysiCell_cell.cpp:1332):
//   1. Mark target as dead, zero its secretion/uptake
//   2. Absorb fluid → eater's cytoplasmic_fluid
//   3. Absorb solid_cytoplasmic and solid_nuclear → eater's cyto_solid
//   4. Recompute totals and fluid_fraction
//   5. Recompute geometry
//   6. Remove target

void ingestCell(CellData& cells, uint32_t eater, uint32_t target) {
    // Safety checks
    if (eater == target) return;
    if (eater >= cells.num_cells || target >= cells.num_cells) return;

    // Don't ingest a cell that's already been ingested (negligible volume)
    if (cells.total_volume[target] < 1e-15f) return;

    // ── Mark target as dead ──
    cells.is_alive[target] = 0;
    if (cells.current_death_model[target] == DEATH_NONE) {
        cells.current_death_model[target] = DEATH_APOPTOSIS;
    }

    // Zero target secretion/uptake
    if (cells.secretion_rate && cells.n_substrates > 0) {
        for (uint32_t s = 0; s < cells.n_substrates; s++) {
            size_t off = static_cast<size_t>(target) * cells.n_substrates + s;
            cells.secretion_rate[off] = 0.0;
            cells.uptake_rate[off]    = 0.0;
        }
    }

    // ── Absorb volumes (all into eater's cytoplasm, matching PhysiCell) ──

    // Absorb fluid (target's entire fluid → eater's fluid)
    if (cells.fluid) {
        cells.fluid[eater] += cells.fluid[target];
        cells.fluid[target] = 0.0;
    }

    // Absorb cytoplasmic solid
    if (cells.solid_cytoplasmic) {
        cells.solid_cytoplasmic[eater] += cells.solid_cytoplasmic[target];
        cells.solid_cytoplasmic[target] = 0.0;
    }

    // Absorb nuclear solid (into eater's cytoplasmic solid, per PhysiCell)
    if (cells.solid_cytoplasmic && cells.solid_nuclear) {
        cells.solid_cytoplasmic[eater] += cells.solid_nuclear[target];
        cells.solid_nuclear[target] = 0.0;
    }

    // ── Recompute eater's totals ──
    double eater_fluid  = cells.fluid ? cells.fluid[eater] : 0.0;
    double eater_cyto_s = cells.solid_cytoplasmic ?
        cells.solid_cytoplasmic[eater] : 0.0;
    double eater_nuc_s  = cells.solid_nuclear ?
        cells.solid_nuclear[eater] : 0.0;

    // Total volume = fluid + all solid compartments
    double total = eater_fluid + eater_cyto_s + eater_nuc_s;

    cells.total_volume[eater]  = static_cast<float>(total);
    // Nuclear volume unchanged for eater (per PhysiCell: "no change to nuclear volume")
    if (cells.fluid_fraction) {
        cells.fluid_fraction[eater] = eater_fluid / (total + 1e-16);
    }

    // Zero target volumes
    cells.total_volume[target]   = 0.0f;
    cells.nuclear_volume[target] = 0.0f;
    if (cells.fluid_fraction) cells.fluid_fraction[target] = 0.0;

    // ── Recompute eater geometry ──
    recomputeGeometry(cells, eater);

    // ── Stop target motility ──
    cells.motility_speed[target] = 0.0f;

    // ── Remove the target cell ──
    // NOTE: After removeCell, the target index is no longer valid.
    // The caller must be aware that indices may shift (swap-with-last).
    cells.removeCell(target);
}

// ─── fuseCells ───────────────────────────────────────────────────────
//
// PhysiCell logic (core/PhysiCell_cell.cpp:1496):
//   1. Compute volume-weighted centroid for the surviving cell
//   2. Absorb all volume compartments
//   3. Recompute totals, geometry
//   4. Remove the smaller cell

void fuseCells(CellData& cells, uint32_t cell_a, uint32_t cell_b) {
    // Safety checks
    if (cell_a == cell_b) return;
    if (cell_a >= cells.num_cells || cell_b >= cells.num_cells) return;

    // Don't fuse if either has negligible volume
    if (cells.total_volume[cell_a] < 1e-15f) return;
    if (cells.total_volume[cell_b] < 1e-15f) return;

    // ── Determine survivor (larger) and donor (smaller) ──
    double vol_a = static_cast<double>(cells.total_volume[cell_a]);
    double vol_b = static_cast<double>(cells.total_volume[cell_b]);

    uint32_t survivor, donor;
    double vol_surv, vol_donor;
    if (vol_a >= vol_b) {
        survivor = cell_a; donor = cell_b;
        vol_surv = vol_a; vol_donor = vol_b;
    } else {
        survivor = cell_b; donor = cell_a;
        vol_surv = vol_b; vol_donor = vol_a;
    }

    // ── Compute volume-weighted centroid (PhysiCell fuse_cell) ──
    double total_vol = vol_surv + vol_donor;
    double new_x = (vol_surv * cells.position_x[survivor] +
                    vol_donor * cells.position_x[donor]) / total_vol;
    double new_y = (vol_surv * cells.position_y[survivor] +
                    vol_donor * cells.position_y[donor]) / total_vol;
    double new_z = (vol_surv * cells.position_z[survivor] +
                    vol_donor * cells.position_z[donor]) / total_vol;

    cells.position_x[survivor] = static_cast<float>(new_x);
    cells.position_y[survivor] = static_cast<float>(new_y);
    cells.position_z[survivor] = static_cast<float>(new_z);

    // ── Absorb volume compartments ──
    // Unlike ingest, fusion preserves nuclear identity — both nuclear
    // compartments are summed (PhysiCell sums nuclear_fluid and nuclear_solid)

    if (cells.fluid) {
        cells.fluid[survivor] += cells.fluid[donor];
        cells.fluid[donor] = 0.0;
    }
    if (cells.solid_cytoplasmic) {
        cells.solid_cytoplasmic[survivor] += cells.solid_cytoplasmic[donor];
        cells.solid_cytoplasmic[donor] = 0.0;
    }
    if (cells.solid_nuclear) {
        cells.solid_nuclear[survivor] += cells.solid_nuclear[donor];
        cells.solid_nuclear[donor] = 0.0;
    }

    // ── Recompute totals ──
    double surv_fluid = cells.fluid ? cells.fluid[survivor] : 0.0;
    double surv_cyto  = cells.solid_cytoplasmic ?
        cells.solid_cytoplasmic[survivor] : 0.0;
    double surv_nuc   = cells.solid_nuclear ?
        cells.solid_nuclear[survivor] : 0.0;

    double new_total = surv_fluid + surv_cyto + surv_nuc;

    // In PhysiCell, nuclear_volume = nuclear_fluid + nuclear_solid.
    // We compute nuclear_fluid as: nuclear / total * fluid (proportional distribution)
    double old_nuc_vol = static_cast<double>(cells.nuclear_volume[survivor])
                       + static_cast<double>(cells.nuclear_volume[donor]);
    double nuc_frac = old_nuc_vol / (vol_surv + vol_donor + 1e-16);
    double new_nuclear = surv_nuc + nuc_frac * surv_fluid;

    cells.total_volume[survivor]   = static_cast<float>(new_total);
    cells.nuclear_volume[survivor] = static_cast<float>(new_nuclear);
    if (cells.fluid_fraction) {
        cells.fluid_fraction[survivor] = surv_fluid / (new_total + 1e-16);
    }

    // Zero donor volumes
    cells.total_volume[donor]   = 0.0f;
    cells.nuclear_volume[donor] = 0.0f;
    if (cells.fluid_fraction) cells.fluid_fraction[donor] = 0.0;

    // ── Recompute geometry ──
    recomputeGeometry(cells, survivor);

    // ── Mark donor dead and remove ──
    cells.is_alive[donor] = 0;
    cells.motility_speed[donor] = 0.0f;
    cells.removeCell(donor);
}
