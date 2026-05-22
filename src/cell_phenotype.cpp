// ─────────────────────────────────────────────────────────────────────
// Cell phenotype updates — CPU-side, float64 precision
//
// Tier 1 Feature Parity: Full PhysiCell phenotype models ported to SoA.
//
// Models implemented:
//   - Multi-compartment volume ODE (standard_volume_update_function)
//   - Phase-based cell cycle (live, Ki67 basic, Ki67 advanced,
//     flow cytometry basic, cycling-quiescent)
//   - Full death models (apoptosis + necrosis with swelling/lysis)
//   - Volume-scaled implicit secretion/uptake
//   - Cell division with proper compartment halving
//
// Reference: PhysiCell core/PhysiCell_phenotype.cpp,
//            core/PhysiCell_standard_models.cpp
// ─────────────────────────────────────────────────────────────────────

#include "cell_phenotype.h"
#include "cell_definitions.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>

// Thread-local RNG for stochastic phenotype decisions.
// Using thread_local so this is safe with dispatch_apply / OpenMP.
// Fixed seed (42) for reproducible results; use std::random_device{}() for non-deterministic.
static thread_local std::mt19937 tl_rng{42};

static inline double uniformRandom() {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(tl_rng);
}

static inline double normalRandom(double mean, double sd) {
    std::normal_distribution<double> dist(mean, sd);
    return dist(tl_rng);
}

// ─── Constants ───────────────────────────────────────────────────────

static constexpr double PI_D = 3.14159265358979323846;
static constexpr double FOUR_THIRDS_PI = 4.188790204786391;

// ─── Volume ODE (PhysiCell standard_volume_update_function) ──────────
//
// Ported from PhysiCell_standard_models.cpp lines 518-560.
// This is the exact multi-compartment volume model.

void updateVolume(CellData& cells, uint32_t i, double dt) {
    double total = static_cast<double>(cells.total_volume[i]);
    double V_fluid = cells.fluid[i];
    double nuclear_solid = cells.solid_nuclear[i];
    double cyto_solid = cells.solid_cytoplasmic[i];

    // Step 1: Fluid volume
    // dV_fluid/dt = fluid_change_rate * (target_fluid_fraction * V_total - V_fluid)
    V_fluid += dt * cells.fluid_change_rate[i] *
        (cells.target_fluid_fraction[i] * total - V_fluid);
    if (V_fluid < 0.0) V_fluid = 0.0;

    // Step 2: Distribute fluid to nuclear and cytoplasmic compartments
    double nuclear = static_cast<double>(cells.nuclear_volume[i]);
    double nuclear_fluid = (nuclear / (total + 1e-16)) * V_fluid;
    double cyto_fluid = V_fluid - nuclear_fluid;

    // Step 3: Nuclear solid biomass
    // dV_ns/dt = nuclear_biomass_change_rate * (target_solid_nuclear - V_ns)
    nuclear_solid += dt * cells.nuclear_biomass_change_rate[i] *
        (cells.target_solid_nuclear[i] - nuclear_solid);
    if (nuclear_solid < 0.0) nuclear_solid = 0.0;

    // Step 4: Cytoplasmic solid biomass
    // In PhysiCell, target_solid_cytoplasmic is dynamically set as:
    //   target_sc = target_cytoplasmic_to_nuclear_ratio * target_solid_nuclear
    // We use the stored target_solid_cytoplasmic which should already be set.
    // dV_cs/dt = cytoplasmic_biomass_change_rate * (target_solid_cyto - V_cs)
    cyto_solid += dt * cells.cytoplasmic_biomass_change_rate[i] *
        (cells.target_solid_cytoplasmic[i] - cyto_solid);
    if (cyto_solid < 0.0) cyto_solid = 0.0;

    // Step 5: Calcification
    double calc_frac = cells.calcified_fraction[i];
    calc_frac += dt * cells.calcification_rate[i] * (1.0 - calc_frac);

    // Step 6: Reconstruct compartment totals
    nuclear = nuclear_solid + nuclear_fluid;
    double cytoplasmic = cyto_solid + cyto_fluid;
    total = cytoplasmic + nuclear;
    double fluid_fraction = V_fluid / (total + 1e-16);

    // Step 7: Write back to SoA arrays
    cells.total_volume[i]          = static_cast<float>(total);
    cells.nuclear_volume[i]        = static_cast<float>(nuclear);
    cells.fluid[i]                 = V_fluid;
    cells.solid_nuclear[i]         = nuclear_solid;
    cells.solid_cytoplasmic[i]     = cyto_solid;
    cells.fluid_fraction[i]        = fluid_fraction;
    cells.calcified_fraction[i]    = calc_frac;

    // Step 8: Update geometry (radius from volume)
    double r  = std::cbrt(total / FOUR_THIRDS_PI);
    double nr = std::cbrt(nuclear / FOUR_THIRDS_PI);
    cells.radius[i]         = static_cast<float>(r);
    cells.nuclear_radius[i] = static_cast<float>(nr);
}

// ─── Cell cycle ──────────────────────────────────────────────────────
//
// Phase-based cycle model matching PhysiCell's Cycle_Model::advance_model.
// Supports:
//   - Live model (code=5): single phase, division at exit, stochastic
//   - Ki67 basic (code=1): two phases, Ki67- stochastic, Ki67+ fixed duration
//   - Ki67 advanced (code=0): three phases, Ki67- stochastic, Ki67+ pre-M fixed,
//     Ki67+ post-M fixed, division at phase 1 exit
//   - Flow cytometry basic (code=2) / separated (code=6):
//     G0/G1 stochastic → S fixed → G2 fixed → M fixed (division at exit)
//   - Cycling-quiescent (code=7): Q stochastic → Cycling fixed (division at exit)

bool updateCellCycle(CellData& cells, uint32_t i, double dt,
                     const float* density, const GridParams& grid) {
    // Dead cells don't cycle
    if (cells.current_death_model[i] != DEATH_NONE) return false;

    uint32_t model = cells.cycle_model_code[i];
    uint32_t phase = cells.current_phase[i];

    // Accumulate elapsed time
    cells.elapsed_time_in_phase[i] += dt;

    // ── O₂-dependent cycle rate modulation ──
    // From PhysiCell's update_cell_and_death_parameters_O2_based:
    //   multiplier = 1.0 if pO2 >= saturation
    //   multiplier = 0.0 if pO2 < threshold (5 mmHg)
    //   linear interpolation between
    // Heterogeneity sample sets o2_proliferation_saturation = 38 (not 160!)
    double o2_multiplier = 1.0;
    double o2_prolif_threshold = cells.o2_proliferation_threshold[i];   // mmHg
    double o2_prolif_saturation = cells.o2_proliferation_saturation[i]; // mmHg

    if (density && grid.n_substrates > 0) {
        uint32_t vi = cells.voxel_index[i];
        uint32_t total_voxels = grid.nx * grid.ny * grid.nz;
        if (vi < total_voxels) {
            double pO2 = static_cast<double>(density[vi]); // substrate 0 = oxygen
            if (pO2 < o2_prolif_threshold) {
                o2_multiplier = 0.0;
            } else if (pO2 < o2_prolif_saturation) {
                o2_multiplier = (pO2 - o2_prolif_threshold)
                              / (o2_prolif_saturation - o2_prolif_threshold);
            }
            // else: o2_multiplier stays 1.0
        }
    }

    if (model == CYCLE_LIVE) {
        // ── Live model: single phase 0→0, division at phase exit ──
        // PhysiCell order: O₂ modulation first, then oncoprotein multiplication
        double rate = cells.transition_rate_01[i];

        // Step 1: O₂-dependent modulation (from update_cell_and_death_parameters_O2_based)
        rate *= o2_multiplier;

        // Step 2: Oncoprotein modulation (from tumor_cell_phenotype_with_oncoprotein)
        // PhysiCell: rate *= oncoprotein (direct multiplication, not offset formula)
        double onco = static_cast<double>(cells.oncoprotein[i]);
        rate *= onco;
        if (rate < 0.0) rate = 0.0;

        // NOTE: PhysiCell's standard live cycle model has no volume gate.
        // Division is purely stochastic based on the transition rate.

        // Stochastic transition
        if (uniformRandom() < rate * dt) {
            // Entry function: double the target solid volumes for growth
            cells.target_solid_cytoplasmic[i] *= 2.0;
            cells.target_solid_nuclear[i] *= 2.0;
            cells.elapsed_time_in_phase[i] = 0.0;
            return true;  // flag for division
        }

    } else if (model == CYCLE_KI67_BASIC) {
        // ── Ki67 basic: phase 0 = Ki67-, phase 1 = Ki67+ ──

        if (phase == PHASE_KI67_NEG) {
            // Phase 0→1: stochastic transition
            double rate = cells.transition_rate_01[i]; // 1/(4.59*60)

            // O₂-dependent modulation
            rate *= o2_multiplier;

            if (uniformRandom() < rate * dt) {
                // Transition to Ki67+ phase
                cells.current_phase[i] = PHASE_KI67_POS;
                cells.elapsed_time_in_phase[i] = 0.0;
                // Entry function: double target solid volumes
                cells.target_solid_cytoplasmic[i] *= 2.0;
                cells.target_solid_nuclear[i] *= 2.0;
            }

        } else if (phase == PHASE_KI67_POS) {
            // Phase 1→0: fixed duration transition
            double rate = cells.transition_rate_10[i]; // 1/(15.5*60)
            double duration = 1.0 / rate;  // 15.5 * 60 = 930 min

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                // Division at phase exit (phase 1 has division_at_phase_exit)
                cells.current_phase[i] = PHASE_KI67_NEG;
                cells.elapsed_time_in_phase[i] = 0.0;
                return true;  // flag for division
            }
        }
    } else if (model == CYCLE_KI67_ADVANCED) {
        // ── Ki67 advanced: 3 phases ──
        // Phase 0: Ki67- (quiescent), stochastic transition to phase 1
        // Phase 1: Ki67+ premitotic, fixed duration, division_at_phase_exit
        // Phase 2: Ki67+ postmitotic, fixed duration, transition to phase 0

        if (phase == 0) {
            // Phase 0→1: stochastic, O₂-modulated
            double rate = cells.transition_rate_01[i]; // 1/(3.62*60)
            rate *= o2_multiplier;

            if (uniformRandom() < rate * dt) {
                cells.current_phase[i] = 1;
                cells.elapsed_time_in_phase[i] = 0.0;
                // Entry function: double target solid volumes for growth
                cells.target_solid_cytoplasmic[i] *= 2.0;
                cells.target_solid_nuclear[i] *= 2.0;
            }

        } else if (phase == 1) {
            // Phase 1→2: fixed duration
            double rate = cells.transition_rate_10[i]; // 1/(13*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                // Division at phase exit
                cells.current_phase[i] = 2;
                cells.elapsed_time_in_phase[i] = 0.0;
                return true;  // flag for division
            }

        } else if (phase == 2) {
            // Phase 2→0: fixed duration
            double rate = cells.transition_rate_20[i]; // 1/(2.5*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                cells.current_phase[i] = 0;
                cells.elapsed_time_in_phase[i] = 0.0;
            }
        }

    } else if (model == CYCLE_FLOW_CYTO || model == CYCLE_FLOW_CYTO_SEP) {
        // ── Flow cytometry basic: 4 phases ──
        // Phase 0: G0/G1, stochastic → S
        // Phase 1: S, fixed duration → G2  (entry: double volumes)
        // Phase 2: G2, fixed duration → M
        // Phase 3: M, fixed duration → G0/G1  (division_at_phase_exit)

        if (phase == 0) {
            // G0/G1 → S: stochastic, O₂-modulated
            double rate = cells.transition_rate_01[i]; // 0.00324
            rate *= o2_multiplier;

            if (uniformRandom() < rate * dt) {
                cells.current_phase[i] = 1;
                cells.elapsed_time_in_phase[i] = 0.0;
                // Entry function for S phase: double target solid volumes
                cells.target_solid_cytoplasmic[i] *= 2.0;
                cells.target_solid_nuclear[i] *= 2.0;
            }

        } else if (phase == 1) {
            // S → G2: fixed duration
            double rate = cells.transition_rate_10[i]; // 1/(8*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                cells.current_phase[i] = 2;
                cells.elapsed_time_in_phase[i] = 0.0;
            }

        } else if (phase == 2) {
            // G2 → M: fixed duration
            double rate = cells.transition_rate_23[i]; // 1/(4*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                cells.current_phase[i] = 3;
                cells.elapsed_time_in_phase[i] = 0.0;
            }

        } else if (phase == 3) {
            // M → G0/G1: fixed duration, division at phase exit
            double rate = cells.transition_rate_30[i]; // 1/(1*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                cells.current_phase[i] = 0;
                cells.elapsed_time_in_phase[i] = 0.0;
                return true;  // flag for division
            }
        }

    } else if (model == CYCLE_CYCLING_QUIESCENT) {
        // ── Cycling-quiescent: 2 phases ──
        // Phase 0: Quiescent, stochastic → Cycling
        // Phase 1: Cycling, fixed duration → Quiescent (division_at_phase_exit)

        if (phase == 0) {
            // Q → C: stochastic, O₂-modulated
            double rate = cells.transition_rate_01[i]; // 1/(4.59*60)
            rate *= o2_multiplier;

            if (uniformRandom() < rate * dt) {
                cells.current_phase[i] = 1;
                cells.elapsed_time_in_phase[i] = 0.0;
                // Entry function: double target solid volumes
                cells.target_solid_cytoplasmic[i] *= 2.0;
                cells.target_solid_nuclear[i] *= 2.0;
            }

        } else if (phase == 1) {
            // C → Q: fixed duration, division at phase exit
            double rate = cells.transition_rate_10[i]; // 1/(15.5*60)
            double duration = 1.0 / rate;

            if (cells.elapsed_time_in_phase[i] >= duration - 0.5 * dt) {
                cells.current_phase[i] = 0;
                cells.elapsed_time_in_phase[i] = 0.0;
                return true;  // flag for division
            }
        }
    }

    return false;
}

// ─── Death check ─────────────────────────────────────────────────────
//
// Checks for stochastic apoptosis and oxygen-threshold necrosis.
// When death is triggered, sets up the appropriate death model parameters
// on the volume system (matching PhysiCell entry functions).

void updateDeath(CellData& cells, uint32_t i, double dt,
                 const float* density, const GridParams& grid) {
    // Only alive cells with no active death model can die
    if (cells.current_death_model[i] != DEATH_NONE) return;

    // ── Stochastic apoptosis ──
    double apop_rate = cells.death_rate[i];
    if (uniformRandom() < apop_rate * dt) {
        // Trigger apoptosis
        cells.current_death_model[i] = DEATH_APOPTOSIS;
        cells.is_alive[i] = 0;
        cells.apoptosis_duration[i] = 0.0;

        // Apoptosis entry function (from standard_apoptosis_entry_function):
        // Target volumes → 0 (cell wants to shrink)
        cells.target_fluid_fraction[i] = 0.0;
        cells.target_solid_cytoplasmic[i] = 0.0;
        cells.target_solid_nuclear[i] = 0.0;

        // Change rate parameters to apoptosis death parameters
        cells.cytoplasmic_biomass_change_rate[i] = 1.0 / 60.0;
        cells.nuclear_biomass_change_rate[i] = 0.35 / 60.0;
        cells.fluid_change_rate[i] = cells.unlysed_fluid_change_rate[i]; // 3.0/60.0
        cells.calcification_rate[i] = 0.0;

        // Set rupture volume
        cells.relative_rupture_volume[i] = 2.0;
        cells.rupture_volume[i] =
            static_cast<double>(cells.total_volume[i]) *
            cells.relative_rupture_volume[i];

        // Turn off motility
        cells.motility_speed[i] = 0.0f;

        return;
    }

    // ── O₂-dependent necrosis (from update_cell_and_death_parameters_O2_based) ──
    // PhysiCell formula:
    //   multiplier = (threshold - pO2) / (threshold - o2_necrosis_max)
    //   If pO2 < o2_necrosis_max → multiplier = 1.0
    //   If pO2 >= threshold → multiplier = 0.0 (no necrosis)
    //   necrosis_rate = multiplier * max_necrosis_rate
    double necrosis_threshold = cells.o2_necrosis_threshold[i]; // mmHg
    double necrosis_max = cells.o2_necrosis_max[i];             // mmHg
    double max_necrosis_rate_val = cells.max_necrosis_rate[i];  // 1/min

    if (density && grid.n_substrates > 0) {
        uint32_t vi = cells.voxel_index[i];
        uint32_t total_voxels = grid.nx * grid.ny * grid.nz;

        if (vi < total_voxels) {
            double pO2 = static_cast<double>(density[vi]); // substrate 0 = oxygen

            if (pO2 < necrosis_threshold) {
                double multiplier = (necrosis_threshold - pO2)
                                  / (necrosis_threshold - necrosis_max);
                if (pO2 < necrosis_max) {
                    multiplier = 1.0;
                }
                double effective_rate = max_necrosis_rate_val * multiplier;

                if (uniformRandom() < effective_rate * dt) {
                    // Trigger necrosis
                    cells.current_death_model[i] = DEATH_NECROSIS;
                    cells.is_alive[i] = 0;
                    cells.lysed[i] = 0;

                    // Necrosis entry function (from standard_necrosis_entry_function):
                    cells.target_fluid_fraction[i] = 1.0;
                    cells.target_solid_cytoplasmic[i] = 0.0;
                    cells.target_solid_nuclear[i] = 0.0;

                    cells.cytoplasmic_biomass_change_rate[i] = 0.0032 / 60.0;
                    cells.nuclear_biomass_change_rate[i] = 0.013 / 60.0;
                    cells.fluid_change_rate[i] = 0.67 / 60.0;
                    cells.calcification_rate[i] = 0.0042 / 60.0;

                    cells.relative_rupture_volume[i] = 2.0;
                    cells.rupture_volume[i] =
                        static_cast<double>(cells.total_volume[i]) *
                        cells.relative_rupture_volume[i];

                    cells.motility_speed[i] = 0.0f;

                    return;
                }
            }
            // If pO2 >= threshold, necrosis is not triggered (rate = 0)
        }
    }
}

// ─── Death model advancement ─────────────────────────────────────────
//
// Advances the death model for dead cells:
//   - Apoptosis: shrinks via volume ODE, removal after fixed duration (8.6 hrs)
//   - Necrosis: swelling phase → lysis at rupture_volume → shrinkage phase
// Returns true if cell should be removed.

bool advanceDeathModel(CellData& cells, uint32_t i, double dt) {
    uint32_t death_model = cells.current_death_model[i];

    if (death_model == DEATH_APOPTOSIS) {
        // ── Apoptosis: deterministic removal after 8.6 hours (516 min) ──
        cells.apoptosis_duration[i] += dt;

        // Volume still updates (handled by updateVolume called separately)

        // Fixed duration: 1/(1/(8.6*60)) = 516 min
        static constexpr double APOPTOSIS_DURATION = 8.6 * 60.0; // 516 min
        if (cells.apoptosis_duration[i] >= APOPTOSIS_DURATION - 0.5 * dt) {
            return true;  // remove cell
        }

        // Also remove if volume is negligible
        if (cells.total_volume[i] < 1.0f) {
            return true;
        }

    } else if (death_model == DEATH_NECROSIS) {
        if (cells.lysed[i] == 0) {
            // ── Phase 0: Necrotic swelling ──
            // Arrest function: stay in swelling until volume > rupture_volume
            double V = static_cast<double>(cells.total_volume[i]);
            if (V >= cells.rupture_volume[i]) {
                // Transition to lysed phase
                cells.lysed[i] = 1;

                // Lysis entry function (from standard_lysis_entry_function):
                cells.target_fluid_fraction[i] = 0.0;
                cells.target_solid_cytoplasmic[i] = 0.0;
                cells.target_solid_nuclear[i] = 0.0;

                // Switch to lysed fluid change rate
                cells.fluid_change_rate[i] = 0.050 / 60.0;

                // Prevent further bursting
                cells.relative_rupture_volume[i] = 9e99;
                cells.rupture_volume[i] = V * 9e99;

                // Reset elapsed time for the lysed phase
                cells.elapsed_time_in_phase[i] = 0.0;
            }
        } else {
            // ── Phase 1: Necrotic lysed ──
            // Deterministic removal after 60 days
            cells.elapsed_time_in_phase[i] += dt;

            static constexpr double NECROTIC_LYSED_DURATION = 60.0 * 24.0 * 60.0; // 86400 min
            if (cells.elapsed_time_in_phase[i] >= NECROTIC_LYSED_DURATION - 0.5 * dt) {
                return true;  // remove cell
            }

            // Also remove if volume is negligible
            if (cells.total_volume[i] < 1.0f) {
                return true;
            }
        }
    }

    return false;
}

// ─── Secretion / Uptake ──────────────────────────────────────────────
//
// Volume-scaled implicit solve from PhysiCell/BioFVM:
//   internal_constant = dt * V_cell / V_voxel
//   temp1 = internal_constant * sec_rate * sat_density
//   temp2 = 1 + internal_constant * (sec_rate + upt_rate)
//   rho_new = (rho + temp1) / temp2

void processSecretion(CellData& cells, float* density,
                      const GridParams& grid, uint32_t i, double dt)
{
    if (!density) return;
    if (grid.n_substrates == 0) return;

    uint32_t vi = cells.voxel_index[i];
    uint32_t total_voxels = grid.nx * grid.ny * grid.nz;
    if (vi >= total_voxels) return;

    double V_cell  = static_cast<double>(cells.total_volume[i]);
    double V_voxel = static_cast<double>(grid.dx) *
                     static_cast<double>(grid.dy) *
                     static_cast<double>(grid.dz);
    double internal_constant = dt * V_cell / V_voxel;

    // Currently: secretion_rate and uptake_rate are stored per-cell
    // (single substrate). Access by cell index i.
    // TODO: extend to multi-substrate with per-cell×per-substrate arrays.
    double sec_rate = cells.secretion_rate[i];
    double upt_rate = cells.uptake_rate[i];

    // Density for substrate 0 at this voxel
    double rho = static_cast<double>(density[vi]);

    // Saturation density = 1.0 (simplified; could be per-substrate)
    double sat_density = 1.0;

    // Implicit volume-scaled solve (PhysiCell BioFVM formula)
    double temp1 = internal_constant * sec_rate * sat_density;
    double temp2 = 1.0 + internal_constant * (sec_rate + upt_rate);
    rho = (rho + temp1) / temp2;
    if (rho < 0.0) rho = 0.0;

    density[vi] = static_cast<float>(rho);
}

// ─── Cell division ───────────────────────────────────────────────────
//
// PhysiCell Volume::divide(): multiply_by_ratio(0.5) for all compartments.
// Creates daughter cell inheriting parent properties.

uint32_t divideCell(CellData& cells, uint32_t parent) {
    uint32_t daughter = cells.addCell();
    if (daughter == UINT32_MAX) return UINT32_MAX;

    // ── Halve all volume compartments (Volume::divide) ──
    auto halve_gpu_f = [&](float* arr) {
        float v = arr[parent] * 0.5f;
        arr[parent] = v;
        arr[daughter] = v;
    };
    auto halve_cpu_d = [&](double* arr) {
        if (!arr) return;
        double v = arr[parent] * 0.5;
        arr[parent] = v;
        arr[daughter] = v;
    };

    halve_gpu_f(cells.total_volume);
    halve_gpu_f(cells.nuclear_volume);
    halve_cpu_d(cells.fluid);
    halve_cpu_d(cells.solid_cytoplasmic);
    halve_cpu_d(cells.solid_nuclear);

    // Rupture volume also halves (like PhysiCell multiply_by_ratio)
    halve_cpu_d(cells.rupture_volume);

    // Target solid volumes halve too (they were doubled on entry to cycling)
    halve_cpu_d(cells.target_solid_nuclear);
    halve_cpu_d(cells.target_solid_cytoplasmic);

    // Recompute radii for both cells
    auto update_radius = [&](uint32_t idx) {
        double V  = static_cast<double>(cells.total_volume[idx]);
        double nV = static_cast<double>(cells.nuclear_volume[idx]);
        cells.radius[idx]         = static_cast<float>(std::cbrt(V / FOUR_THIRDS_PI));
        cells.nuclear_radius[idx] = static_cast<float>(std::cbrt(nV / FOUR_THIRDS_PI));
    };
    update_radius(parent);
    update_radius(daughter);

    // ── Position: offset daughter slightly from parent ──
    double theta = uniformRandom() * 2.0 * PI_D;
    double offset = static_cast<double>(cells.radius[parent]) * 0.5;

    cells.position_x[daughter] = cells.position_x[parent] + static_cast<float>(offset * std::cos(theta));
    cells.position_y[daughter] = cells.position_y[parent] + static_cast<float>(offset * std::sin(theta));
    cells.position_z[daughter] = cells.position_z[parent];  // 2D: keep z

    // ── Inherit cell type and mechanics ──
    cells.cell_type[daughter]     = cells.cell_type[parent];
    cells.cell_cell_repulsion[daughter] = cells.cell_cell_repulsion[parent];
    cells.cell_cell_adhesion[daughter]  = cells.cell_cell_adhesion[parent];
    cells.relative_max_adhesion_distance[daughter] = cells.relative_max_adhesion_distance[parent];
    cells.motility_speed[daughter] = cells.motility_speed[parent];
    cells.motility_bias[daughter]  = cells.motility_bias[parent];

    // ── Inherit oncoprotein with small noise ──
    double parent_onco = static_cast<double>(cells.oncoprotein[parent]);
    double noise = normalRandom(0.0, 0.02 * parent_onco);
    double daughter_onco = std::max(0.0, parent_onco + noise);
    cells.oncoprotein[daughter] = static_cast<float>(daughter_onco);
    double parent_new_onco = std::max(0.0, parent_onco - noise);
    cells.oncoprotein[parent] = static_cast<float>(parent_new_onco);

    // ── Reset cycle state ──
    cells.current_phase[parent]  = 0;
    cells.current_phase[daughter] = 0;
    cells.elapsed_time_in_phase[parent]  = 0.0;
    cells.elapsed_time_in_phase[daughter] = 0.0;

    // ── Inherit CPU-only fields ──
    // Cycle fields
    cells.birth_rate[daughter]        = cells.birth_rate[parent];
    cells.death_rate[daughter]        = cells.death_rate[parent];
    cells.target_volume[daughter]     = cells.target_volume[parent];
    cells.volume_change_rate[daughter] = cells.volume_change_rate[parent];
    cells.cycle_model_code[daughter]  = cells.cycle_model_code[parent];
    cells.num_phases[daughter]        = cells.num_phases[parent];
    cells.transition_rate_01[daughter] = cells.transition_rate_01[parent];
    cells.transition_rate_10[daughter] = cells.transition_rate_10[parent];
    if (cells.transition_rate_20)
        cells.transition_rate_20[daughter] = cells.transition_rate_20[parent];
    if (cells.transition_rate_23)
        cells.transition_rate_23[daughter] = cells.transition_rate_23[parent];
    if (cells.transition_rate_30)
        cells.transition_rate_30[daughter] = cells.transition_rate_30[parent];
    cells.phase_duration_fixed[daughter] = cells.phase_duration_fixed[parent];

    // Volume parameters (rates, targets already set by halving above)
    cells.fluid_fraction[daughter]        = cells.fluid_fraction[parent];
    cells.cytoplasmic_biomass_change_rate[daughter] = cells.cytoplasmic_biomass_change_rate[parent];
    cells.nuclear_biomass_change_rate[daughter]     = cells.nuclear_biomass_change_rate[parent];
    cells.fluid_change_rate[daughter]     = cells.fluid_change_rate[parent];
    cells.calcification_rate[daughter]    = cells.calcification_rate[parent];
    cells.calcified_fraction[daughter]    = cells.calcified_fraction[parent];
    cells.target_fluid_fraction[daughter] = cells.target_fluid_fraction[parent];
    cells.relative_rupture_volume[daughter] = cells.relative_rupture_volume[parent];

    // Death fields
    cells.necrosis_rate[daughter]        = cells.necrosis_rate[parent];
    cells.necrosis_threshold[daughter]   = cells.necrosis_threshold[parent];
    cells.current_death_model[daughter]  = DEATH_NONE;
    cells.lysed[daughter]               = 0;
    cells.apoptosis_duration[daughter]   = 0.0;
    cells.unlysed_fluid_change_rate[daughter] = cells.unlysed_fluid_change_rate[parent];
    cells.lysed_fluid_change_rate[daughter]   = cells.lysed_fluid_change_rate[parent];

    // O2/necrosis per-cell parameters
    cells.o2_proliferation_saturation[daughter] = cells.o2_proliferation_saturation[parent];
    cells.o2_proliferation_threshold[daughter]  = cells.o2_proliferation_threshold[parent];
    cells.o2_necrosis_threshold[daughter]       = cells.o2_necrosis_threshold[parent];
    cells.o2_necrosis_max[daughter]             = cells.o2_necrosis_max[parent];
    cells.max_necrosis_rate[daughter]           = cells.max_necrosis_rate[parent];

    // Interaction / integrity (inherit rates, reset state)
    cells.damage[daughter]              = 0.0;   // daughter starts undamaged
    cells.damage_rate[daughter]         = cells.damage_rate[parent];
    cells.damage_repair_rate[daughter]  = cells.damage_repair_rate[parent];
    cells.attack_elapsed[daughter]      = 0.0;
    cells.attacking_cell[daughter]      = UINT32_MAX;

    // ── Inherit secretion/uptake rates ──
    if (cells.secretion_rate && cells.n_substrates > 0) {
        for (uint32_t s = 0; s < cells.n_substrates; s++) {
            size_t p_off = static_cast<size_t>(parent)   * cells.n_substrates + s;
            size_t d_off = static_cast<size_t>(daughter)  * cells.n_substrates + s;
            cells.secretion_rate[d_off] = cells.secretion_rate[p_off];
            cells.uptake_rate[d_off]    = cells.uptake_rate[p_off];
        }
    }

    cells.is_alive[daughter]      = 1;
    cells.voxel_index[daughter]   = cells.voxel_index[parent];
    cells.mech_voxel_index[daughter] = cells.mech_voxel_index[parent];

    return daughter;
}

// ─── Cell Integrity / Damage ─────────────────────────────────────────
//
// Updates per-cell damage: accumulation + repair.
// Matches PhysiCell's standard_cell_cell_interactions_and_damage:
//   damage += dt * damage_rate
//   damage -= dt * damage_repair_rate
//   damage = max(0, damage)
//
// If damage >= 1.0, triggers apoptosis.

void updateCellIntegrity(CellData& cells, uint32_t i, double dt) {
    if (cells.current_death_model[i] != DEATH_NONE) return;
    if (!cells.damage) return;

    double d = cells.damage[i];

    // Accumulate damage (from ongoing attacks, etc.)
    d += dt * cells.damage_rate[i];

    // Repair damage
    d -= dt * cells.damage_repair_rate[i];

    // Clamp to [0, ∞)
    if (d < 0.0) d = 0.0;

    cells.damage[i] = d;

    // Check if damage exceeds lethal threshold
    if (d >= 1.0) {
        // Trigger apoptosis from damage
        cells.current_death_model[i] = DEATH_APOPTOSIS;
        cells.is_alive[i] = 0;
        cells.apoptosis_duration[i] = 0.0;

        cells.target_fluid_fraction[i] = 0.0;
        cells.target_solid_cytoplasmic[i] = 0.0;
        cells.target_solid_nuclear[i] = 0.0;

        cells.cytoplasmic_biomass_change_rate[i] = 1.0 / 60.0;
        cells.nuclear_biomass_change_rate[i] = 0.35 / 60.0;
        cells.fluid_change_rate[i] = cells.unlysed_fluid_change_rate[i];
        cells.calcification_rate[i] = 0.0;

        cells.relative_rupture_volume[i] = 2.0;
        cells.rupture_volume[i] =
            static_cast<double>(cells.total_volume[i]) *
            cells.relative_rupture_volume[i];

        cells.motility_speed[i] = 0.0f;
    }
}

// ─── Master phenotype update ─────────────────────────────────────────

void updatePhenotype(CellData& cells, double dt, double /*current_time*/,
                     const float* density, const GridParams& grid) {
    // Collect indices that need division (can't modify num_cells mid-loop)
    std::vector<uint32_t> to_divide;
    to_divide.reserve(64);

    // Collect indices of dead cells to remove
    std::vector<uint32_t> to_remove;
    to_remove.reserve(16);

    uint32_t n = cells.num_cells;

    // Phase 0: update voxel_index for all cells based on current position
    // This is needed because cells move during mechanics but voxel_index
    // is only set at creation time.
    if (density && grid.n_substrates > 0) {
        for (uint32_t i = 0; i < n; i++) {
            int ix = static_cast<int>((cells.position_x[i] - grid.x_min) / grid.dx);
            int iy = static_cast<int>((cells.position_y[i] - grid.y_min) / grid.dy);
            ix = std::max(0, std::min(ix, static_cast<int>(grid.nx) - 1));
            iy = std::max(0, std::min(iy, static_cast<int>(grid.ny) - 1));
            cells.voxel_index[i] = static_cast<uint32_t>(iy * grid.nx + ix);
        }
    }

    // Phase 1: update each cell's phenotype
    for (uint32_t i = 0; i < n; i++) {
        if (cells.current_death_model[i] != DEATH_NONE) {
            // Dead cell: advance death model + volume
            updateVolume(cells, i, dt);
            if (advanceDeathModel(cells, i, dt)) {
                to_remove.push_back(i);
            }
            continue;
        }

        // Cell integrity / damage repair
        updateCellIntegrity(cells, i, dt);

        // If damage just killed this cell, skip to next
        if (cells.current_death_model[i] != DEATH_NONE) {
            continue;
        }

        // Check for death first (PhysiCell order: phenotype → volume → death → cycle)
        updateDeath(cells, i, dt, density, grid);

        // If death was just triggered, skip cycle/volume for this step
        if (cells.current_death_model[i] != DEATH_NONE) {
            continue;
        }

        // Volume update
        updateVolume(cells, i, dt);

        // Cell cycle (may flag for division)
        if (updateCellCycle(cells, i, dt, density, grid)) {
            to_divide.push_back(i);
        }

        // Cell-cell interactions (attack, phagocytosis, fusion)
        standardCellCellInteractions(cells, i, dt, grid);

        // Cell transformations (type switching)
        standardCellTransformations(cells, i, dt);
    }

    // Phase 2: perform divisions
    for (uint32_t parent : to_divide) {
        divideCell(cells, parent);
    }

    // Phase 3: remove dead cells (iterate in reverse to keep indices valid)
    std::sort(to_remove.begin(), to_remove.end(), std::greater<uint32_t>());
    for (uint32_t idx : to_remove) {
        if (idx < cells.num_cells) {
            cells.removeCell(idx);
        }
    }
}

// ─── standardCellCellInteractions ────────────────────────────────────
//
// Port of PhysiCell standard_cell_cell_interactions()
// (PhysiCell_standard_models.cpp:1187)
//
// For each nearby cell (within interaction distance), checks:
//   1. Dead target → phagocytosis (apoptotic/necrotic/other rates)
//   2. Live target → live phagocytosis, attack, fusion (per-type rates)
//
// Uses spatial proximity check via position-based neighbor scan.

void standardCellCellInteractions(CellData& cells, uint32_t i, double dt,
                                   const GridParams& grid) {
    if (cells.current_death_model[i] != DEATH_NONE) return;
    if (cells.num_cell_types == 0) return;

    // Early exit: skip neighbor scan if this cell has no interaction rates configured
    bool has_any_rate = false;
    if (cells.apoptotic_phagocytosis_rate && cells.apoptotic_phagocytosis_rate[i] > 0) has_any_rate = true;
    if (cells.necrotic_phagocytosis_rate && cells.necrotic_phagocytosis_rate[i] > 0) has_any_rate = true;
    if (cells.other_dead_phagocytosis_rate && cells.other_dead_phagocytosis_rate[i] > 0) has_any_rate = true;
    if (!has_any_rate && cells.num_cell_types > 0) {
        // Check per-type arrays for any non-zero rate
        for (uint32_t t = 0; t < cells.num_cell_types && !has_any_rate; t++) {
            size_t off = static_cast<size_t>(i) * cells.num_cell_types + t;
            if (cells.live_phagocytosis_rates && cells.live_phagocytosis_rates[off] > 0) has_any_rate = true;
            if (cells.attack_rates && cells.attack_rates[off] > 0) has_any_rate = true;
            if (cells.fusion_rates && cells.fusion_rates[off] > 0) has_any_rate = true;
        }
    }
    if (!has_any_rate) return;

    bool phagocytosed = false;
    bool attacked = false;
    bool fused = false;

    float xi = cells.position_x[i];
    float yi = cells.position_y[i];
    float ri = cells.radius[i];

    // Interaction distance: PhysiCell uses mean interaction distance
    // = sqrt(R_cell^2 + R_neighbor^2) as a rough contact test.
    // We use a simpler 2.5 * radius as max interaction range.
    float max_interact_dist = 2.5f * ri;

    for (uint32_t j = 0; j < cells.num_cells; j++) {
        if (j == i) continue;
        if (cells.total_volume[j] < 1e-15f) continue;

        // Quick distance check
        float dx = cells.position_x[j] - xi;
        float dy = cells.position_y[j] - yi;
        float dist2 = dx * dx + dy * dy;
        float rj = cells.radius[j];
        float interact_r = ri + rj;  // contact distance
        float max_r = max_interact_dist + rj;

        if (dist2 > max_r * max_r) continue;

        // Only interact if in contact (distance < sum of radii * max_adhesion_distance)
        float dist = std::sqrt(dist2);
        if (dist > interact_r * 1.25f) continue;  // 1.25 = default max_adhesion_distance

        uint32_t target_type = cells.cell_type[j];

        if (cells.is_alive[j] == 0) {
            // ── Dead cell: phagocytosis ──
            if (phagocytosed) continue;

            // Determine death type
            bool is_apoptotic = (cells.current_death_model[j] == DEATH_APOPTOSIS);
            bool is_necrotic  = (cells.current_death_model[j] == DEATH_NECROSIS);

            double prob = 0.0;
            if (is_apoptotic && cells.apoptotic_phagocytosis_rate) {
                prob = cells.apoptotic_phagocytosis_rate[i] * dt;
            } else if (is_necrotic && cells.necrotic_phagocytosis_rate) {
                prob = cells.necrotic_phagocytosis_rate[i] * dt;
            } else if (cells.other_dead_phagocytosis_rate) {
                prob = cells.other_dead_phagocytosis_rate[i] * dt;
            }

            if (prob > 0.0 && uniformRandom() < prob) {
                // Absorb target volumes into eater
                if (cells.fluid) {
                    cells.fluid[i] += cells.fluid[j];
                    cells.fluid[j] = 0.0;
                }
                if (cells.solid_cytoplasmic) {
                    cells.solid_cytoplasmic[i] += cells.solid_cytoplasmic[j];
                    cells.solid_cytoplasmic[j] = 0.0;
                }
                if (cells.solid_nuclear) {
                    cells.solid_cytoplasmic[i] += cells.solid_nuclear[j];
                    cells.solid_nuclear[j] = 0.0;
                }
                // Zero target
                cells.total_volume[j] = 0.0f;
                cells.nuclear_volume[j] = 0.0f;
                phagocytosed = true;
            }
        } else {
            // ── Live cell interactions ──

            // Live phagocytosis
            if (!phagocytosed && cells.live_phagocytosis_rates && cells.num_cell_types > 0) {
                size_t off = static_cast<size_t>(i) * cells.num_cell_types + target_type;
                double prob = cells.live_phagocytosis_rates[off] * dt;
                if (prob > 0.0 && uniformRandom() < prob) {
                    if (cells.fluid) {
                        cells.fluid[i] += cells.fluid[j];
                        cells.fluid[j] = 0.0;
                    }
                    if (cells.solid_cytoplasmic) {
                        cells.solid_cytoplasmic[i] += cells.solid_cytoplasmic[j];
                        cells.solid_cytoplasmic[j] = 0.0;
                    }
                    if (cells.solid_nuclear) {
                        cells.solid_cytoplasmic[i] += cells.solid_nuclear[j];
                        cells.solid_nuclear[j] = 0.0;
                    }
                    cells.total_volume[j] = 0.0f;
                    cells.nuclear_volume[j] = 0.0f;
                    cells.is_alive[j] = 0;
                    cells.current_death_model[j] = DEATH_APOPTOSIS;
                    phagocytosed = true;
                }
            }

            // Attack
            if (!attacked && cells.attack_rates && cells.num_cell_types > 0) {
                size_t off = static_cast<size_t>(i) * cells.num_cell_types + target_type;
                double attack_rate = cells.attack_rates[off];
                // PhysiCell: probability = attack_rate * immunogenicity * dt
                double prob = attack_rate * dt;
                if (prob > 0.0 && uniformRandom() < prob) {
                    // Apply damage to target
                    if (cells.damage && cells.attack_damage_rate) {
                        cells.damage[j] += cells.attack_damage_rate[i] * dt;
                    }
                    attacked = true;
                }
            }

            // Fusion
            if (!fused && cells.fusion_rates && cells.num_cell_types > 0) {
                size_t off = static_cast<size_t>(i) * cells.num_cell_types + target_type;
                double prob = cells.fusion_rates[off] * dt;
                if (prob > 0.0 && uniformRandom() < prob) {
                    // Absorb j into i (simplified fusion)
                    double vol_i = static_cast<double>(cells.total_volume[i]);
                    double vol_j = static_cast<double>(cells.total_volume[j]);
                    double total = vol_i + vol_j;
                    // Volume-weighted position
                    cells.position_x[i] = static_cast<float>(
                        (vol_i * cells.position_x[i] + vol_j * cells.position_x[j]) / total);
                    cells.position_y[i] = static_cast<float>(
                        (vol_i * cells.position_y[i] + vol_j * cells.position_y[j]) / total);
                    // Absorb volumes
                    if (cells.fluid) { cells.fluid[i] += cells.fluid[j]; cells.fluid[j] = 0; }
                    if (cells.solid_cytoplasmic) { cells.solid_cytoplasmic[i] += cells.solid_cytoplasmic[j]; cells.solid_cytoplasmic[j] = 0; }
                    if (cells.solid_nuclear) { cells.solid_nuclear[i] += cells.solid_nuclear[j]; cells.solid_nuclear[j] = 0; }
                    cells.total_volume[j] = 0.0f;
                    cells.nuclear_volume[j] = 0.0f;
                    cells.is_alive[j] = 0;
                    fused = true;
                }
            }
        }
    }
}

// ─── standardCellTransformations ─────────────────────────────────────
//
// Port of PhysiCell standard_cell_transformations()
// (PhysiCell_standard_models.cpp:1357)
//
// For each registered cell type, checks if the cell transforms with
// probability = transformation_rate[type] * dt.

void standardCellTransformations(CellData& cells, uint32_t i, double dt) {
    if (cells.current_death_model[i] != DEATH_NONE) return;
    if (!cells.transformation_rates || cells.num_cell_types == 0) return;

    for (uint32_t t = 0; t < cells.num_cell_types; t++) {
        if (t == cells.cell_type[i]) continue;  // skip self-type

        size_t off = static_cast<size_t>(i) * cells.num_cell_types + t;
        double prob = cells.transformation_rates[off] * dt;

        if (prob > 0.0 && uniformRandom() <= prob) {
            // Transform: change cell type and reset defaults
            cells.cell_type[i] = t;
            // Note: in the full implementation, this should call
            // convertCellType() to re-apply all defaults from the registry.
            // For now, just change the type ID.
            return;  // only one transformation per step
        }
    }
}

// ─── convertCellType ─────────────────────────────────────────────────
//
// Convert cell i to a new type by re-applying all defaults from the
// CellTypeRegistry. Preserves position, volume, and phase state.

void convertCellType(CellData& cells, uint32_t i, int new_type_id,
                     const CellTypeRegistry& registry) {
    // Save position
    float px = cells.position_x[i];
    float py = cells.position_y[i];
    float pz = cells.position_z[i];
    uint32_t vox = cells.voxel_index[i];

    // Re-apply all defaults from the new type
    registry.applyDefaults(cells, i, new_type_id);

    // Restore position (applyDefaults would have zeroed it)
    cells.position_x[i] = px;
    cells.position_y[i] = py;
    cells.position_z[i] = pz;
    cells.voxel_index[i] = vox;
}
