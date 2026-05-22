#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <cstdio>
#import <cstdlib>
#import <cmath>
#import <random>

#include "metal_context.h"
#include "cell_data.h"
#include "config_parser.h"
#include "microenvironment.h"
#include "cell_mechanics.h"
#include "cell_phenotype.h"
#include "output_writer.h"
#include "motility.h"
#include "cell_definitions.h"
#include "svg_writer.h"
#include "signals.h"
#include "behaviors.h"
#include "rules.h"

// ─────────────────────────────────────────────────────────────────────
// PhysiCell Metal — Main simulation loop
// Maximizes Apple Silicon M-series performance via Metal compute shaders
// ─────────────────────────────────────────────────────────────────────

static double wallTime() {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e9;
}

// ─── Initialize cell population for heterogeneity sample ───
static void initializeHeterogeneityCells(CellData& cells, const SimConfig& config,
                                          Microenvironment& microenv) {
    if (config.cell_types.empty()) {
        printf("  WARNING: No cell types defined, using defaults\n");
        return;
    }

    const auto& ct = config.cell_types[0]; // primary cell type

    // Place cells in a disc/sphere pattern (matching PhysiCell heterogeneity)
    float x_min = config.x_min + 20;
    float x_max = config.x_max - 20;
    float y_min = config.y_min + 20;
    float y_max = config.y_max - 20;

    // Cell spacing ~ 2 * radius
    float radius = std::cbrt(3.0f * ct.total_volume / (4.0f * PI_F));
    float spacing = 0.95f * 2.0f * radius; // slight overlap like PhysiCell

    // RNG for oncoprotein distribution
    std::mt19937 rng(0);  // deterministic seed
    std::uniform_real_distribution<float> onco_dist(ct.oncoprotein_min, ct.oncoprotein_max);
    std::normal_distribution<float> onco_normal(ct.oncoprotein_mean, ct.oncoprotein_sd);

    // Place cells using PhysiCell's exact hex-packed, four-fold symmetric layout.
    // PhysiCell iterates over the first quadrant (x>=0, y>=0) and mirrors to all four.
    float center_x = (config.x_min + config.x_max) / 2.0f;
    float center_y = (config.y_min + config.y_max) / 2.0f;
    float disc_radius = config.tumor_radius;  // from config (default 250 µm)
    float y_step = spacing * std::sqrt(3.0f) / 2.0f; // hex packing row offset

    int count = 0;
    auto placeCell = [&](float px, float py) {
        uint32_t idx = cells.addCell();
        if (idx == UINT32_MAX) return;

        cells.position_x[idx] = center_x + px;
        cells.position_y[idx] = center_y + py;
        cells.position_z[idx] = 0.0f;

        cells.velocity_x[idx] = 0.0f;
        cells.velocity_y[idx] = 0.0f;
        cells.velocity_z[idx] = 0.0f;
        cells.prev_velocity_x[idx] = 0.0f;
        cells.prev_velocity_y[idx] = 0.0f;
        cells.prev_velocity_z[idx] = 0.0f;

        // ── Volume: full PhysiCell decomposition ──
        double V_total = ct.total_volume;
        double V_nuc   = ct.nuclear_volume;
        double ff      = ct.fluid_fraction;
        double V_fluid = ff * V_total;
        double V_nuc_fluid = ff * V_nuc;
        double V_nuc_solid = V_nuc - V_nuc_fluid;
        double V_cyto  = V_total - V_nuc;
        double V_cyto_solid = V_cyto - ff * V_cyto;

        cells.total_volume[idx] = ct.total_volume;
        cells.nuclear_volume[idx] = ct.nuclear_volume;
        cells.radius[idx] = radius;
        cells.nuclear_radius[idx] = std::cbrt(3.0f * ct.nuclear_volume / (4.0f * PI_F));

        cells.fluid_fraction[idx] = ff;
        cells.solid_cytoplasmic[idx] = V_cyto_solid;
        cells.solid_nuclear[idx] = V_nuc_solid;
        cells.fluid[idx] = V_fluid;

        // Volume ODE rates from config
        cells.cytoplasmic_biomass_change_rate[idx] = ct.cytoplasmic_biomass_change_rate;
        cells.nuclear_biomass_change_rate[idx] = ct.nuclear_biomass_change_rate;
        cells.fluid_change_rate[idx] = ct.fluid_change_rate;
        cells.calcification_rate[idx] = ct.calcification_rate;
        cells.calcified_fraction[idx] = 0.0;

        // Volume targets
        cells.target_solid_cytoplasmic[idx] = V_cyto_solid;
        cells.target_solid_nuclear[idx] = V_nuc_solid;
        cells.target_fluid_fraction[idx] = ff;
        cells.target_volume[idx] = V_total;
        cells.volume_change_rate[idx] = ct.cytoplasmic_biomass_change_rate;
        cells.rupture_volume[idx] = ct.relative_rupture_volume * V_total;
        cells.relative_rupture_volume[idx] = ct.relative_rupture_volume;

        // ── Mechanics ──
        cells.cell_cell_repulsion[idx] = ct.cell_cell_repulsion;
        cells.cell_cell_adhesion[idx] = ct.cell_cell_adhesion;
        cells.relative_max_adhesion_distance[idx] = ct.max_adhesion_distance;
        cells.motility_speed[idx] = ct.motility_speed;

        // Oncoprotein: normal distribution (matching heterogeneity sample)
        float onco = onco_normal(rng);
        onco = std::max(ct.oncoprotein_min, std::min(ct.oncoprotein_max, onco));
        cells.oncoprotein[idx] = onco;

        cells.cell_type[idx] = 0;
        cells.current_phase[idx] = 0;
        cells.is_alive[idx] = 1;

        // Assign to microenvironment voxel
        cells.voxel_index[idx] = microenv.voxelIndex(center_x + px, center_y + py, 0.0f);

        // ── CPU-only: Cycle fields ──
        cells.elapsed_time_in_phase[idx] = 0.0;
        cells.phase_duration[idx] = ct.cycle_rate > 0 ? (1.0 / ct.cycle_rate) : 1e10;
        cells.birth_rate[idx] = ct.cycle_rate;
        cells.cycle_model_code[idx] = static_cast<uint32_t>(ct.cycle_model_code);
        cells.transition_rate_01[idx] = ct.cycle_rate;
        cells.transition_rate_10[idx] = ct.cycle_rate_10;
        cells.phase_duration_fixed[idx] = (ct.cycle_fixed_01 ? 1u : 0u)
                                        | (ct.cycle_fixed_10 ? 2u : 0u);
        cells.num_phases[idx] = (ct.cycle_model_code == 1) ? 2 : 1;

        // ── CPU-only: Death fields ──
        cells.death_rate[idx] = ct.apoptosis_rate;
        cells.necrosis_rate[idx] = ct.necrosis_rate;
        cells.necrosis_threshold[idx] = 5.0;
        cells.current_death_model[idx] = 0;
        cells.lysed[idx] = 0;
        cells.apoptosis_duration[idx] = 0.0;
        cells.unlysed_fluid_change_rate[idx] = ct.apop_unlysed_fluid_change_rate;
        cells.lysed_fluid_change_rate[idx] = ct.apop_lysed_fluid_change_rate;

        // ── CPU-only: Secretion/uptake ──
        if (cells.uptake_rate)    cells.uptake_rate[idx] = ct.uptake_rate;
        if (cells.secretion_rate) cells.secretion_rate[idx] = 0.0;

        count++;
    };

    // PhysiCell hex-packed layout: iterate first quadrant, mirror to all 4
    int row = 0;
    for (float y = 0.0f; y < disc_radius; y += y_step) {
        float x_start = (row % 2 == 1) ? 0.5f * spacing : 0.0f;
        float x_outer = std::sqrt(disc_radius * disc_radius - y * y);

        for (float x = x_start; x < x_outer; x += spacing) {
            placeCell(x, y);                                   // (+x, +y)
            if (std::fabs(y) > 0.01f) placeCell(x, -y);       // (+x, -y)
            if (std::fabs(x) > 0.01f) {
                placeCell(-x, y);                              // (-x, +y)
                if (std::fabs(y) > 0.01f) placeCell(-x, -y);  // (-x, -y)
            }
        }
        row++;
    }

    printf("  Initialized %d cells (disc radius %.0f µm, spacing %.1f µm)\n",
           count, disc_radius, spacing);
    printf("  Oncoprotein range: [%.2f, %.2f], mean=%.2f\n",
           ct.oncoprotein_min, ct.oncoprotein_max, ct.oncoprotein_mean);
}

// ─── MAIN ───

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);  // disable buffering for piped output
        setbuf(stderr, NULL);
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║   PhysiCell Metal — Apple Silicon Accelerated           ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // ─── Parse config ───
        const char* config_path = (argc > 1) ? argv[1] : "config/PhysiCell_settings.xml";
        printf("[1/6] Parsing config: %s\n", config_path);
        SimConfig config = parseConfig(config_path);
        printf("  Domain: [%.0f, %.0f] x [%.0f, %.0f] µm\n",
               config.x_min, config.x_max, config.y_min, config.y_max);
        printf("  Max time: %.0f min (%.1f hrs)\n", config.max_time, config.max_time / 60.0);
        printf("  dt: diffusion=%.2f, mechanics=%.2f, phenotype=%.2f min\n",
               config.dt_diffusion, config.dt_mechanics, config.dt_phenotype);

        // ─── Initialize Metal ───
        printf("\n[2/6] Initializing Metal GPU\n");
        MetalContext metal;
        metal.loadShaders("build/shaders.metallib");

        // ─── Initialize microenvironment ───
        printf("\n[3/6] Setting up microenvironment\n");
        Microenvironment microenv;
        microenv.initialize(config, &metal);

        // ─── Initialize cell data (SoA in Metal shared buffer) ───
        printf("\n[4/6] Allocating cell data (SoA layout)\n");
        uint32_t max_cells = 50000; // support growth up to 50K
        CellData cells(max_cells);

        // Allocate GPU buffer (shared memory — zero-copy on Apple Silicon)
        size_t gpu_bytes = cells.gpuBufferSize();
        id<MTLBuffer> gpuBuf = metal.allocateBuffer(gpu_bytes);
        cells.setGPUBufferBase([gpuBuf contents]);
        cells.gpu_buffer = (__bridge_retained void*)gpuBuf;
        cells.allocateCPUBuffers();

        printf("  GPU buffer: %.2f MB (Metal shared, zero-copy)\n", gpu_bytes / (1024.0 * 1024.0));
        printf("  Max cells: %u\n", max_cells);

        // ─── Initialize cells ───
        printf("\n[5/6] Populating cells\n");

        // Allocate substrate secretion/uptake arrays
        GridParams gp = microenv.getGridParams();
        cells.allocateSubstrateBuffers(gp.n_substrates);

        initializeHeterogeneityCells(cells, config, microenv);

        // ─── Initialize mechanics ───
        CellMechanics mechanics;
        mechanics.initialize(config, &metal);

        // ─── Output writers ───
        OutputWriter output(config.output_folder);
        SVGWriter svgWriter;

        // ─── Initialize motility data ───
        MotilityData motility;
        motility.allocate(max_cells);
        // Set motility defaults from config for all existing cells
        for (uint32_t i = 0; i < cells.num_cells; i++) {
            if (!config.cell_types.empty()) {
                const auto& ct = config.cell_types[0];
                motility.is_motile[i] = (ct.motility_speed > 0) ? 1 : 0;
                motility.persistence_time[i] = 15.0;  // default 15 min
                motility.motility_bias[i] = ct.motility_bias;
                motility.motility_elapsed[i] = 0.0;
                motility.motility_vec_x[i] = 0.0;
                motility.motility_vec_y[i] = 0.0;
                motility.motility_vec_z[i] = 0.0;
                motility.restrict_to_2D[i] = config.use_2D ? 1 : 0;
                motility.chemotaxis_index[i] = 0;
                motility.chemotaxis_direction[i] = 1;
            }
        }

        // ─── Initialize signal/behavior dictionaries ───
        std::vector<std::string> substrate_names;
        for (auto& s : config.substrates) substrate_names.push_back(s.name);
        PhysiCellMetal::initSignalDictionary(config.substrates.size(), substrate_names);
        PhysiCellMetal::initBehaviorDictionary(config.substrates.size(), substrate_names);

        // ─── Rules engine (ready for CSV rules files) ───
        PhysiCellMetal::RuleEngine ruleEngine;

        // ─── Simulation loop ───
        printf("\n[6/6] Starting simulation\n");
        printf("══════════════════════════════════════════════════════════\n");

        double current_time = 0.0;
        double max_time = config.max_time;
        double dt_diffusion = config.dt_diffusion;
        double dt_mechanics = config.dt_mechanics;
        double dt_phenotype = config.dt_phenotype;

        int diffusion_steps = 0;
        int mechanics_steps = 0;
        int phenotype_steps = 0;
        int output_frames = 0;
        double next_output_time = 0.0;
        double output_interval = config.save_interval;

        // Timing
        double t_diffusion_total = 0, t_mechanics_total = 0;
        double t_phenotype_total = 0, t_output_total = 0;
        double t_start = wallTime();

        // Sub-step accumulators
        double mech_accumulator = 0.0;
        double pheno_accumulator = 0.0;

        while (current_time < max_time + 0.5 * dt_diffusion) {

            // ── Save output ──
            if (current_time >= next_output_time - 0.5 * dt_diffusion) {
                double t0 = wallTime();
                output.writeFrame(cells, current_time, output_frames);

                // SVG output every 10th frame to avoid excessive I/O
                if (output_frames % 10 == 0) {
                    svgWriter.writeFrame(cells, current_time, output_frames,
                                        config.output_folder,
                                        config.x_min, config.x_max,
                                        config.y_min, config.y_max);
                }
                t_output_total += wallTime() - t0;

                if (output_frames % 10 == 0) {
                    double elapsed = wallTime() - t_start;
                    double pct = 100.0 * current_time / max_time;
                    printf("  t=%.1f min (%.1f%%) | cells=%u | elapsed=%.1fs | "
                           "diff=%.2fms mech=%.2fms pheno=%.2fms\n",
                           current_time, pct, cells.num_cells, elapsed,
                           diffusion_steps > 0 ? 1000.0 * t_diffusion_total / diffusion_steps : 0,
                           mechanics_steps > 0 ? 1000.0 * t_mechanics_total / mechanics_steps : 0,
                           phenotype_steps > 0 ? 1000.0 * t_phenotype_total / phenotype_steps : 0);

                    // O₂ diagnostics: show min/mean/max across cell voxels
                    if (cells.num_cells > 0 && microenv.densityBuffer()) {
                        float o2_min = 1e9f, o2_max = -1e9f, o2_sum = 0.0f;
                        const float* dens = microenv.densityBuffer();
                        for (uint32_t ci = 0; ci < cells.num_cells; ci++) {
                            uint32_t vi = cells.voxel_index[ci];
                            float o2 = dens[vi];
                            o2_min = std::min(o2_min, o2);
                            o2_max = std::max(o2_max, o2);
                            o2_sum += o2;
                        }
                        printf("    O₂ at cells: min=%.2f mean=%.2f max=%.2f mmHg\n",
                               o2_min, o2_sum / cells.num_cells, o2_max);
                    }
                }

                output_frames++;
                next_output_time += output_interval;
            }

            {
                double t0 = wallTime();
                microenv.simulateDiffusionDecay();
                metal.waitForCompletion();

                // Enforce Dirichlet BCs (reset boundary voxels to target)
                microenv.applyDirichletBC();

                // Cell secretion/uptake — runs at diffusion_dt per PhysiCell
                // (PhysiCell calls secretion.advance() every diffusion timestep)
                {
                    float* dens = microenv.densityBuffer();
                    GridParams gp = microenv.getGridParams();
                    for (uint32_t ci = 0; ci < cells.num_cells; ci++) {
                        if (cells.current_death_model[ci] != 0) continue; // skip dead
                        processSecretion(cells, dens, gp, ci, dt_diffusion);
                    }
                }

                // Compute gradients for chemotaxis (CPU-side)
                microenv.computeGradients();
                t_diffusion_total += wallTime() - t0;
                diffusion_steps++;
            }

            // ── GPU: Mechanics (at mechanics_dt interval) ──
            mech_accumulator += dt_diffusion;
            if (mech_accumulator >= dt_mechanics - 0.5 * dt_diffusion) {
                double t0 = wallTime();
                mechanics.update(cells, (float)dt_mechanics);
                metal.waitForCompletion();
                t_mechanics_total += wallTime() - t0;
                mechanics_steps++;
                mech_accumulator = 0.0;
            }

            // ── CPU: Phenotype (at phenotype_dt interval) ──
            pheno_accumulator += dt_diffusion;
            if (pheno_accumulator >= dt_phenotype - 0.5 * dt_diffusion) {
                double t0 = wallTime();

                // Apply rules engine (modifies cell behaviors based on signals)
                GridParams gp = microenv.getGridParams();
                ruleEngine.applyAll(cells, microenv.densityBuffer(), gp, current_time);

                // Update motility vectors (before mechanics adds forces)
                updateMotility(cells, motility, microenv.densityBuffer(),
                               microenv.getGridParams(), dt_phenotype);

                updatePhenotype(cells, dt_phenotype, current_time,
                                microenv.densityBuffer(), microenv.getGridParams());
                t_phenotype_total += wallTime() - t0;
                phenotype_steps++;
                pheno_accumulator = 0.0;
            }

            current_time += dt_diffusion;
        }

        // ─── Final output ───
        output.writeFrame(cells, current_time, output_frames);
        output.writeAllFrames();

        // ─── Timing report ───
        double total_time = wallTime() - t_start;

        printf("\n══════════════════════════════════════════════════════════\n");
        printf("  SIMULATION COMPLETE\n");
        printf("══════════════════════════════════════════════════════════\n");
        printf("  Simulated time:  %.1f min (%.1f hrs)\n", current_time, current_time / 60.0);
        printf("  Final cells:     %u\n", cells.num_cells);
        printf("  Output frames:   %d\n", output_frames);
        printf("\n");
        printf("  ── Timing Breakdown ──\n");
        printf("  Total wall time: %.2f sec\n", total_time);
        printf("  Diffusion (GPU): %.2f sec (%d steps, %.3f ms/step)\n",
               t_diffusion_total, diffusion_steps,
               diffusion_steps > 0 ? 1000.0 * t_diffusion_total / diffusion_steps : 0);
        printf("  Mechanics (GPU): %.2f sec (%d steps, %.3f ms/step)\n",
               t_mechanics_total, mechanics_steps,
               mechanics_steps > 0 ? 1000.0 * t_mechanics_total / mechanics_steps : 0);
        printf("  Phenotype (CPU): %.2f sec (%d steps, %.3f ms/step)\n",
               t_phenotype_total, phenotype_steps,
               phenotype_steps > 0 ? 1000.0 * t_phenotype_total / phenotype_steps : 0);
        printf("  Output (I/O):    %.2f sec\n", t_output_total);
        printf("  Other:           %.2f sec\n",
               total_time - t_diffusion_total - t_mechanics_total - t_phenotype_total - t_output_total);
        printf("\n");
        printf("  GPU buffer:      %.2f MB (shared, zero-copy)\n", gpu_bytes / (1024.0 * 1024.0));
        printf("  Cells processed: %llu total updates\n",
               (uint64_t)cells.num_cells * (phenotype_steps + mechanics_steps));
        printf("══════════════════════════════════════════════════════════\n\n");

        // Cleanup
        cells.freeCPUBuffers();
        motility.free_all();
    }
    return 0;
}
