#!/usr/bin/env python3
"""
test_cycle_death.py — Cell cycle and death model binary readback tests

Runs the Metal binary with specific cycle/death rate configs, reads cells.mat,
and verifies observable outcomes:
  - Zero rates: all cells alive, count unchanged
  - High apoptosis rate: cells flagged dead (field 26 = 1.0)
  - High cycle rate: cell count increases due to divisions

Binary uses deterministic phenotype RNG (seed=42, thread_local mt19937).
Oncoprotein set to sd=0.0 → all cells get mean=1.0, removing that variance.

cells.mat field layout (from output_writer.cpp::writeMatlabSnapshots):
  col 0  = ID
  col 1  = position_x
  col 2  = position_y
  col 3  = position_z
  col 4  = total_volume
  col 5  = cell_type
  col 6  = cycle_model_code
  col 7  = current_phase
  col 8  = elapsed_time_in_phase
  col 9  = nuclear_volume
  col 10 = cytoplasmic_volume (total - nuclear)
  ...
  col 26 = is_alive[i] ? 0.0 : 1.0  (1.0 = dead flag)
  col 27 = current_death_model
  ...
mat is column-major: mat.get(field, cell_idx), mat.cols = n_cells
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from parity.mat_reader import read_mat


def _make_config(output_dir, tumor_radius=50.0,
                 cycle_rate=0.0, death_rate=0.0,
                 max_time=60.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None,
                 oncoprotein_sd=0.0):
    """
    Build PhysiCell XML with cells. oncoprotein_sd=0 pins all cells to mean=1.0,
    keeping cycle probability = cycle_rate * dt (no per-cell variation).
    """
    if save_interval is None:
        save_interval = max_time

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PhysiCell_settings>
    <domain>
        <x_min>-500</x_min><x_max>500</x_max>
        <y_min>-500</y_min><y_max>500</y_max>
        <z_min>0</z_min><z_max>20</z_max>
        <dx>20</dx><dy>20</dy><dz>20</dz>
        <use_2D>true</use_2D>
    </domain>
    <overall>
        <max_time units="min">{max_time}</max_time>
        <dt_diffusion units="min">{dt_diffusion}</dt_diffusion>
        <dt_mechanics units="min">{dt_mechanics}</dt_mechanics>
        <dt_phenotype units="min">{dt_phenotype}</dt_phenotype>
    </overall>
    <microenvironment_setup>
        <variable name="oxygen" units="mmHg" ID="0">
            <physical_parameter_set>
                <diffusion_coefficient units="micron^2/min">0.0</diffusion_coefficient>
                <decay_rate units="1/min">0.0</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">38.0</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="false">38.0</Dirichlet_boundary_condition>
        </variable>
    </microenvironment_setup>
    <user_parameters>
        <number_of_cells>0</number_of_cells>
        <tumor_radius>{tumor_radius}</tumor_radius>
        <oncoprotein_mean>1.0</oncoprotein_mean>
        <oncoprotein_sd>{oncoprotein_sd}</oncoprotein_sd>
        <oncoprotein_min>0.0</oncoprotein_min>
        <oncoprotein_max>2.0</oncoprotein_max>
    </user_parameters>
    <cell_definitions>
        <cell_definition name="default" ID="0">
            <phenotype>
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">{cycle_rate}</rate>
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">{death_rate}</death_rate>
                    </model>
                </death>
                <volume>
                    <total>2494</total><nuclear>540</nuclear>
                    <fluid_fraction>0.75</fluid_fraction>
                    <cytoplasmic_biomass_change_rate>0.0045</cytoplasmic_biomass_change_rate>
                    <nuclear_biomass_change_rate>0.0055</nuclear_biomass_change_rate>
                    <fluid_change_rate>0.05</fluid_change_rate>
                    <calcification_rate>0</calcification_rate>
                    <relative_rupture_volume>2.0</relative_rupture_volume>
                </volume>
                <mechanics>
                    <cell_cell_adhesion_strength>0.4</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>10</cell_cell_repulsion_strength>
                    <relative_maximum_adhesion_distance>1.25</relative_maximum_adhesion_distance>
                </mechanics>
                <motility>
                    <speed>0</speed>
                    <migration_bias>0</migration_bias>
                    <persistence_time>0</persistence_time>
                    <options>
                        <enabled>false</enabled>
                        <use_2D>true</use_2D>
                    </options>
                </motility>
                <secretion></secretion>
            </phenotype>
        </cell_definition>
    </cell_definitions>
    <initial_conditions>
        <cell_positions type="csv" enabled="false"><filename>./cells.csv</filename></cell_positions>
    </initial_conditions>
    <save>
        <folder>{output_dir}</folder>
        <full_data><interval units="min">{save_interval}</interval></full_data>
        <SVG><interval units="min">{max_time * 1000}</interval></SVG>
    </save>
    <options>
        <virtual_wall_at_domain_edge>false</virtual_wall_at_domain_edge>
        <random_seed>0</random_seed>
    </options>
</PhysiCell_settings>"""


def _run_and_read_cells(ctx, **config_kwargs):
    """
    Run binary, return (initial_mat, final_mat, error_msg).
    initial_mat = cells.mat at t=0 (before any phenotype steps)
    final_mat   = cells.mat at end of simulation
    """
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_cd_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        config_xml = _make_config(output_dir, **config_kwargs)
        config_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(config_path, "w") as f:
            f.write(config_xml)

        result = subprocess.run(
            [binary, config_path],
            cwd=project_dir,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None, None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

        initial_mat_path = os.path.join(output_dir, "initial_cells.mat")
        final_mat_path = os.path.join(output_dir, "final_cells.mat")

        if not os.path.isfile(initial_mat_path):
            # Fall back to frame 0
            candidates = sorted(glob.glob(os.path.join(output_dir, "output00000000_cells.mat")))
            if candidates:
                initial_mat_path = candidates[0]
            else:
                return None, None, "initial_cells.mat not found"

        if not os.path.isfile(final_mat_path):
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_cells.mat")))
            if not mats:
                return None, None, "No cells.mat output found"
            final_mat_path = mats[-1]

        initial_mat = read_mat(initial_mat_path)
        final_mat = read_mat(final_mat_path)
        return initial_mat, final_mat, None

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Tests ────────────────────────────────────────────────────────────

_DEAD_FLAG_FIELD = 26  # is_alive[i] ? 0.0 : 1.0 in cells.mat


def test_zero_rates_no_change(ctx):
    """
    With death_rate=0 and cycle_rate=0, cell count and alive status unchanged.
    All cells should have dead_flag==0.0 in the final frame.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=50.0,
        cycle_rate=0.0,
        death_rate=0.0,
        max_time=60.0,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    n_init = initial_mat.cols
    n_final = final_mat.cols

    assert n_init > 0, "Initial cell count should be > 0 for tumor_radius=50"
    assert n_final == n_init, (
        f"Cell count changed with zero rates: {n_init} → {n_final}"
    )

    dead_count = sum(
        1 for c in range(final_mat.cols)
        if final_mat.get(_DEAD_FLAG_FIELD, c) > 0.5
    )
    assert dead_count == 0, (
        f"Expected no dead cells with death_rate=0, found {dead_count}/{n_final}"
    )
    return True, f"n={n_init} cells, all alive after 60 min with zero rates"


def test_high_apoptosis_kills_cells(ctx):
    """
    With high death_rate=0.1/min and dt_phenotype=6, most cells die within 60 min.
    P(die per step) = 0.1 * 6 = 0.6. After 10 steps most cells flagged dead.
    Dead cells stay in cells.mat for 516 min (apoptosis removal duration).
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=50.0,
        cycle_rate=0.0,
        death_rate=0.1,
        max_time=60.0,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    n_init = initial_mat.cols
    assert n_init > 0, "Need cells to test death"

    dead_count = sum(
        1 for c in range(final_mat.cols)
        if final_mat.get(_DEAD_FLAG_FIELD, c) > 0.5
    )

    assert dead_count > 0, (
        f"Expected dead cells with death_rate=0.1/min after 60 min, "
        f"got 0/{final_mat.cols}"
    )
    assert dead_count > n_init // 2, (
        f"Expected majority dead (P=0.6/step, 10 steps), "
        f"got only {dead_count}/{n_init} dead"
    )
    return True, f"{dead_count}/{n_init} cells dead after 60 min (death_rate=0.1/min)"


def test_high_cycle_rate_grows_population(ctx):
    """
    With cycle_rate=0.5/min and dt_phenotype=6, P(divide)=3.0 → all cells divide.
    One phenotype step (max_time=6) should double the cell count.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=50.0,
        cycle_rate=0.5,
        death_rate=0.0,
        max_time=6.0,
        dt_phenotype=6.0,
        oncoprotein_sd=0.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    n_init = initial_mat.cols
    n_final = final_mat.cols

    assert n_init > 0, "Need initial cells"
    assert n_final > n_init, (
        f"Expected population growth with cycle_rate=0.5/min, "
        f"got {n_init} → {n_final}"
    )
    assert n_final >= int(n_init * 1.5), (
        f"Expected at least 50% growth (1 phenotype step, all divide), "
        f"got {n_init} → {n_final}"
    )
    return True, f"Population: {n_init} → {n_final} cells (cycle_rate=0.5/min, 1 step)"


def test_apoptosis_entry_changes_targets(ctx):
    """
    On apoptosis entry, volume targets are set to 0 (cell shrinks).
    Dead cells should have smaller volume than initial after sufficient time.
    Uses high death_rate + long run to let dead cells shrink.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=50.0,
        cycle_rate=0.0,
        death_rate=0.1,
        max_time=60.0,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    initial_vol = initial_mat.get(4, 0)  # total_volume of first cell

    dead_cells = [
        c for c in range(final_mat.cols)
        if final_mat.get(_DEAD_FLAG_FIELD, c) > 0.5
    ]
    if not dead_cells:
        return False, "No dead cells found — death_rate=0.1 should kill some"

    dead_vols = [final_mat.get(4, c) for c in dead_cells]
    shrinking = sum(1 for v in dead_vols if v < initial_vol)

    assert shrinking > 0, (
        f"Expected at least some dead cells to shrink from {initial_vol:.1f} µm³, "
        f"dead cell volumes: {[f'{v:.1f}' for v in dead_vols[:5]]}"
    )
    return True, (
        f"{shrinking}/{len(dead_cells)} dead cells shrank from {initial_vol:.1f} µm³ "
        f"(min dead vol={min(dead_vols):.1f})"
    )


def test_zero_tumor_radius_no_cells(ctx):
    """tumor_radius=0 places no cells; initial_cells.mat does not exist (n_cells=0 skips write)."""
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=0.0,
        cycle_rate=0.0,
        death_rate=0.0,
        max_time=6.0,
        dt_phenotype=6.0,
    )
    # Binary should succeed; cells.mat may not exist if n_cells=0
    if err and "failed" in err.lower() and "rc=" in err:
        return False, f"Binary failed: {err}"

    if initial_mat is None and final_mat is None:
        # No cells → mat files not written (n_cells=0 path in writeMatlabSnapshots)
        return True, "No cells placed with tumor_radius=0; cells.mat not written (expected)"

    if initial_mat is not None:
        assert initial_mat.cols == 0 or initial_mat is None, (
            f"Expected 0 cells with tumor_radius=0, got {initial_mat.cols}"
        )
    return True, "tumor_radius=0: no cells placed"
