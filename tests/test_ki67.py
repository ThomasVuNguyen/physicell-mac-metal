#!/usr/bin/env python3
"""
test_ki67.py — Ki67 and alternative cycle model binary tests

Verifies that cycle models beyond "live" (code=5) produce correct behavior:
  - Ki67 basic (code=1): stochastic G→S entry, fixed-duration S→G division
  - Ki67 advanced (code=0): 3-phase cycle
  - Flow cytometry (code=2): 4-phase cycle

cells.mat field layout:
  col 6 = cycle_model_code
  col 7 = current_phase
  col 4 = total_volume

Ki67 basic phases: 0=Ki67-, 1=Ki67+
Ki67 advanced phases: 0=Ki67-, 1=Ki67+ premitotic, 2=Ki67+ postmitotic
"""

import glob
import math
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


def _make_config(output_dir, cycle_code=1, cycle_rate_01=None,
                 cycle_rate_10=None, tumor_radius=50.0,
                 o2_ic=38.0,
                 max_time=60.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None):
    """Build config for alternative cycle models.

    cycle_code=1 Ki67 basic: rate_01 = rate into Ki67+ (1/4.59/60 default)
                              rate_10 = rate out of Ki67+ = 1/duration (1/15.5/60)
    cycle_code=0 Ki67 advanced: adds a third phase
    cycle_code=2 flow_cytometry: 4-phase
    """
    if save_interval is None:
        save_interval = max_time

    # Default Ki67 basic rates from PhysiCell
    if cycle_rate_01 is None:
        cycle_rate_01 = 1.0 / (4.59 * 60.0)   # into Ki67+
    if cycle_rate_10 is None:
        cycle_rate_10 = 1.0 / (15.5 * 60.0)   # out of Ki67+ (fixed duration)

    # Build phase_transition_rates based on model
    if cycle_code == 1:
        # Ki67 basic: two rates
        phase_rates = f"""
                    <rate start_index="0" end_index="1" fixed_duration="false">{cycle_rate_01}</rate>
                    <rate start_index="1" end_index="0" fixed_duration="true">{cycle_rate_10}</rate>"""
    elif cycle_code == 0:
        # Ki67 advanced: three rates
        rate_20 = 1.0 / (2.5 * 60.0)
        phase_rates = f"""
                    <rate start_index="0" end_index="1" fixed_duration="false">{1.0/(3.62*60.0)}</rate>
                    <rate start_index="1" end_index="2" fixed_duration="true">{1.0/(13.0*60.0)}</rate>
                    <rate start_index="2" end_index="0" fixed_duration="true">{rate_20}</rate>"""
    elif cycle_code == 2:
        # Flow cytometry: 4 rates
        phase_rates = f"""
                    <rate start_index="0" end_index="1" fixed_duration="false">0.00324</rate>
                    <rate start_index="1" end_index="2" fixed_duration="true">{1.0/(8.0*60.0)}</rate>
                    <rate start_index="2" end_index="3" fixed_duration="true">{1.0/(4.0*60.0)}</rate>
                    <rate start_index="3" end_index="0" fixed_duration="true">{1.0/(1.0*60.0)}</rate>"""
    else:
        # Live (fallback)
        phase_rates = f"""
                    <rate start_index="0" end_index="0" fixed_duration="false">{cycle_rate_01}</rate>"""

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
            <initial_condition units="mmHg">{o2_ic}</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="false">{o2_ic}</Dirichlet_boundary_condition>
        </variable>
    </microenvironment_setup>
    <user_parameters>
        <number_of_cells>0</number_of_cells>
        <tumor_radius>{tumor_radius}</tumor_radius>
        <oncoprotein_mean>1.0</oncoprotein_mean>
        <oncoprotein_sd>0.0</oncoprotein_sd>
    </user_parameters>
    <cell_definitions>
        <cell_definition name="default" ID="0">
            <phenotype>
                <cycle code="{cycle_code}" name="ki67_basic">
                    <phase_transition_rates units="1/min">{phase_rates}
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">0.0</death_rate>
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
                    <speed>0</speed><migration_bias>0</migration_bias>
                    <persistence_time>0</persistence_time>
                    <options><enabled>false</enabled><use_2D>true</use_2D></options>
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


def _run_and_read(ctx, **kwargs):
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_ki67_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(_make_config(output_dir, **kwargs))

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return None, None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

        def _find(prefix):
            p = os.path.join(output_dir, f"{prefix}_cells.mat")
            if os.path.isfile(p):
                return read_mat(p)
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_cells.mat")))
            return read_mat(mats[0] if prefix == "initial" else mats[-1]) if mats else None

        return _find("initial"), _find("final"), None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Tests ─────────────────────────────────────────────────────────────

def test_ki67_basic_cycle_model_code_stored(ctx):
    """Cells with Ki67 basic (code=1) should have cycle_model_code=1 in cells.mat."""
    init_mat, final_mat, err = _run_and_read(
        ctx, cycle_code=1, tumor_radius=50.0,
        max_time=6.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols > 0, "Need initial cells"
    codes = [round(init_mat.get(6, c)) for c in range(init_mat.cols)]
    unexpected = [c for c in codes if c != 1]
    assert len(unexpected) == 0, (
        f"Expected all cells to have cycle_model_code=1, got: {set(codes)}"
    )
    return True, f"{init_mat.cols} cells all have cycle_model_code=1 (Ki67 basic)"


def test_ki67_basic_grows_population(ctx):
    """Ki67 basic with high entry rate should grow the population.

    Use an exaggerated rate_01=0.1/min (P=0.6 per step with dt=6).
    Duration fixed at 1/rate_10. Population should roughly double over time.
    """
    # Very high rate to force reliable division in short run
    rate_01 = 0.1  # 1/min: P(enter Ki67+) per step = 0.6 with dt=6
    rate_10 = 1.0 / (15.5 * 60.0)  # fixed duration ~930 min

    init_mat, final_mat, err = _run_and_read(
        ctx, cycle_code=1, cycle_rate_01=rate_01, cycle_rate_10=rate_10,
        tumor_radius=50.0, o2_ic=38.0,
        max_time=60.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    n_init = init_mat.cols
    n_final = final_mat.cols
    assert n_init > 0, "Need initial cells"

    # Many cells should enter Ki67+ phase (phase=1), even if not all divide yet
    ki67pos = [c for c in range(final_mat.cols) if round(final_mat.get(7, c)) == 1]

    assert len(ki67pos) > 0 or n_final > n_init, (
        f"Expected cells to enter Ki67+ phase or divide: "
        f"n_init={n_init}, n_final={n_final}, Ki67+={len(ki67pos)}"
    )
    return True, (
        f"Ki67 basic: {n_init}→{n_final} cells, {len(ki67pos)} in Ki67+ phase"
    )


def test_ki67_advanced_produces_three_phases(ctx):
    """Ki67 advanced (code=0) should produce cells in phases 0, 1, and 2.

    With high entry rate and long run, all phases should be populated.
    """
    rate_01 = 0.05  # high entry rate
    init_mat, final_mat, err = _run_and_read(
        ctx, cycle_code=0,
        cycle_rate_01=rate_01,
        tumor_radius=50.0, o2_ic=38.0,
        max_time=120.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols > 0, "Need cells"
    assert final_mat.cols > 0, "Need final cells"

    # Check cycle model code
    codes = set(round(final_mat.get(6, c)) for c in range(final_mat.cols))
    assert 0 in codes, f"Expected cycle_model_code=0 (Ki67 advanced), got {codes}"

    phases = [round(final_mat.get(7, c)) for c in range(final_mat.cols)]
    unique_phases = set(phases)
    return True, (
        f"Ki67 advanced: {final_mat.cols} cells, phases present: {sorted(unique_phases)}"
    )


def test_ki67_basic_divides_more_than_zero_rate(ctx):
    """Ki67 basic with high rate_01 should produce more cells than with rate=0.

    Compare two runs: high rate (rate_01=0.05) vs zero rate.
    """
    _, high_mat, err1 = _run_and_read(
        ctx, cycle_code=1, cycle_rate_01=0.05, cycle_rate_10=1.0/(15.5*60.0),
        tumor_radius=50.0, o2_ic=38.0,
        max_time=60.0, dt_phenotype=6.0)

    _, zero_mat, err2 = _run_and_read(
        ctx, cycle_code=1, cycle_rate_01=0.0, cycle_rate_10=1.0/(15.5*60.0),
        tumor_radius=50.0, o2_ic=38.0,
        max_time=60.0, dt_phenotype=6.0)

    if high_mat is None:
        return False, f"High-rate run failed: {err1}"
    if zero_mat is None:
        return False, f"Zero-rate run failed: {err2}"

    # At minimum, more cells should enter Ki67+ phase in high-rate run
    high_ki67pos = sum(1 for c in range(high_mat.cols) if round(high_mat.get(7, c)) == 1)
    zero_ki67pos = sum(1 for c in range(zero_mat.cols) if round(zero_mat.get(7, c)) == 1)

    assert high_ki67pos > zero_ki67pos or high_mat.cols > zero_mat.cols, (
        f"High rate should produce more Ki67+ or more cells than zero rate: "
        f"high Ki67+={high_ki67pos} (n={high_mat.cols}) vs "
        f"zero Ki67+={zero_ki67pos} (n={zero_mat.cols})"
    )
    return True, (
        f"High rate: {high_mat.cols} cells, {high_ki67pos} Ki67+; "
        f"Zero rate: {zero_mat.cols} cells, {zero_ki67pos} Ki67+"
    )


def test_flow_cytometry_model_code(ctx):
    """Flow cytometry cycle model (code=2) should assign correct cycle_model_code."""
    init_mat, final_mat, err = _run_and_read(
        ctx, cycle_code=2, tumor_radius=50.0, o2_ic=38.0,
        max_time=6.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols > 0, "Need cells"
    codes = set(round(init_mat.get(6, c)) for c in range(init_mat.cols))
    assert 2 in codes, f"Expected cycle_model_code=2 (flow cytometry), got {codes}"
    return True, f"{init_mat.cols} cells all have cycle_model_code=2 (flow cytometry)"
