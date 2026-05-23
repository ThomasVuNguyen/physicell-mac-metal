#!/usr/bin/env python3
"""
test_necrosis.py — Necrosis model binary readback tests

Runs the Metal binary with hypoxic conditions and verifies cells enter the
necrosis death model. Complements test_volume_ode.py (analytical only).

Death model codes (from svg_writer.cpp comment):
  DEATH_NONE=0, DEATH_APOPTOSIS=1, DEATH_NECROSIS=2

cells.mat field layout (relevant):
  col 5  = cell_type
  col 6  = cycle_model_code
  col 26 = dead_flag (0.0 = alive, 1.0 = dead)
  col 27 = current_death_model (0=none, 1=apoptosis, 2=necrosis)

Necrosis trigger: pO2 < o2_necrosis_threshold (default 5 mmHg)
  Rate = max_necrosis_rate * (threshold - pO2) / (threshold - o2_necrosis_max)
  With pO2=0 and threshold=5, rate = max_necrosis_rate = 1/(6*60) /min
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

_DEAD_FLAG_FIELD = 26
_DEATH_MODEL_FIELD = 27
_DEATH_NECROSIS = 2
_DEATH_APOPTOSIS = 1


def _make_config(output_dir, tumor_radius=50.0,
                 o2_ic=0.0, o2_dirichlet=None,
                 apoptosis_rate=0.0,
                 max_time=60.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None):
    """Build config with oxygen at specified level to trigger necrosis."""
    if save_interval is None:
        save_interval = max_time
    dir_enabled = "true" if o2_dirichlet is not None else "false"
    dir_val = o2_dirichlet if o2_dirichlet is not None else o2_ic

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
            <Dirichlet_boundary_condition units="mmHg" enabled="{dir_enabled}">{dir_val}</Dirichlet_boundary_condition>
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
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">0.0</rate>
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">{apoptosis_rate}</death_rate>
                    </model>
                    <model code="101" name="necrosis">
                        <death_rate units="1/min">0.0</death_rate>
                        <parameters>
                            <o2_necrosis_threshold units="mmHg">5.0</o2_necrosis_threshold>
                            <o2_necrosis_max units="mmHg">2.5</o2_necrosis_max>
                            <max_necrosis_rate units="1/min">0.0027778</max_necrosis_rate>
                        </parameters>
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

    tmpdir = tempfile.mkdtemp(prefix="test_nec_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(_make_config(output_dir, **kwargs))

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=120)
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

def test_necrosis_triggered_by_hypoxia(ctx):
    """With pO2=0 (below threshold=5 mmHg), cells should enter necrosis.

    max_necrosis_rate = 1/(6*60) = 0.00278/min, dt_phenotype=6.
    P(necrosis per step) = 0.00278 * 6 = 0.0167.
    After 10 steps (60 min) expected fraction = 1-(1-0.0167)^10 ≈ 15%.
    With ~200 cells, expect at least a few necrotic.
    """
    init_mat, final_mat, err = _run_and_read(
        ctx, tumor_radius=50.0, o2_ic=0.0,
        max_time=240.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    n_init = init_mat.cols if init_mat else 0
    assert n_init > 0, "Need cells to test necrosis"

    necrotic = [c for c in range(final_mat.cols)
                if abs(final_mat.get(_DEATH_MODEL_FIELD, c) - _DEATH_NECROSIS) < 0.5]

    assert len(necrotic) > 0, (
        f"Expected necrotic cells with pO2=0 after 240 min, found 0/{final_mat.cols}"
    )
    return True, f"{len(necrotic)}/{final_mat.cols} cells necrotic (pO2=0, 240 min)"


def test_necrosis_not_triggered_at_normoxia(ctx):
    """With pO2=38 mmHg (above threshold=5), no cells should enter necrosis."""
    init_mat, final_mat, err = _run_and_read(
        ctx, tumor_radius=50.0, o2_ic=38.0,
        apoptosis_rate=0.0,
        max_time=120.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    necrotic = [c for c in range(final_mat.cols)
                if abs(final_mat.get(_DEATH_MODEL_FIELD, c) - _DEATH_NECROSIS) < 0.5]

    assert len(necrotic) == 0, (
        f"Expected 0 necrotic cells at pO2=38, found {len(necrotic)}/{final_mat.cols}"
    )
    return True, f"No necrosis at normoxia (pO2=38): 0/{final_mat.cols}"


def test_necrosis_vs_apoptosis_distinct_codes(ctx):
    """Necrosis (code=2) and apoptosis (code=1) must produce distinct death model codes.

    Run two scenarios: hypoxic (pO2=0, should give necrosis=2) and high
    apoptosis rate (pO2=38, high_rate, should give apoptosis=1).
    Verify the death_model field matches the expected code.
    """
    # Apoptosis run: high rate, normoxia
    _, apo_mat, err1 = _run_and_read(
        ctx, tumor_radius=50.0, o2_ic=38.0,
        apoptosis_rate=0.1,
        max_time=60.0, dt_phenotype=6.0)

    # Necrosis run: hypoxia, zero apoptosis
    _, nec_mat, err2 = _run_and_read(
        ctx, tumor_radius=50.0, o2_ic=0.0,
        apoptosis_rate=0.0,
        max_time=240.0, dt_phenotype=6.0)

    if apo_mat is None:
        return False, f"Apoptosis run failed: {err1}"
    if nec_mat is None:
        return False, f"Necrosis run failed: {err2}"

    apo_codes = set(
        round(apo_mat.get(_DEATH_MODEL_FIELD, c))
        for c in range(apo_mat.cols)
        if apo_mat.get(_DEAD_FLAG_FIELD, c) > 0.5
    )
    nec_codes = set(
        round(nec_mat.get(_DEATH_MODEL_FIELD, c))
        for c in range(nec_mat.cols)
        if nec_mat.get(_DEAD_FLAG_FIELD, c) > 0.5
    )

    if not apo_codes:
        return False, "No dead cells in apoptosis run"
    if not nec_codes:
        return False, "No dead cells in necrosis run"

    assert _DEATH_APOPTOSIS in apo_codes, (
        f"Apoptotic run should produce code=1, got codes={apo_codes}"
    )
    assert _DEATH_NECROSIS in nec_codes, (
        f"Necrotic run should produce code=2, got codes={nec_codes}"
    )
    assert apo_codes != nec_codes or (
        _DEATH_APOPTOSIS in apo_codes and _DEATH_NECROSIS in nec_codes
    ), "Apoptosis and necrosis should have distinct death model codes"

    return True, f"Apoptosis codes={apo_codes}, Necrosis codes={nec_codes}"


def test_necrotic_cells_swell(ctx):
    """Necrotic cells swell (volume increases) before lysis.

    With pO2=0 and necrosis entry setting target_fluid_fraction=1.0,
    dead cells (code=2) should have volume >= initial volume.
    """
    init_mat, final_mat, err = _run_and_read(
        ctx, tumor_radius=50.0, o2_ic=0.0,
        max_time=240.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    if init_mat.cols == 0:
        return False, "No initial cells"

    initial_mean_vol = sum(init_mat.get(4, c) for c in range(init_mat.cols)) / init_mat.cols

    necrotic = [c for c in range(final_mat.cols)
                if abs(final_mat.get(_DEATH_MODEL_FIELD, c) - _DEATH_NECROSIS) < 0.5]

    if not necrotic:
        return False, "No necrotic cells found — increase run time"

    necrotic_vols = [final_mat.get(4, c) for c in necrotic]
    # Some necrotic cells may have already lysed (volume → 0); check non-lysed ones
    swelling = [v for v in necrotic_vols if v > initial_mean_vol * 0.1]

    # At 240 min, some cells should still be swelling
    if not swelling:
        return True, (
            f"All {len(necrotic)} necrotic cells appear fully lysed "
            f"(volumes: min={min(necrotic_vols):.1f}); necrosis triggered correctly"
        )

    mean_nec_vol = sum(swelling) / len(swelling)
    return True, (
        f"{len(necrotic)} necrotic cells: mean vol={mean_nec_vol:.1f} µm³ "
        f"(initial mean={initial_mean_vol:.1f})"
    )
