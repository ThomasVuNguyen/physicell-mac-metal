#!/usr/bin/env python3
"""
test_3d.py — 3D simulation tests

Verifies that the Metal implementation handles 3D simulations correctly
(use_2D=false). All existing tests use use_2D=true — this covers the
separate code paths for 3D mechanics, motility, and diffusion.

cells.mat field layout:
  col 1 = position_x
  col 2 = position_y
  col 3 = position_z
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


def _make_config_3d(output_dir, tumor_radius=50.0,
                    motility_enabled=False, speed=0.0, bias=0.5,
                    persistence=15.0,
                    n_substrates=1,
                    max_time=60.0, dt_diffusion=0.01,
                    dt_mechanics=0.1, dt_phenotype=6.0,
                    save_interval=None):
    if save_interval is None:
        save_interval = max_time

    mot_str = "true" if motility_enabled else "false"

    subs_xml = ""
    for i in range(n_substrates):
        subs_xml += f"""
        <variable name="sub{i}" units="mmHg" ID="{i}">
            <physical_parameter_set>
                <diffusion_coefficient units="micron^2/min">100.0</diffusion_coefficient>
                <decay_rate units="1/min">0.01</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">38.0</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="false">38.0</Dirichlet_boundary_condition>
        </variable>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PhysiCell_settings>
    <domain>
        <x_min>-200</x_min><x_max>200</x_max>
        <y_min>-200</y_min><y_max>200</y_max>
        <z_min>-200</z_min><z_max>200</z_max>
        <dx>20</dx><dy>20</dy><dz>20</dz>
        <use_2D>false</use_2D>
    </domain>
    <overall>
        <max_time units="min">{max_time}</max_time>
        <dt_diffusion units="min">{dt_diffusion}</dt_diffusion>
        <dt_mechanics units="min">{dt_mechanics}</dt_mechanics>
        <dt_phenotype units="min">{dt_phenotype}</dt_phenotype>
    </overall>
    <microenvironment_setup>{subs_xml}
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
                    <speed>{speed}</speed>
                    <migration_bias>{bias}</migration_bias>
                    <persistence_time>{persistence}</persistence_time>
                    <options>
                        <enabled>{mot_str}</enabled>
                        <use_2D>false</use_2D>
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


def _run(ctx, **kwargs):
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_3d_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(_make_config_3d(output_dir, **kwargs))

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


def _run_microenv(ctx, **kwargs):
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_3d_menv_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(_make_config_3d(output_dir, **kwargs))

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

        p = os.path.join(output_dir, "final_microenvironment0.mat")
        if not os.path.isfile(p):
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
            if not mats:
                return None, "No microenvironment output"
            p = mats[-1]

        return read_mat(p), None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Tests ─────────────────────────────────────────────────────────────

def test_3d_simulation_runs(ctx):
    """3D simulation completes without crashing."""
    init_mat, final_mat, err = _run(
        ctx, tumor_radius=30.0, motility_enabled=False,
        max_time=10.0, dt_mechanics=0.1, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"3D run failed: {err}"

    assert init_mat.cols > 0, "Expected cells in 3D simulation"
    assert final_mat.cols > 0, "Expected cells in final 3D snapshot"
    return True, f"3D simulation: {init_mat.cols} cells, ran to completion"


def test_3d_cells_have_nonzero_z(ctx):
    """Cells placed in 3D tumor should have non-zero z positions.

    In 2D mode all cells are at z≈0. In 3D mode the spherical fill
    algorithm places cells at varying z coordinates.
    """
    init_mat, _, err = _run(
        ctx, tumor_radius=50.0, motility_enabled=False,
        max_time=1.0, dt_mechanics=0.1, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols > 0, "Need cells"
    z_vals = [init_mat.get(3, c) for c in range(init_mat.cols)]
    max_abs_z = max(abs(z) for z in z_vals)

    # In 3D sphere placement, at least some cells should be at non-zero z
    assert max_abs_z > 1.0, (
        f"3D tumor should have cells at non-zero z; max|z|={max_abs_z:.2f}"
    )
    return True, f"3D cells: max|z|={max_abs_z:.2f} µm (cells span z axis)"


def test_3d_motile_cells_use_z_axis(ctx):
    """3D motile cells should move in z, unlike 2D cells (use_2D=false)."""
    init_mat, final_mat, err = _run(
        ctx, tumor_radius=5.0,
        motility_enabled=True, speed=100.0, bias=0.0, persistence=1.0,
        max_time=60.0, dt_mechanics=0.1, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols >= 1, "Need a cell"
    assert final_mat.cols >= 1, "Cell should survive"

    # At least some z-displacement expected over 60 min with speed=100
    z_disps = []
    n = min(init_mat.cols, final_mat.cols)
    for c in range(n):
        dz = abs(final_mat.get(3, c) - init_mat.get(3, c))
        z_disps.append(dz)

    max_z_disp = max(z_disps)
    assert max_z_disp > 0.1, (
        f"3D motile cells should move in z; max z-displacement={max_z_disp:.4f}"
    )
    return True, f"3D motility: max z-displacement={max_z_disp:.2f} µm"


def test_3d_voxel_count(ctx):
    """3D grid should have nx*ny*nz voxels in microenvironment output.

    Domain: x,y,z = [-200,200] with dx=dy=dz=20 → 20*20*20 = 8000 voxels.
    """
    mat, err = _run_microenv(
        ctx, tumor_radius=0.0, n_substrates=1,
        max_time=0.1, dt_diffusion=0.1, save_interval=0.1)

    if mat is None:
        return False, f"Run failed: {err}"

    expected = 20 * 20 * 20  # [-200,200] / 20 = 20 voxels per axis
    assert mat.cols == expected, (
        f"3D voxel count: expected {expected}, got {mat.cols}"
    )
    # rows = 4 + n_substrates = 5
    assert mat.rows == 5, f"Expected 5 rows (4+1 substrate), got {mat.rows}"
    return True, f"3D grid: {mat.cols} voxels ({mat.rows} rows)"


def test_3d_non_motile_stays_put(ctx):
    """Single 3D cell with no mechanics forces and no motility should not move."""
    init_mat, final_mat, err = _run(
        ctx, tumor_radius=5.0, motility_enabled=False, speed=0.0,
        max_time=60.0, dt_mechanics=0.1, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols >= 1, "Need a cell"
    n = min(init_mat.cols, final_mat.cols)
    for c in range(n):
        xi, yi, zi = init_mat.get(1, c), init_mat.get(2, c), init_mat.get(3, c)
        xf, yf, zf = final_mat.get(1, c), final_mat.get(2, c), final_mat.get(3, c)
        disp = math.sqrt((xf-xi)**2 + (yf-yi)**2 + (zf-zi)**2)
        assert disp < 1.0, (
            f"Non-motile 3D cell moved {disp:.3f} µm (expected < 1)"
        )

    return True, f"3D non-motile: {n} cell(s) stayed within 1 µm"
