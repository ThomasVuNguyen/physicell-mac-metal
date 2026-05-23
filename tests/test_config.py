#!/usr/bin/env python3
"""
test_config.py — Config parsing and output structure verification tests

Runs the binary with specific configs and verifies:
  - Domain grid dimensions in microenvironment output
  - Frame count matches expected save_interval/max_time ratio
  - Multiple substrates produce correct field counts
  - initial and final mat files are always written
  - Voxel coordinates match domain geometry

microenvironment0.mat field layout:
  field 0 = x coord
  field 1 = y coord
  field 2 = z coord
  field 3 = voxel_volume
  field 4+s = substrate_s density
  mat.cols = n_voxels = nx * ny (2D, nz=1)
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


def _build_and_run(ctx, nx=10, ny=10, dx=20.0,
                   substrates=None, max_time=1.0,
                   dt_diffusion=0.01, dt_mechanics=0.1, dt_phenotype=6.0,
                   save_interval=None, tumor_radius=0.0):
    """
    Build config, run binary, return (output_dir_path, tmpdir, error_msg).
    Caller is responsible for shutil.rmtree(tmpdir).
    """
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    if substrates is None:
        substrates = [{"name": "oxygen", "D": 0.0, "decay": 0.0,
                       "IC": 38.0, "dirichlet": None}]
    if save_interval is None:
        save_interval = max_time

    tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    subs_xml = ""
    for i, s in enumerate(substrates):
        dirichlet_val = s.get("dirichlet")
        dir_enabled = "true" if dirichlet_val is not None else "false"
        dir_val = dirichlet_val if dirichlet_val is not None else s.get("IC", 38.0)
        subs_xml += f"""
        <variable name="{s['name']}" units="mmHg" ID="{i}">
            <physical_parameter_set>
                <diffusion_coefficient units="micron^2/min">{s['D']}</diffusion_coefficient>
                <decay_rate units="1/min">{s['decay']}</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">{s.get('IC', 38.0)}</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="{dir_enabled}">{dir_val}</Dirichlet_boundary_condition>
        </variable>"""

    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<PhysiCell_settings>
    <domain>
        <x_min>0</x_min><x_max>{nx * dx}</x_max>
        <y_min>0</y_min><y_max>{ny * dx}</y_max>
        <z_min>0</z_min><z_max>{dx}</z_max>
        <dx>{dx}</dx><dy>{dx}</dy><dz>{dx}</dz>
        <use_2D>true</use_2D>
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
    </user_parameters>
    <cell_definitions>
        <cell_definition name="default" ID="0">
            <phenotype>
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">0.0</rate>
                    </phase_transition_rates>
                </cycle>
                <death><model code="100" name="apoptosis">
                    <death_rate units="1/min">0</death_rate>
                </model></death>
                <volume><total>2494</total><nuclear>540</nuclear><fluid_fraction>0.75</fluid_fraction>
                    <cytoplasmic_biomass_change_rate>0.0045</cytoplasmic_biomass_change_rate>
                    <nuclear_biomass_change_rate>0.0055</nuclear_biomass_change_rate>
                    <fluid_change_rate>0.05</fluid_change_rate>
                    <calcification_rate>0</calcification_rate>
                    <relative_rupture_volume>2.0</relative_rupture_volume>
                </volume>
                <mechanics>
                    <cell_cell_adhesion_strength>0</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>0</cell_cell_repulsion_strength>
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

    cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
    with open(cfg_path, "w") as f:
        f.write(config_xml)

    result = subprocess.run(
        [binary, cfg_path], cwd=project_dir,
        capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

    return output_dir, tmpdir, None


def _read_final_microenv(output_dir):
    """Return final microenvironment mat or None."""
    p = os.path.join(output_dir, "final_microenvironment0.mat")
    if os.path.isfile(p):
        return read_mat(p)
    mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
    return read_mat(mats[-1]) if mats else None


# ─── Tests ────────────────────────────────────────────────────────────

def test_grid_dimensions_match_config(ctx):
    """
    microenvironment0.mat cols should equal nx * ny (2D, nz=1).
    Verified with a non-square domain: nx=8, ny=12.
    """
    nx, ny, dx = 8, 12, 25.0
    output_dir, tmpdir, err = _build_and_run(
        ctx, nx=nx, ny=ny, dx=dx, max_time=0.1)
    if output_dir is None:
        return False, f"Run failed: {err}"
    try:
        mat = _read_final_microenv(output_dir)
        assert mat is not None, "No microenvironment mat found"
        expected_voxels = nx * ny
        assert mat.cols == expected_voxels, (
            f"Expected {expected_voxels} voxels ({nx}×{ny}), got {mat.cols}"
        )
        return True, f"Grid {nx}×{ny}: mat.cols={mat.cols} (correct)"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_voxel_coordinates_match_domain(ctx):
    """
    mat field 0 (x), field 1 (y) should be cell-center positions:
      x = x_min + (ix + 0.5) * dx  for ix in [0, nx)
      y = y_min + (iy + 0.5) * dy  for iy in [0, ny)
    """
    nx, ny, dx = 10, 10, 20.0
    x_min, y_min = 0.0, 0.0

    output_dir, tmpdir, err = _build_and_run(
        ctx, nx=nx, ny=ny, dx=dx, max_time=0.1)
    if output_dir is None:
        return False, f"Run failed: {err}"
    try:
        mat = _read_final_microenv(output_dir)
        assert mat is not None, "No mat found"

        # Build set of expected cell-center (x, y) pairs
        expected = set()
        for iy in range(ny):
            for ix in range(nx):
                xc = x_min + (ix + 0.5) * dx
                yc = y_min + (iy + 0.5) * dx
                expected.add((round(xc, 4), round(yc, 4)))

        x_coords = mat.row(0)
        y_coords = mat.row(1)
        mismatches = 0
        for v in range(mat.cols):
            key = (round(x_coords[v], 4), round(y_coords[v], 4))
            if key not in expected:
                # Allow 1 µm tolerance by checking nearest expected
                dists = [math.sqrt((x_coords[v] - xe)**2 + (y_coords[v] - ye)**2)
                         for xe, ye in expected]
                if min(dists) > 1.0:
                    mismatches += 1

        assert mismatches == 0, (
            f"{mismatches}/{mat.cols} voxels at unexpected coordinates"
        )
        return True, f"{mat.cols} voxels all at correct cell-center positions"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_multi_substrate_field_count(ctx):
    """
    With N substrates, microenvironment0.mat should have N+4 rows
    (x, y, z, voxel_volume, then one row per substrate).
    Tested for N = 1, 2, 3.
    """
    for n_subs in [1, 2, 3]:
        substrates = [
            {"name": f"sub{i}", "D": 0.0, "decay": 0.0,
             "IC": float(i * 10 + 10), "dirichlet": None}
            for i in range(n_subs)
        ]
        output_dir, tmpdir, err = _build_and_run(
            ctx, substrates=substrates, max_time=0.1)
        if output_dir is None:
            return False, f"{n_subs}-substrate run failed: {err}"
        try:
            mat = _read_final_microenv(output_dir)
            assert mat is not None, f"No mat for {n_subs} substrates"
            expected_rows = 4 + n_subs
            assert mat.rows == expected_rows, (
                f"{n_subs} substrates: expected {expected_rows} rows, got {mat.rows}"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return True, "1, 2, and 3 substrates all produce correct mat row counts (4+N)"


def test_save_interval_frame_count(ctx):
    """
    Number of numbered output*_microenvironment0.mat files should be:
    floor(max_time / save_interval) + 1 in-loop frames + 1 post-loop frame.
    With max_time=30, interval=10: frames at t=0,10,20,30 (4 in-loop) + 1 post = 5 total.
    """
    max_time = 30.0
    save_interval = 10.0
    expected_min_frames = int(max_time / save_interval) + 1  # 4 in-loop frames

    output_dir, tmpdir, err = _build_and_run(
        ctx, max_time=max_time, dt_diffusion=0.1,
        save_interval=save_interval)
    if output_dir is None:
        return False, f"Run failed: {err}"
    try:
        mats = sorted(glob.glob(
            os.path.join(output_dir, "output*_microenvironment0.mat")))
        n_frames = len(mats)
        assert n_frames >= expected_min_frames, (
            f"Expected ≥{expected_min_frames} frames "
            f"(max_time={max_time}, interval={save_interval}), got {n_frames}"
        )
        return True, (
            f"max_time={max_time}, interval={save_interval}: "
            f"{n_frames} numbered output frames"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_initial_and_final_mats_written(ctx):
    """
    initial_microenvironment0.mat and final_microenvironment0.mat must both
    exist after a successful run, and must be valid readable mat files.
    """
    output_dir, tmpdir, err = _build_and_run(ctx, max_time=1.0)
    if output_dir is None:
        return False, f"Run failed: {err}"
    try:
        initial_path = os.path.join(output_dir, "initial_microenvironment0.mat")
        final_path = os.path.join(output_dir, "final_microenvironment0.mat")

        assert os.path.isfile(initial_path), "initial_microenvironment0.mat not written"
        assert os.path.isfile(final_path), "final_microenvironment0.mat not written"

        im = read_mat(initial_path)
        fm = read_mat(final_path)
        assert im.cols > 0, "initial mat has 0 columns"
        assert fm.cols > 0, "final mat has 0 columns"
        assert im.rows == fm.rows, (
            f"Row count changed between initial and final: {im.rows} → {fm.rows}"
        )
        return True, (
            f"initial ({im.rows}×{im.cols}) and final ({fm.rows}×{fm.cols}) "
            f"mats both present and valid"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
