#!/usr/bin/env python3
"""
test_diffusion_readback.py — Diffusion solver tests with binary + mat output readback

Runs the Metal binary with carefully crafted configs, reads the output
microenvironment0.mat files, and compares substrate densities against:
  - Analytical solutions (pure decay, zero decay)
  - Expected boundary conditions (Dirichlet)
  - Expected independence (multi-substrate)

This catches numerical bugs that source-inspection tests cannot.
"""

import glob
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

# Import mat reader from parity package
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from parity.mat_reader import read_mat


def _make_config(output_dir, nx=16, ny=16, dx=20.0,
                 substrates=None,
                 max_time=1.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None,
                 tumor_radius=0.0,
                 options=""):
    """Build a minimal PhysiCell XML config. substrate dicts have name, D, decay, IC, dirichlet."""
    if substrates is None:
        substrates = [{"name": "oxygen", "D": 0.0, "decay": 0.1,
                       "IC": 38.0, "dirichlet": None}]
    if save_interval is None:
        save_interval = max_time

    subs_xml = ""
    for i, s in enumerate(substrates):
        dir_enabled = "false"
        dir_val = s.get("IC", 38.0)
        if s.get("dirichlet") is not None:
            dir_enabled = "true"
            dir_val = s["dirichlet"]
        subs_xml += f"""
        <variable name="{s['name']}" units="mmHg" ID="{i}">
            <physical_parameter_set>
                <diffusion_coefficient units="micron^2/min">{s['D']}</diffusion_coefficient>
                <decay_rate units="1/min">{s['decay']}</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">{s.get('IC', 38.0)}</initial_condition>
            <Dirichlet_boundary_condition units="mmHg" enabled="{dir_enabled}">{dir_val}</Dirichlet_boundary_condition>
        </variable>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
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
                <death><model code="100" name="apoptosis"><death_rate units="1/min">0</death_rate></model></death>
                <volume><total>2494</total><nuclear>540</nuclear><fluid_fraction>0.75</fluid_fraction>
                    <cytoplasmic_biomass_change_rate>0.0045</cytoplasmic_biomass_change_rate>
                    <nuclear_biomass_change_rate>0.0055</nuclear_biomass_change_rate>
                    <fluid_change_rate>0.05</fluid_change_rate>
                    <calcification_rate>0</calcification_rate>
                    <relative_rupture_volume>2.0</relative_rupture_volume></volume>
                <mechanics>
                    <cell_cell_adhesion_strength>0.4</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>10</cell_cell_repulsion_strength>
                    <relative_maximum_adhesion_distance>1.25</relative_maximum_adhesion_distance>
                </mechanics>
                <motility>
                    <speed>1</speed><migration_bias>0.5</migration_bias><persistence_time>15</persistence_time>
                    <options><enabled>false</enabled><use_2D>true</use_2D>
                        <chemotaxis><enabled>false</enabled><substrate>oxygen</substrate>
                        <direction>1</direction></chemotaxis></options>
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
        <SVG><interval units="min">{max_time * 100}</interval></SVG>
    </save>
    <options>
        <virtual_wall_at_domain_edge>false</virtual_wall_at_domain_edge>
        <random_seed>0</random_seed>
    </options>
    {options}
</PhysiCell_settings>"""


def _run_and_read_microenv(ctx, substrates, nx=16, ny=16, max_time=1.0,
                           dt_diffusion=0.01, **kwargs):
    """Run sim, return (list of substrate density arrays, full mat object, error)."""
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_diff_rb_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        config_xml = _make_config(output_dir, nx=nx, ny=ny, substrates=substrates,
                                   max_time=max_time, dt_diffusion=dt_diffusion, **kwargs)
        config_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(config_path, "w") as f:
            f.write(config_xml)

        result = subprocess.run(
            [binary, config_path],
            cwd=project_dir,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None, None, f"Binary failed (rc={result.returncode}): {result.stderr[:300]}"

        # Find final microenvironment mat — prefer final_microenvironment0.mat
        final_mat_path = os.path.join(output_dir, "final_microenvironment0.mat")
        if not os.path.isfile(final_mat_path):
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
            if not mats:
                return None, None, "No microenvironment output files found"
            final_mat_path = mats[-1]

        mat = read_mat(final_mat_path)
        n_subs = mat.rows - 4
        densities = []
        for s in range(n_subs):
            densities.append(mat.row(4 + s))
        return densities, mat, None

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Test 1: Pure decay, D=0, no BC — compare to LOD analytical ──────

def test_pure_decay_lod_analytical(ctx):
    """D=0, no Dirichlet BC, uniform IC=38 → verify LOD analytical on interior voxels.

    The Metal Thomas solver only updates interior voxels (ix=1..nx-2, iy=1..ny-2)
    when Dirichlet is not enabled. Boundary voxels get fewer sweep updates.

    Interior voxel formula per step: rho /= (1 + 0.5*λ*dt)^2
    Expected after N steps: rho_interior = rho0 / (1 + 0.5*λ*dt)^(2N)

    We identify interior voxels from the mat's x/y coordinate fields.
    """
    lam = 0.1
    dt = 0.01
    max_time = 1.0
    N = int(max_time / dt)  # ~100 steps (actual may be 101 due to loop epsilon)
    rho0 = 38.0

    # LOD analytical prediction for interior voxels
    half_decay = 0.5 * lam * dt
    rho_interior_expected = rho0 / (1.0 + half_decay) ** (2 * N)

    substrates = [{"name": "oxygen", "D": 0.0, "decay": lam,
                   "IC": rho0, "dirichlet": None}]
    densities, mat, err = _run_and_read_microenv(
        ctx, substrates, nx=16, ny=16, max_time=max_time, dt_diffusion=dt)

    if densities is None:
        return False, err

    rho_actual = densities[0]

    # Identify interior voxels by coordinate (not at domain edge)
    x_coords = mat.row(0)
    y_coords = mat.row(1)
    x_min_c = min(x_coords)
    x_max_c = max(x_coords)
    y_min_c = min(y_coords)
    y_max_c = max(y_coords)
    interior = [
        v for v in range(mat.cols)
        if (x_coords[v] != x_min_c and x_coords[v] != x_max_c and
            y_coords[v] != y_min_c and y_coords[v] != y_max_c)
    ]

    assert len(interior) > 0, "No interior voxels found"
    interior_vals = [rho_actual[v] for v in interior]
    max_dev = max(abs(v - rho_interior_expected) for v in interior_vals)
    mean_interior = sum(interior_vals) / len(interior_vals)

    tol = 0.5  # allow 0.5 mmHg: 1 extra step (N±1) + float32 accumulation
    assert max_dev < tol, (
        f"Interior LOD deviation: {max_dev:.4f} mmHg > tol={tol}\n"
        f"  Expected: {rho_interior_expected:.4f}, Interior mean: {mean_interior:.4f}"
    )
    return True, (
        f"LOD interior decay: expected={rho_interior_expected:.4f}, "
        f"actual_mean={mean_interior:.4f}, max_dev={max_dev:.5f} "
        f"(interior={len(interior)}/{mat.cols} voxels)"
    )


# ─── Test 2: Zero decay, D=0, no BC — density should be conserved ────

def test_zero_decay_conservation(ctx):
    """D=0, decay=0, no BC → density stays at IC exactly (to float32 precision)."""
    rho0 = 38.0
    substrates = [{"name": "oxygen", "D": 0.0, "decay": 0.0,
                   "IC": rho0, "dirichlet": None}]
    densities, mat, err = _run_and_read_microenv(
        ctx, substrates, nx=16, ny=16, max_time=1.0, dt_diffusion=0.1)

    if densities is None:
        return False, err

    rho_actual = densities[0]
    max_dev = max(abs(v - rho0) for v in rho_actual)

    tol = 0.01  # should be near machine epsilon for float32
    assert max_dev < tol, (
        f"Density not conserved: max deviation {max_dev:.6f} > tol={tol} "
        f"(D=0, decay=0, no BC)"
    )
    return True, f"Conservation: max_dev={max_dev:.2e} mmHg over 1 min"


# ─── Test 3: Dirichlet BC — boundary voxels stay at target value ─────

def test_dirichlet_boundary_enforced(ctx):
    """With Dirichlet BC=0 and IC=38, boundary voxels should be reset to 0."""
    substrates = [{"name": "oxygen", "D": 100000.0, "decay": 0.0,
                   "IC": 38.0, "dirichlet": 0.0}]

    # Run long enough for diffusion to propagate
    densities, mat, err = _run_and_read_microenv(
        ctx, substrates, nx=16, ny=16, max_time=10.0, dt_diffusion=0.01,
        dt_mechanics=0.1, dt_phenotype=6.0, save_interval=10.0)

    if densities is None:
        return False, err

    rho_actual = densities[0]
    # Boundary voxels are at ix=0, ix=nx-1, iy=0, iy=ny-1 (for nx=ny=16)
    # In the microenvironment mat, voxels are ordered with voxel v = ix + iy*nx
    nx = 16
    ny = 16
    boundary_voxels = []
    for v in range(nx * ny):
        ix = v % nx
        iy = v // nx
        if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
            boundary_voxels.append(v)

    max_boundary_val = max(abs(rho_actual[v]) for v in boundary_voxels)

    tol = 1.0  # boundary should be ~0 after Dirichlet enforcement
    assert max_boundary_val < tol, (
        f"Boundary voxel not reset to Dirichlet value: max={max_boundary_val:.4f} > tol={tol}"
    )

    # Interior voxels should be > 0 (still diffusing inward)
    interior_voxels = [v for v in range(nx * ny) if v not in set(boundary_voxels)]
    interior_vals = [rho_actual[v] for v in interior_voxels[:10]]
    mean_interior = sum(interior_vals) / len(interior_vals)
    assert mean_interior > 0.1, f"Interior voxels unexpectedly near zero: mean={mean_interior:.4f}"

    return True, f"Dirichlet BC enforced: boundary_max={max_boundary_val:.4f}, interior_mean={mean_interior:.2f}"


# ─── Test 4: Multi-substrate independence ────────────────────────────

def test_multi_substrate_independent(ctx):
    """Two substrates evolve independently.

    Substrate 0 (D=0, decay=0.1): decays via LOD formula (checked on interior voxels).
    Substrate 1 (D=0, decay=0):   stays at IC=10.0 for all voxels.
    Checks that substrate 0 and 1 don't cross-contaminate each other.
    """
    rho0_s0, rho0_s1 = 38.0, 10.0
    lam = 0.1
    dt = 0.01
    max_time = 1.0
    N = int(max_time / dt)

    # Interior voxels: LOD formula applies exactly
    rho_s0_interior_expected = rho0_s0 / (1.0 + 0.5 * lam * dt) ** (2 * N)

    substrates = [
        {"name": "oxygen", "D": 0.0, "decay": lam, "IC": rho0_s0, "dirichlet": None},
        {"name": "glucose", "D": 0.0, "decay": 0.0, "IC": rho0_s1, "dirichlet": None},
    ]

    densities, mat, err = _run_and_read_microenv(
        ctx, substrates, nx=16, ny=16, max_time=max_time, dt_diffusion=dt)

    if densities is None:
        return False, err

    assert len(densities) == 2, f"Expected 2 substrates in output, got {len(densities)}"

    # Identify interior voxels from coordinate fields
    x_coords = mat.row(0)
    y_coords = mat.row(1)
    x_min_c, x_max_c = min(x_coords), max(x_coords)
    y_min_c, y_max_c = min(y_coords), max(y_coords)
    interior = [
        v for v in range(mat.cols)
        if (x_coords[v] != x_min_c and x_coords[v] != x_max_c and
            y_coords[v] != y_min_c and y_coords[v] != y_max_c)
    ]

    # Substrate 0 interior: should match LOD prediction (within 0.5 mmHg)
    s0_interior = [densities[0][v] for v in interior]
    mean_s0_int = sum(s0_interior) / len(s0_interior)
    err_s0 = abs(mean_s0_int - rho_s0_interior_expected)
    assert err_s0 < 0.5, (
        f"Substrate 0 interior decay: mean={mean_s0_int:.4f}, "
        f"expected={rho_s0_interior_expected:.4f}, err={err_s0:.4f}"
    )

    # Substrate 1 (no decay): all voxels should stay at 10.0
    mean_s1 = sum(densities[1]) / len(densities[1])
    err_s1 = abs(mean_s1 - rho0_s1)
    assert err_s1 < 0.05, (
        f"Substrate 1 (no decay) drifted: mean={mean_s1:.4f}, expected={rho0_s1:.4f}"
    )

    # Substrates must be clearly different
    assert abs(mean_s0_int - mean_s1) > 2.0, (
        f"Substrates not independent: s0_int={mean_s0_int:.4f}, s1={mean_s1:.4f}"
    )

    return True, (
        f"S0 interior decay: {rho0_s0:.1f}→{mean_s0_int:.3f} "
        f"(expected {rho_s0_interior_expected:.3f}); "
        f"S1(no decay): {mean_s1:.3f}"
    )


# ─── Test 5: Domain grid dimensions in output ─────────────────────────

def test_output_grid_dimensions_match_config(ctx):
    """Verify nx*ny*nz voxels appear in the microenvironment mat."""
    nx, ny, dx = 8, 12, 20.0  # deliberate non-square grid

    substrates = [{"name": "oxygen", "D": 0.0, "decay": 0.0,
                   "IC": 38.0, "dirichlet": None}]
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_grid_dim_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        config_xml = _make_config(output_dir, nx=nx, ny=ny, dx=dx,
                                   substrates=substrates, max_time=0.1,
                                   dt_diffusion=0.1, save_interval=0.1)
        config_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(config_path, "w") as f:
            f.write(config_xml)

        result = subprocess.run([binary, config_path], cwd=project_dir,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return False, f"Binary failed: {result.stderr[:200]}"

        # Find mat
        mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
        if not mats:
            return False, "No microenvironment mat found"

        mat = read_mat(mats[0])
        expected_voxels = nx * ny  # 2D: nz=1
        assert mat.cols == expected_voxels, (
            f"Expected {expected_voxels} voxels ({nx}x{ny}), got {mat.cols}"
        )
        # rows = 4 + n_substrates = 5
        assert mat.rows == 5, f"Expected 5 rows (4+1 substrate), got {mat.rows}"

        return True, f"Grid {nx}×{ny}: mat has {mat.cols} voxels, {mat.rows} rows"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Test 6: Voxel center coordinates are correct ────────────────────

def test_voxel_coordinates_correct(ctx):
    """Verify voxel center (x,y) coordinates match config-derived grid."""
    nx, ny, dx = 8, 8, 20.0
    x_min, y_min = 0.0, 0.0

    substrates = [{"name": "oxygen", "D": 0.0, "decay": 0.0,
                   "IC": 1.0, "dirichlet": None}]
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_voxel_coord_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        config_xml = _make_config(output_dir, nx=nx, ny=ny, dx=dx,
                                   substrates=substrates, max_time=0.1,
                                   dt_diffusion=0.1, save_interval=0.1)
        config_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(config_path, "w") as f:
            f.write(config_xml)

        result = subprocess.run([binary, config_path], cwd=project_dir,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return False, f"Binary failed: {result.stderr[:200]}"

        mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
        if not mats:
            return False, "No mat files found"

        mat = read_mat(mats[0])
        max_x_err, max_y_err = 0.0, 0.0

        for v in range(mat.cols):
            ix = v % nx
            iy = v // nx
            expected_x = x_min + (ix + 0.5) * dx
            expected_y = y_min + (iy + 0.5) * dx
            actual_x = mat.get(0, v)
            actual_y = mat.get(1, v)
            max_x_err = max(max_x_err, abs(actual_x - expected_x))
            max_y_err = max(max_y_err, abs(actual_y - expected_y))

        tol = 0.01
        assert max_x_err < tol, f"Voxel x-coord error {max_x_err:.4f} > {tol}"
        assert max_y_err < tol, f"Voxel y-coord error {max_y_err:.4f} > {tol}"

        return True, f"Voxel coords: max_x_err={max_x_err:.2e}, max_y_err={max_y_err:.2e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
