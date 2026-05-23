#!/usr/bin/env python3
"""
test_csv_placement.py — CSV cell placement binary tests

Verifies that cells can be loaded from a CSV file using the
cell_positions type="csv" config option (geometry.cpp::loadCellsCSV).

CSV format: x, y, z, cell_type_id (PhysiCell load_cells_csv_v1)

cells.mat field layout:
  col 0 = ID
  col 1 = position_x
  col 2 = position_y
  col 3 = position_z
  col 5 = cell_type
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


def _make_config(output_dir, csv_path, max_time=1.0,
                 dt_diffusion=0.01, dt_mechanics=0.1, dt_phenotype=6.0):
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
        <tumor_radius>0</tumor_radius>
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
                    <cell_cell_adhesion_strength>0.0</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>0.0</cell_cell_repulsion_strength>
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
        <cell_positions type="csv" enabled="true">
            <filename>{csv_path}</filename>
        </cell_positions>
    </initial_conditions>
    <save>
        <folder>{output_dir}</folder>
        <full_data><interval units="min">{max_time}</interval></full_data>
        <SVG><interval units="min">{max_time * 1000}</interval></SVG>
    </save>
    <options>
        <virtual_wall_at_domain_edge>false</virtual_wall_at_domain_edge>
        <random_seed>0</random_seed>
    </options>
</PhysiCell_settings>"""


def _run_with_csv(ctx, csv_content, max_time=1.0):
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_csv_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(tmpdir, "cells.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)

        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(_make_config(output_dir, csv_path, max_time=max_time))

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

        p = os.path.join(output_dir, "initial_cells.mat")
        if not os.path.isfile(p):
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_cells.mat")))
            if not mats:
                return None, "No cells.mat output"
            p = mats[0]

        return read_mat(p), None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Tests ─────────────────────────────────────────────────────────────

def test_csv_single_cell(ctx):
    """Single cell at exact coordinates should appear at that position."""
    x, y, z = 50.0, -30.0, 0.0
    csv = f"{x},{y},{z},0\n"

    mat, err = _run_with_csv(ctx, csv)
    if mat is None:
        return False, f"Run failed: {err}"

    assert mat.cols == 1, f"Expected 1 cell from CSV, got {mat.cols}"

    actual_x = mat.get(1, 0)
    actual_y = mat.get(2, 0)
    actual_z = mat.get(3, 0)

    tol = 1.0  # 1 µm tolerance
    assert abs(actual_x - x) < tol, f"x: expected {x}, got {actual_x:.2f}"
    assert abs(actual_y - y) < tol, f"y: expected {y}, got {actual_y:.2f}"
    assert abs(actual_z - z) < tol, f"z: expected {z}, got {actual_z:.2f}"

    return True, f"CSV single cell: placed at ({actual_x:.1f},{actual_y:.1f},{actual_z:.1f})"


def test_csv_three_cells_correct_count(ctx):
    """Three cells in CSV should produce exactly 3 cells in the simulation."""
    csv = "100,0,0,0\n-100,0,0,0\n0,100,0,0\n"

    mat, err = _run_with_csv(ctx, csv)
    if mat is None:
        return False, f"Run failed: {err}"

    assert mat.cols == 3, f"Expected 3 cells from CSV, got {mat.cols}"
    return True, f"CSV 3-cell placement: {mat.cols} cells loaded correctly"


def test_csv_positions_match(ctx):
    """Cell positions from CSV should match within 1 µm tolerance."""
    positions = [(100.0, 0.0, 0.0), (-100.0, 50.0, 0.0), (0.0, -200.0, 0.0)]
    csv = "\n".join(f"{x},{y},{z},0" for x, y, z in positions) + "\n"

    mat, err = _run_with_csv(ctx, csv)
    if mat is None:
        return False, f"Run failed: {err}"

    assert mat.cols == len(positions), (
        f"Expected {len(positions)} cells, got {mat.cols}"
    )

    # Sort actual positions by x for matching
    actual = sorted(
        [(mat.get(1, c), mat.get(2, c), mat.get(3, c)) for c in range(mat.cols)],
        key=lambda p: p[0]
    )
    expected = sorted(positions, key=lambda p: p[0])

    tol = 1.0
    for (ax, ay, az), (ex, ey, ez) in zip(actual, expected):
        d = math.sqrt((ax - ex)**2 + (ay - ey)**2 + (az - ez)**2)
        assert d < tol, (
            f"Position mismatch: expected ({ex},{ey},{ez}), "
            f"got ({ax:.2f},{ay:.2f},{az:.2f}), dist={d:.3f}"
        )

    return True, f"All {len(positions)} CSV positions match within {tol} µm"


def test_csv_header_line_skipped(ctx):
    """CSV with a header line should still load cells correctly."""
    csv = "x,y,z,type\n50,50,0,0\n-50,-50,0,0\n"

    mat, err = _run_with_csv(ctx, csv)
    if mat is None:
        return False, f"Run failed: {err}"

    # Header should be skipped — expect 2 cells
    assert mat.cols == 2, (
        f"Expected 2 cells after skipping header, got {mat.cols}"
    )
    return True, f"CSV with header: {mat.cols} cells loaded (header correctly skipped)"


def test_csv_empty_lines_ignored(ctx):
    """CSV with blank lines should load only valid rows."""
    csv = "\n100,0,0,0\n\n-100,0,0,0\n\n"

    mat, err = _run_with_csv(ctx, csv)
    if mat is None:
        return False, f"Run failed: {err}"

    # Only 2 valid data rows
    assert mat.cols == 2, f"Expected 2 cells (blank lines ignored), got {mat.cols}"
    return True, f"CSV with blank lines: {mat.cols} cells loaded correctly"
