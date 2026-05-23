#!/usr/bin/env python3
"""
test_motility.py — Motility model binary readback tests

Verifies observable motility outcomes by running the binary and comparing
initial vs. final cell positions from cells.mat output files.

Key constraints:
  - Motility RNG is non-deterministic (std::random_device), so tests use
    statistical assertions (any movement, not specific positions).
  - 1-cell configs isolate motility from mechanics cell-cell interactions.

cells.mat field layout (output_writer.cpp::writeMatlabSnapshots):
  field 1 = position_x
  field 2 = position_y
  field 3 = position_z
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


def _make_config(output_dir, tumor_radius=5.0,
                 motility_enabled=False, speed=0.0, bias=0.5,
                 persistence=15.0, use_2D=True,
                 max_time=60.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None):
    if save_interval is None:
        save_interval = max_time

    mot_enabled_str = "true" if motility_enabled else "false"
    use2d_str = "true" if use_2D else "false"

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
                    <cell_cell_adhesion_strength>0.0</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>0.0</cell_cell_repulsion_strength>
                    <relative_maximum_adhesion_distance>1.25</relative_maximum_adhesion_distance>
                </mechanics>
                <motility>
                    <speed>{speed}</speed>
                    <migration_bias>{bias}</migration_bias>
                    <persistence_time>{persistence}</persistence_time>
                    <options>
                        <enabled>{mot_enabled_str}</enabled>
                        <use_2D>{use2d_str}</use_2D>
                        <chemotaxis>
                            <enabled>false</enabled>
                            <substrate>oxygen</substrate>
                            <direction>1</direction>
                        </chemotaxis>
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
    """Run binary, return (initial_mat, final_mat, error_msg)."""
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_mot_")
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
            frames = sorted(glob.glob(os.path.join(output_dir, "output00000000_cells.mat")))
            if not frames:
                return None, None, "initial_cells.mat not found"
            initial_mat_path = frames[0]

        if not os.path.isfile(final_mat_path):
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_cells.mat")))
            if not mats:
                return None, None, "No cells.mat output found"
            final_mat_path = mats[-1]

        return read_mat(initial_mat_path), read_mat(final_mat_path), None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _positions(mat):
    """Return list of (x, y, z) for all cells in a mat file."""
    return [
        (mat.get(1, c), mat.get(2, c), mat.get(3, c))
        for c in range(mat.cols)
    ]


# ─── Tests ────────────────────────────────────────────────────────────

def test_non_motile_stays_put(ctx):
    """
    Single isolated cell with motility disabled should not move.
    Adhesion/repulsion = 0 to eliminate mechanics forces.
    tumor_radius=5 → exactly 1 cell placed at center.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=5.0,
        motility_enabled=False,
        speed=0.0,
        max_time=60.0,
        dt_mechanics=0.1,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    assert initial_mat.cols >= 1, "Expected at least 1 cell for tumor_radius=5"
    assert final_mat.cols == initial_mat.cols, (
        f"Cell count changed: {initial_mat.cols} → {final_mat.cols}"
    )

    for c in range(initial_mat.cols):
        xi, yi, zi = initial_mat.get(1, c), initial_mat.get(2, c), initial_mat.get(3, c)
        xf, yf, zf = final_mat.get(1, c), final_mat.get(2, c), final_mat.get(3, c)
        disp = math.sqrt((xf - xi)**2 + (yf - yi)**2 + (zf - zi)**2)
        assert disp < 0.1, (
            f"Cell {c} moved {disp:.4f} µm despite motility disabled "
            f"(from ({xi:.2f},{yi:.2f}) to ({xf:.2f},{yf:.2f}))"
        )

    n = initial_mat.cols
    return True, f"{n} cell(s) stayed put over 60 min with motility disabled"


def test_motile_cells_displace(ctx):
    """
    Cells with motility enabled and high speed should move significantly.
    Speed=100 µm/min, 60 min → cells should travel measurably.
    Motility RNG is non-deterministic; test uses any-movement assertion.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=5.0,
        motility_enabled=True,
        speed=100.0,
        bias=0.0,
        persistence=1.0,
        max_time=60.0,
        dt_mechanics=0.1,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    assert initial_mat.cols >= 1, "Need at least 1 cell"
    assert final_mat.cols >= 1, "Expected cells to survive"

    # Track position of the first cell that exists in both snapshots
    # (cell indices may differ if cells were removed/added, but count=1 here)
    xi = initial_mat.get(1, 0)
    yi = initial_mat.get(2, 0)
    xf = final_mat.get(1, 0)
    yf = final_mat.get(2, 0)

    displacement = math.sqrt((xf - xi)**2 + (yf - yi)**2)
    assert displacement > 1.0, (
        f"Motile cell barely moved: displacement={displacement:.4f} µm "
        f"(expected >> 0 with speed=100 µm/min)"
    )
    return True, f"Cell displaced {displacement:.2f} µm over 60 min (speed=100 µm/min)"


def test_2d_restriction_z_stays_zero(ctx):
    """
    With use_2D=true, all cells should have z≈0 regardless of speed or bias.
    """
    initial_mat, final_mat, err = _run_and_read_cells(
        ctx,
        tumor_radius=50.0,
        motility_enabled=True,
        speed=10.0,
        bias=0.5,
        use_2D=True,
        max_time=60.0,
        dt_mechanics=0.1,
        dt_phenotype=6.0,
    )
    if initial_mat is None:
        return False, f"Run failed: {err}"

    assert final_mat.cols > 0, "Need cells to check z"

    z_vals = [final_mat.get(3, c) for c in range(final_mat.cols)]
    max_z = max(abs(z) for z in z_vals)
    assert max_z < 1.0, (
        f"use_2D=true but max |z| = {max_z:.4f} µm (expected ≈ 0)"
    )
    return True, f"{final_mat.cols} cells with max |z|={max_z:.4f} µm (use_2D restriction working)"


def test_motility_disabled_vs_enabled_differ(ctx):
    """
    Positions of motile vs. non-motile cells should differ significantly.
    Both runs are independent; comparison uses total displacement across all cells.
    """
    kwargs_common = dict(
        tumor_radius=50.0,
        max_time=60.0,
        dt_mechanics=0.1,
        dt_phenotype=6.0,
    )

    static_init, static_final, err1 = _run_and_read_cells(
        ctx, motility_enabled=False, speed=0.0, **kwargs_common)
    motile_init, motile_final, err2 = _run_and_read_cells(
        ctx, motility_enabled=True, speed=50.0, bias=0.5, **kwargs_common)

    if static_init is None:
        return False, f"Static run failed: {err1}"
    if motile_init is None:
        return False, f"Motile run failed: {err2}"

    if static_final.cols == 0 or motile_final.cols == 0:
        return False, "No cells in output"

    # Compare mean displacement in each run from their own t=0
    n = min(static_init.cols, static_final.cols)
    static_disp = sum(
        math.sqrt((static_final.get(1, c) - static_init.get(1, c))**2 +
                  (static_final.get(2, c) - static_init.get(2, c))**2)
        for c in range(n)
    ) / max(n, 1)

    n2 = min(motile_init.cols, motile_final.cols)
    motile_disp = sum(
        math.sqrt((motile_final.get(1, c) - motile_init.get(1, c))**2 +
                  (motile_final.get(2, c) - motile_init.get(2, c))**2)
        for c in range(n2)
    ) / max(n2, 1)

    assert motile_disp > static_disp + 1.0, (
        f"Motile cells should displace more than static: "
        f"motile_mean={motile_disp:.2f} µm vs static_mean={static_disp:.2f} µm"
    )
    return True, (
        f"Motile mean displacement {motile_disp:.2f} µm > "
        f"static {static_disp:.2f} µm"
    )
