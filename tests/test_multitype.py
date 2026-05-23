#!/usr/bin/env python3
"""
test_multitype.py — Multiple cell type binary tests

Verifies that the Metal binary correctly handles multiple cell definitions:
  - Different cell types are assigned correct type IDs
  - Type-specific phenotype parameters are respected
  - Cell count per type matches placement configuration
  - Type-specific cycle rates produce different population dynamics

cells.mat field layout:
  col 5 = cell_type (0-indexed type ID)
  col 4 = total_volume
  col 6 = cycle_model_code
  col 26 = dead_flag
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


def _make_two_type_config(output_dir,
                          type0_cycle_rate=0.0, type0_death_rate=0.0,
                          type0_adhesion=0.4, type0_repulsion=10.0,
                          type1_cycle_rate=0.0, type1_death_rate=0.0,
                          type1_adhesion=0.4, type1_repulsion=10.0,
                          type0_radius=50.0, type1_positions=None,
                          max_time=60.0, dt_diffusion=0.01,
                          dt_mechanics=0.1, dt_phenotype=6.0,
                          save_interval=None):
    """Config with two cell types placed via tumor_radius (type 0) and CSV (type 1)."""
    if save_interval is None:
        save_interval = max_time

    # type1_positions: list of (x, y, z) for type 1 cells
    if type1_positions is None:
        type1_positions = []

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
        <tumor_radius>{type0_radius}</tumor_radius>
        <oncoprotein_mean>1.0</oncoprotein_mean>
        <oncoprotein_sd>0.0</oncoprotein_sd>
    </user_parameters>
    <cell_definitions>
        <cell_definition name="default" ID="0">
            <phenotype>
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">{type0_cycle_rate}</rate>
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">{type0_death_rate}</death_rate>
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
                    <cell_cell_adhesion_strength>{type0_adhesion}</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>{type0_repulsion}</cell_cell_repulsion_strength>
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
        <cell_definition name="immune" ID="1">
            <phenotype>
                <cycle code="5" name="live">
                    <phase_transition_rates units="1/min">
                        <rate start_index="0" end_index="0" fixed_duration="false">{type1_cycle_rate}</rate>
                    </phase_transition_rates>
                </cycle>
                <death>
                    <model code="100" name="apoptosis">
                        <death_rate units="1/min">{type1_death_rate}</death_rate>
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
                    <cell_cell_adhesion_strength>{type1_adhesion}</cell_cell_adhesion_strength>
                    <cell_cell_repulsion_strength>{type1_repulsion}</cell_cell_repulsion_strength>
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


def _run(ctx, csv_content=None, **kwargs):
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_multi_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cfg = _make_two_type_config(output_dir, **kwargs)

        if csv_content:
            csv_path = os.path.join(tmpdir, "cells.csv")
            with open(csv_path, "w") as f:
                f.write(csv_content)
            cfg = cfg.replace(
                'enabled="false"><filename>./cells.csv</filename>',
                f'enabled="true"><filename>{csv_path}</filename>'
            )

        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(cfg)

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

def test_two_types_both_placed(ctx):
    """Tumor type-0 cells and CSV type-1 cells both appear in cells.mat.

    type0: placed by tumor_radius=30
    type1: 5 cells placed by CSV with type_id=1
    """
    csv = "\n".join(f"{x},0,0,1" for x in [200, 210, 220, 230, 240]) + "\n"

    init_mat, _, err = _run(ctx, csv_content=csv,
                            type0_radius=30.0, max_time=1.0, dt_phenotype=6.0)
    if init_mat is None:
        return False, f"Run failed: {err}"

    n_type0 = sum(1 for c in range(init_mat.cols) if round(init_mat.get(5, c)) == 0)
    n_type1 = sum(1 for c in range(init_mat.cols) if round(init_mat.get(5, c)) == 1)

    assert n_type0 > 0, f"Expected type-0 cells from tumor_radius=30, got {n_type0}"
    assert n_type1 == 5, f"Expected 5 type-1 cells from CSV, got {n_type1}"

    return True, (
        f"Two types placed: {n_type0} type-0 (tumor) + {n_type1} type-1 (CSV)"
    )


def test_type_ids_correct_in_mat(ctx):
    """cell_type field (col 5) must match the cell_definition ID."""
    # Place type-0 cells from tumor and type-1 from CSV
    csv = "300,0,0,1\n-300,0,0,1\n"

    init_mat, _, err = _run(ctx, csv_content=csv,
                            type0_radius=30.0, max_time=1.0, dt_phenotype=6.0)
    if init_mat is None:
        return False, f"Run failed: {err}"

    types = set(round(init_mat.get(5, c)) for c in range(init_mat.cols))
    assert 0 in types, f"Type-0 cells missing from mat, found types: {types}"
    assert 1 in types, f"Type-1 cells missing from mat, found types: {types}"

    return True, f"Cell types in mat: {sorted(types)}"


def test_type_specific_cycle_rate(ctx):
    """Type-1 with high cycle rate should grow; type-0 with zero rate should not.

    After 60 min, type-1 count should exceed initial while type-0 stays constant.
    """
    csv = "\n".join(f"{x},{y},0,1" for x, y in
                    [(300,0),(310,0),(320,0),(330,0),(340,0),
                     (350,0),(360,0),(370,0),(380,0),(390,0)]) + "\n"

    init_mat, final_mat, err = _run(
        ctx, csv_content=csv,
        type0_radius=30.0, type0_cycle_rate=0.0,
        type1_cycle_rate=0.5,  # P(divide)=3.0 per step → all divide
        max_time=6.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    n0_init = sum(1 for c in range(init_mat.cols) if round(init_mat.get(5, c)) == 0)
    n1_init = sum(1 for c in range(init_mat.cols) if round(init_mat.get(5, c)) == 1)
    n0_final = sum(1 for c in range(final_mat.cols) if round(final_mat.get(5, c)) == 0)
    n1_final = sum(1 for c in range(final_mat.cols) if round(final_mat.get(5, c)) == 1)

    assert n0_init > 0, "Need type-0 cells"
    assert n1_init > 0, "Need type-1 cells"

    # Type-1 (high cycle rate) should grow
    assert n1_final > n1_init, (
        f"Type-1 (cycle_rate=0.5) should grow: {n1_init}→{n1_final}"
    )

    # Type-0 (zero cycle rate) should stay constant
    assert n0_final == n0_init, (
        f"Type-0 (cycle_rate=0) should not change: {n0_init}→{n0_final}"
    )

    return True, (
        f"Type-0: {n0_init}→{n0_final} (stable); "
        f"Type-1: {n1_init}→{n1_final} (grew)"
    )


def test_type_specific_death_rate(ctx):
    """Type-1 with high death rate should die; type-0 with zero rate should survive."""
    csv = "\n".join(f"{x},{y},0,1" for x, y in
                    [(300,0),(310,0),(320,0),(330,0),(340,0),
                     (350,0),(360,0),(370,0),(380,0),(390,0)]) + "\n"

    init_mat, final_mat, err = _run(
        ctx, csv_content=csv,
        type0_radius=30.0, type0_death_rate=0.0,
        type1_death_rate=0.1,  # high death rate
        max_time=60.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    _DEAD = 26
    n0_dead = sum(1 for c in range(final_mat.cols)
                  if round(final_mat.get(5, c)) == 0 and final_mat.get(_DEAD, c) > 0.5)
    n1_dead = sum(1 for c in range(final_mat.cols)
                  if round(final_mat.get(5, c)) == 1 and final_mat.get(_DEAD, c) > 0.5)
    n1_init = sum(1 for c in range(init_mat.cols) if round(init_mat.get(5, c)) == 1)

    assert n1_dead > 0, (
        f"Type-1 (death_rate=0.1) should have dead cells, got {n1_dead}/{n1_init}"
    )
    assert n0_dead == 0, (
        f"Type-0 (death_rate=0) should have no dead cells, got {n0_dead}"
    )

    return True, (
        f"Type-0: 0 dead; Type-1: {n1_dead}/{n1_init} dead (death_rate=0.1)"
    )


def test_multitype_simulation_stable(ctx):
    """Two-type simulation completes without crash and maintains cell count."""
    csv = "300,0,0,1\n-300,0,0,1\n0,300,0,1\n"

    init_mat, final_mat, err = _run(
        ctx, csv_content=csv,
        type0_radius=30.0, max_time=60.0, dt_phenotype=6.0)

    if init_mat is None:
        return False, f"Run failed: {err}"

    assert init_mat.cols > 0, "No initial cells"
    assert final_mat.cols > 0, "No final cells"
    assert final_mat.cols == init_mat.cols, (
        f"Cell count changed without division/death: {init_mat.cols}→{final_mat.cols}"
    )

    types = set(round(final_mat.get(5, c)) for c in range(final_mat.cols))
    return True, (
        f"Two-type simulation stable: {final_mat.cols} cells, types={sorted(types)}"
    )
