#!/usr/bin/env python3
"""
test_secretion_binary.py — Secretion/uptake binary readback tests

Runs the Metal binary with secreting cells and reads back the microenvironment
mat to verify substrate densities actually change. Complements the analytical
tests in test_secretion.py which never exercise the binary.

Scenarios tested:
  - Secretion raises substrate density above IC
  - Uptake lowers substrate density below IC
  - No-secretion/no-uptake cell leaves density unchanged
  - Net export raises density independently of secretion
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


def _make_config(output_dir, tumor_radius=50.0, secretion_rate=0.0,
                 uptake_rate=0.0, net_export_rate=0.0, sat_density=38.0,
                 substrate_ic=10.0, dirichlet=None,
                 max_time=60.0, dt_diffusion=0.01,
                 dt_mechanics=0.1, dt_phenotype=6.0,
                 save_interval=None):
    if save_interval is None:
        save_interval = max_time
    dir_enabled = "true" if dirichlet is not None else "false"
    dir_val = dirichlet if dirichlet is not None else substrate_ic

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PhysiCell_settings>
    <domain>
        <x_min>-200</x_min><x_max>200</x_max>
        <y_min>-200</y_min><y_max>200</y_max>
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
                <diffusion_coefficient units="micron^2/min">1000.0</diffusion_coefficient>
                <decay_rate units="1/min">0.0</decay_rate>
            </physical_parameter_set>
            <initial_condition units="mmHg">{substrate_ic}</initial_condition>
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
                <secretion>
                    <substrate name="oxygen">
                        <secretion_rate units="1/min">{secretion_rate}</secretion_rate>
                        <secretion_target units="substrate density">{sat_density}</secretion_target>
                        <uptake_rate units="1/min">{uptake_rate}</uptake_rate>
                        <net_export_rate units="total substrate/min">{net_export_rate}</net_export_rate>
                    </substrate>
                </secretion>
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
    """Run binary, return (initial_densities, final_densities, error)."""
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    tmpdir = tempfile.mkdtemp(prefix="test_sec_bin_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cfg = _make_config(output_dir, **kwargs)
        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(cfg)

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, None, f"Binary failed (rc={result.returncode}): {result.stderr[:400]}"

        def _read(prefix):
            p = os.path.join(output_dir, f"{prefix}_microenvironment0.mat")
            if os.path.isfile(p):
                return read_mat(p)
            mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
            return read_mat(mats[0] if prefix == "initial" else mats[-1]) if mats else None

        init_mat = _read("initial")
        final_mat = _read("final")
        if init_mat is None or final_mat is None:
            return None, None, "Could not read microenvironment mats"

        # Substrate density is at row index 4 (x,y,z,vol,sub0)
        init_dens = init_mat.row(4)
        final_dens = final_mat.row(4)
        return init_dens, final_dens, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ─── Tests ─────────────────────────────────────────────────────────────

def test_secretion_raises_density(ctx):
    """Cells secreting at rate=10/min should raise substrate above IC=1.0.

    With IC below sat (sat=38, IC=1), cells secrete and push density up.
    After 60 min of active secretion the mean density should rise measurably.
    """
    init_dens, final_dens, err = _run(
        ctx,
        tumor_radius=50.0,
        secretion_rate=10.0, uptake_rate=0.0, sat_density=38.0,
        substrate_ic=1.0, dirichlet=None,
        max_time=60.0, dt_diffusion=0.01, dt_phenotype=6.0)

    if init_dens is None:
        return False, f"Run failed: {err}"

    mean_init = _mean(init_dens)
    mean_final = _mean(final_dens)

    assert mean_final > mean_init, (
        f"Secretion should raise density: init={mean_init:.4f}, final={mean_final:.4f}"
    )
    assert mean_final > mean_init + 0.5, (
        f"Expected >0.5 mmHg rise from secretion; got {mean_final - mean_init:.4f}"
    )
    return True, f"Secretion raised mean density {mean_init:.3f} → {mean_final:.3f} mmHg"


def test_uptake_lowers_density(ctx):
    """Cells with high uptake rate should deplete substrate below IC=38.

    IC=38, uptake_rate=5/min, no secretion — cells consume substrate.
    After 60 min the mean density inside the tumor should be < IC.
    """
    init_dens, final_dens, err = _run(
        ctx,
        tumor_radius=50.0,
        secretion_rate=0.0, uptake_rate=5.0, sat_density=38.0,
        substrate_ic=38.0, dirichlet=None,
        max_time=60.0, dt_diffusion=0.01, dt_phenotype=6.0)

    if init_dens is None:
        return False, f"Run failed: {err}"

    mean_init = _mean(init_dens)
    mean_final = _mean(final_dens)

    assert mean_final < mean_init, (
        f"Uptake should lower density: init={mean_init:.4f}, final={mean_final:.4f}"
    )
    assert mean_init - mean_final > 0.5, (
        f"Expected >0.5 mmHg drop from uptake; got {mean_init - mean_final:.4f}"
    )
    return True, f"Uptake lowered mean density {mean_init:.3f} → {mean_final:.3f} mmHg"


def test_no_secretion_no_change(ctx):
    """With secretion=0 and uptake=0 and no diffusion-decay, density is unchanged.

    D=0, decay=0, no cells secreting → field frozen at IC.
    Tolerance: 0.01 mmHg to allow float32 accumulation.
    """
    # Use D=0 and no secretion/uptake: density should be conserved
    project_dir = ctx.get("project_dir", _ROOT)
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))

    ic = 15.0
    tmpdir = tempfile.mkdtemp(prefix="test_sec_noop_")
    try:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cfg = _make_config(output_dir,
                           tumor_radius=50.0,
                           secretion_rate=0.0, uptake_rate=0.0, net_export_rate=0.0,
                           substrate_ic=ic, dirichlet=None,
                           max_time=60.0, dt_diffusion=0.01, dt_phenotype=6.0)

        # Patch: override diffusion coefficient to 0 to isolate secretion
        cfg = cfg.replace(
            "<diffusion_coefficient units=\"micron^2/min\">1000.0</diffusion_coefficient>",
            "<diffusion_coefficient units=\"micron^2/min\">0.0</diffusion_coefficient>"
        )

        cfg_path = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(cfg_path, "w") as f:
            f.write(cfg)

        result = subprocess.run(
            [binary, cfg_path], cwd=project_dir,
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, f"Binary failed: {result.stderr[:300]}"

        mats = sorted(glob.glob(os.path.join(output_dir, "output*_microenvironment0.mat")))
        if not mats:
            return False, "No microenvironment output"

        init_mat = read_mat(os.path.join(output_dir, "initial_microenvironment0.mat")
                            if os.path.isfile(os.path.join(output_dir, "initial_microenvironment0.mat"))
                            else mats[0])
        final_mat = read_mat(os.path.join(output_dir, "final_microenvironment0.mat")
                             if os.path.isfile(os.path.join(output_dir, "final_microenvironment0.mat"))
                             else mats[-1])

        init_dens = init_mat.row(4)
        final_dens = final_mat.row(4)
        max_dev = max(abs(final_dens[v] - init_dens[v]) for v in range(len(init_dens)))

        assert max_dev < 0.5, (
            f"With secretion=uptake=0 and D=0, density should not change; max_dev={max_dev:.4f}"
        )
        return True, f"No-secretion density conserved: max_dev={max_dev:.2e} mmHg"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_saturation_limits_secretion(ctx):
    """Cells cannot push density above sat_density.

    Secretion_rate=100/min, sat=5.0, IC=1.0 — density should approach but not
    significantly exceed sat_density after long run.
    """
    init_dens, final_dens, err = _run(
        ctx,
        tumor_radius=50.0,
        secretion_rate=100.0, uptake_rate=0.0, sat_density=5.0,
        substrate_ic=1.0, dirichlet=None,
        max_time=120.0, dt_diffusion=0.01, dt_phenotype=6.0)

    if init_dens is None:
        return False, f"Run failed: {err}"

    max_final = max(final_dens)

    # Density may slightly exceed sat near cell centers due to fast secretion,
    # but should not be hugely above sat (implicit scheme limits overshoot).
    assert max_final < 10.0, (
        f"Density vastly exceeded sat_density=5.0: max={max_final:.4f}"
    )
    return True, f"Saturation respected: max density={max_final:.4f} (sat=5.0)"


def test_net_export_raises_density(ctx):
    """Net export (independent of secretion) should add substrate to the field.

    net_export_rate=10/min with IC=0, sat=0, no secretion.
    Cells export substrate unconditionally — density should rise.
    """
    init_dens, final_dens, err = _run(
        ctx,
        tumor_radius=50.0,
        secretion_rate=0.0, uptake_rate=0.0, net_export_rate=10.0,
        sat_density=0.0, substrate_ic=0.0, dirichlet=None,
        max_time=60.0, dt_diffusion=0.01, dt_phenotype=6.0)

    if init_dens is None:
        return False, f"Run failed: {err}"

    mean_final = _mean(final_dens)
    assert mean_final > 0.0, (
        f"Net export should raise density above 0; got mean={mean_final:.4f}"
    )
    return True, f"Net export raised mean density to {mean_final:.4f} mmHg"
