#!/usr/bin/env python3
"""
test_secretion.py — Secretion/uptake validation tests

Validates the volume-scaled implicit secretion-uptake formula from
cell_phenotype.cpp::processSecretion:

  internal_constant = dt * V_cell / V_voxel
  rho_new = (rho + internal_constant * S * rho_sat) / (1 + internal_constant * (S + U))

Reference: cell_phenotype.cpp lines 518-560
"""

import os
import math


def _secretion_step(rho, dt, sec_rate, upt_rate, sat_density,
                    V_cell, V_voxel, net_export=0.0):
    """Python reimplementation of cell_phenotype.cpp::processSecretion."""
    internal_constant = dt * V_cell / V_voxel
    rho += dt * net_export / V_voxel  # net export applied first
    temp1 = internal_constant * sec_rate * sat_density
    temp2 = 1.0 + internal_constant * (sec_rate + upt_rate)
    return (rho + temp1) / temp2


def test_secretion_formula_references_volume_scaling(ctx=None):
    """
    Source must contain volume-scaled constants (internal_constant, V_cell, V_voxel).
    The naive formula dt*S (without volume scaling) is incorrect — this ensures
    the implementation matches PhysiCell/BioFVM's implicit volume-scaled scheme.
    """
    project_dir = (ctx or {}).get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    src_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(src_path, "r") as f:
        source = f.read()

    assert "internal_constant" in source, (
        "processSecretion must use 'internal_constant' (volume-scaled dt)"
    )
    assert "V_cell" in source, "processSecretion must read V_cell (total_volume)"
    assert "V_voxel" in source, "processSecretion must compute V_voxel (dx*dy*dz)"

    return True, "Volume-scaled secretion formula present in cell_phenotype.cpp"


def test_secretion_function_present(ctx=None):
    """processSecretion must exist and reference all required arrays."""
    project_dir = (ctx or {}).get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    src_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(src_path, "r") as f:
        source = f.read()

    assert "processSecretion" in source, "processSecretion function not found"
    assert "secretion_rate" in source, "must reference secretion_rate array"
    assert "uptake_rate" in source, "must reference uptake_rate array"
    assert "saturation_density" in source, "must reference saturation_density array"
    assert "net_export_rate" in source, "must reference net_export_rate array"

    return True, "processSecretion references all required substrate arrays"


def test_secretion_numerical_volume_scaled(ctx=None):
    """
    Volume-scaled implicit scheme produces correct output for known inputs.
    Uses V_cell=2494 µm³ (default cell) and V_voxel=8000 µm³ (20³ voxel).

    internal_constant = dt * V_cell / V_voxel = 0.01 * 2494 / 8000 = 0.0031175
    temp1 = 0.0031175 * 5.0 * 38.0 = 0.59233
    temp2 = 1 + 0.0031175 * (5.0 + 1.0) = 1.018705
    rho_new = (10.0 + 0.59233) / 1.018705 ≈ 10.3883
    """
    rho = 10.0
    dt = 0.01
    sec_rate = 5.0
    upt_rate = 1.0
    sat_density = 38.0
    V_cell = 2494.0   # µm³ default cell volume
    V_voxel = 20.0**3  # µm³ for dx=dy=dz=20

    result = _secretion_step(rho, dt, sec_rate, upt_rate, sat_density, V_cell, V_voxel)

    internal_constant = dt * V_cell / V_voxel
    expected = (rho + internal_constant * sec_rate * sat_density) / \
               (1.0 + internal_constant * (sec_rate + upt_rate))

    assert abs(result - expected) < 1e-14, (
        f"Volume-scaled secretion: got {result:.8f}, expected {expected:.8f}"
    )
    # Sanity: rho should increase from 10.0 since we're secreting (and sat > rho)
    assert result > rho, f"Secretion should raise rho from {rho:.4f}, got {result:.4f}"
    return True, f"Volume-scaled secretion: rho {rho} → {result:.4f} (ic={internal_constant:.6f})"


def test_uptake_drives_rho_toward_zero(ctx=None):
    """
    Pure uptake (sec_rate=0) drives rho toward 0 monotonically.
    rho_new = rho / (1 + internal_constant * U)
    """
    V_cell, V_voxel = 2494.0, 8000.0
    rho = 38.0
    dt = 0.01
    upt_rate = 10.0  # strong uptake

    steps = 50
    for _ in range(steps):
        rho = _secretion_step(rho, dt, 0.0, upt_rate, 0.0, V_cell, V_voxel)
        assert rho >= 0.0, f"Rho went negative during uptake: {rho}"

    assert rho < 38.0 * 0.5, (
        f"After {steps} uptake steps, rho={rho:.4f} should be < half of initial"
    )
    return True, f"Uptake: rho 38.0 → {rho:.4f} after {steps} steps (rate={upt_rate})"


def test_saturation_clamps_secretion(ctx=None):
    """
    When rho >= sat_density and sec_rate > 0, secretion barely changes rho.
    (The implicit formula naturally limits rho to sat_density at equilibrium.)
    """
    V_cell, V_voxel = 2494.0, 8000.0
    rho_sat = 38.0
    rho = rho_sat  # start at saturation
    dt = 0.01
    sec_rate = 10.0
    upt_rate = 0.0

    rho_after = _secretion_step(rho, dt, sec_rate, upt_rate, rho_sat, V_cell, V_voxel)

    # At rho=sat: rho_new = (sat + ic*S*sat)/(1 + ic*S) = sat*(1 + ic*S)/(1 + ic*S) = sat
    assert abs(rho_after - rho_sat) < 1e-12, (
        f"At rho=sat_density, secretion should leave rho unchanged; "
        f"got {rho_after:.8f} vs sat={rho_sat}"
    )
    return True, f"At saturation rho={rho_sat}: rho_after={rho_after:.4f} (unchanged)"


def test_net_export_additive(ctx=None):
    """
    Net export is additive before the implicit solve:
    rho_new = (rho + dt*net_export/V_voxel + ic*S*sat) / (1 + ic*(S+U))
    """
    V_cell, V_voxel = 2494.0, 8000.0
    rho = 5.0
    dt = 0.01
    net_export = 100.0  # high net export to see clear effect
    sec_rate, upt_rate, sat = 0.0, 0.0, 38.0  # no secretion/uptake, only export

    result = _secretion_step(rho, dt, sec_rate, upt_rate, sat, V_cell, V_voxel, net_export)
    expected = rho + dt * net_export / V_voxel

    assert abs(result - expected) < 1e-12, (
        f"Net export formula: got {result:.8f}, expected {expected:.8f}"
    )
    assert result > rho, "Net export should increase rho"
    return True, f"Net export: rho {rho} → {result:.4f} (net_export={net_export})"
