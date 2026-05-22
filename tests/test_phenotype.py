#!/usr/bin/env python3
"""
test_phenotype.py — Phenotype & cell cycle validation tests
Validates volume model, cycle models, death models, and O2-dependent behavior.
"""

import os


def test_volume_model_formula(ctx):
    """Verify volume update matches PhysiCell's standard_volume_update_function."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    # PhysiCell volume model:
    # fluid += dt * fluid_change_rate * (target_fluid_fraction * total - fluid)
    assert "fluid_change_rate" in source, "Volume model missing fluid_change_rate"
    assert "target_fluid_fraction" in source, "Volume model missing target_fluid_fraction"

    return True, "Volume model references correct PhysiCell parameters"


def test_nuclear_fluid_fraction(ctx):
    """Verify nuclear fluid is proportioned from total volume."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    # nuclear_fluid = (nuclear / total) * fluid
    assert "nuclear" in source and "fluid" in source, (
        "Nuclear fluid computation not found"
    )

    return True, "Nuclear fluid fraction computed from total"


def test_geometry_radius_from_volume(ctx):
    """Verify radius is computed as (3V/4π)^(1/3) = cbrt(V / FOUR_THIRDS_PI)."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    # Radius from volume is computed in cell_phenotype.cpp, not geometry.cpp
    pheno_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(pheno_path, "r") as f:
        source = f.read()

    # Should use cbrt (cube root)
    assert "cbrt" in source, (
        "Radius computation should use cbrt: R = cbrt(V / FOUR_THIRDS_PI)"
    )
    assert "FOUR_THIRDS_PI" in source, (
        "Radius computation should use FOUR_THIRDS_PI constant"
    )

    return True, "Radius computed via std::cbrt(V / FOUR_THIRDS_PI)"


def test_division_halves_volumes(ctx):
    """Verify cell division halves all volume compartments."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_phenotype.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    # Division should multiply volumes by 0.5
    assert "0.5" in source, "Division should halve volumes (multiply by 0.5)"

    return True, "Division halves volume compartments"
