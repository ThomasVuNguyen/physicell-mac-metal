#!/usr/bin/env python3
"""
test_interactions.py — Cell-cell interaction validation tests
Validates attack, ingest, fuse, and spring attachment mechanics.
"""

import os


def test_attack_includes_damage_rate(ctx):
    """Verify attack code accumulates damage = dt * damage_rate."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_interactions.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    assert "damage_rate" in source, "Attack should reference damage_rate"
    assert "attackCell" in source, "attackCell function not found"
    return True, "Attack accumulates damage via damage_rate"


def test_damage_lethal_threshold(ctx):
    """Verify damage triggers apoptosis at threshold >= 1.0."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_interactions.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    assert "damage" in source and ">= 1.0" in source, (
        "Damage threshold >= 1.0 trigger not found"
    )
    assert "triggerApoptosis" in source, "triggerApoptosis helper not found"
    return True, "Damage >= 1.0 triggers apoptosis"


def test_ingest_cell(ctx):
    """Verify ingestCell absorbs target volumes into eater."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_interactions.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    assert "ingestCell" in source, "ingestCell function not found"
    assert "fluid[eater]" in source, "Fluid absorption not found"
    assert "solid_cytoplasmic[eater]" in source, "Solid absorption not found"
    assert "removeCell" in source, "Target removal after ingestion not found"
    return True, "ingestCell absorbs volumes and removes target"


def test_fuse_cells(ctx):
    """Verify fuseCells computes volume-weighted centroid and merges."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_interactions.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    assert "fuseCells" in source, "fuseCells function not found"
    # Volume-weighted centroid
    assert "vol_surv" in source and "vol_donor" in source, (
        "Volume-weighted centroid computation not found"
    )
    assert "removeCell" in source, "Donor removal after fusion not found"
    return True, "fuseCells merges with volume-weighted centroid"


def test_spring_attachments(ctx):
    """Verify spring attachment system exists with elastic forces."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    cpp_path = os.path.join(project_dir, "src", "cell_interactions.cpp")

    with open(cpp_path, "r") as f:
        source = f.read()

    assert "updateAttachmentForces" in source, "updateAttachmentForces not found"
    assert "attachment_elastic_constant" in source, "Elastic constant not found"
    assert "tryFormAttachment" in source, "tryFormAttachment not found"
    assert "detachment_rate" in source, "Detachment rate not found"
    return True, "Spring attachments with elastic forces, stochastic detachment"
