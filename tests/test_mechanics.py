#!/usr/bin/env python3
"""
test_mechanics.py — Mechanics shader validation tests

Validates GPU spatial hashing, force computation, integration,
and synchronization correctness.
"""

import os


# ─────────────────────────────────────────────────────────────────────
# Test 8: Unified mechanics pipeline uses single command buffer
# ─────────────────────────────────────────────────────────────────────
def test_unified_mechanics_pipeline_exists(ctx):
    """Verify dispatchMechanicsPipeline encodes all 4 kernels in one command buffer."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    mm_path = os.path.join(project_dir, "src", "metal_context.mm")

    with open(mm_path, "r") as f:
        source = f.read()

    assert "dispatchMechanicsPipeline" in source, (
        "dispatchMechanicsPipeline not found in metal_context.mm"
    )

    # Count pipeline state switches within the method
    # There should be 4: clearHash, buildHash, forces, integrate
    method_start = source.find("void MetalContext::dispatchMechanicsPipeline")
    assert method_start >= 0, "Method implementation not found"

    method_body = source[method_start:]
    # Find the closing brace (rough heuristic)
    brace_count = 0
    method_end = method_start
    for i, c in enumerate(method_body):
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                method_end = i
                break
    method_body = method_body[:method_end]

    pipeline_switches = method_body.count("setComputePipelineState")
    assert pipeline_switches == 4, (
        f"Expected 4 pipeline state switches (clear, build, forces, integrate), found {pipeline_switches}"
    )

    return True, "4 kernels encoded in single command buffer"


# ─────────────────────────────────────────────────────────────────────
# Test 9: Memory barriers between mechanics kernels
# ─────────────────────────────────────────────────────────────────────
def test_memory_barriers_present(ctx):
    """Verify memory barriers exist between each mechanics kernel dispatch."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    mm_path = os.path.join(project_dir, "src", "metal_context.mm")

    with open(mm_path, "r") as f:
        source = f.read()

    method_start = source.find("void MetalContext::dispatchMechanicsPipeline")
    method_body = source[method_start:] if method_start >= 0 else ""

    barrier_count = method_body.count("memoryBarrierWithScope")
    assert barrier_count >= 3, (
        f"Expected ≥3 memory barriers (clear→build, build→forces, forces→integrate), "
        f"found {barrier_count}"
    )

    assert "MTLBarrierScopeBuffers" in method_body, (
        "Memory barriers should use MTLBarrierScopeBuffers"
    )

    return True, f"{barrier_count} memory barriers with MTLBarrierScopeBuffers"


# ─────────────────────────────────────────────────────────────────────
# Test 10: Field indices are shared between CPU and GPU
# ─────────────────────────────────────────────────────────────────────
def test_field_indices_unified(ctx):
    """Verify types.h defines shared field indices for both CPU and GPU."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    types_path = os.path.join(project_dir, "shaders", "types.h")

    with open(types_path, "r") as f:
        source = f.read()

    # The shared CELL_FIELD_* defines should be outside the #ifndef __METAL_VERSION__ guard
    metal_guard_pos = source.find("#ifndef __METAL_VERSION__")
    assert metal_guard_pos >= 0, "#ifndef __METAL_VERSION__ guard not found"

    # CELL_FIELD_POS_X should be defined before the guard
    cell_field_pos = source.find("#define CELL_FIELD_POS_X")
    assert cell_field_pos >= 0, "CELL_FIELD_POS_X not found in types.h"
    assert cell_field_pos < metal_guard_pos, (
        "CELL_FIELD_POS_X should be defined before the #ifndef __METAL_VERSION__ guard "
        "(to be visible to both CPU and GPU)"
    )

    return True, "Shared field indices defined before Metal guard"


# ─────────────────────────────────────────────────────────────────────
# Test 11: Metal shaders use shared field indices (not local defines)
# ─────────────────────────────────────────────────────────────────────
def test_shaders_use_shared_indices(ctx):
    """Verify mechanics.metal and integrate.metal use CELL_FIELD_* from types.h."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))

    for shader in ["mechanics.metal", "integrate.metal"]:
        shader_path = os.path.join(project_dir, "shaders", shader)
        with open(shader_path, "r") as f:
            source = f.read()

        # Should NOT have local redefinitions like #define FIELD_POS_X 0
        assert "#define FIELD_POS_X" not in source, (
            f"{shader} still has local #define FIELD_POS_X — "
            "should use CELL_FIELD_POS_X from types.h"
        )

        # Should use CELL_FIELD_* from types.h
        assert "CELL_FIELD_POS_X" in source, (
            f"{shader} doesn't use CELL_FIELD_POS_X from types.h"
        )

    return True, "Both shaders use shared CELL_FIELD_* indices"


# ─────────────────────────────────────────────────────────────────────
# Test 12: Adams-Bashforth integrator formula
# ─────────────────────────────────────────────────────────────────────
def test_adams_bashforth_formula(ctx):
    """Verify integrate.metal uses correct AB2 formula."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    shader_path = os.path.join(project_dir, "shaders", "integrate.metal")

    with open(shader_path, "r") as f:
        source = f.read()

    # AB2: pos += dt * (1.5 * vel - 0.5 * prev_vel)
    assert "1.5f" in source and "0.5f" in source, (
        "Adams-Bashforth coefficients (1.5, 0.5) not found in integrate.metal"
    )

    # Must store current velocity as previous
    assert "CELL_FIELD_PREV_VX" in source, (
        "Previous velocity storage not found — AB2 needs vel→prev_vel"
    )

    return True, "AB2 formula: pos += dt * (1.5*v - 0.5*v_prev)"


# ─────────────────────────────────────────────────────────────────────
# Test 13: Spatial hash uses correct voxel index computation
# ─────────────────────────────────────────────────────────────────────
def test_spatial_hash_voxel_index(ctx):
    """Verify the build_spatial_hash kernel computes voxel indices correctly."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    shader_path = os.path.join(project_dir, "shaders", "mechanics.metal")

    with open(shader_path, "r") as f:
        source = f.read()

    # Should compute vi = (px - x_min) / voxel_size
    assert "grid.x_min" in source, "Missing grid.x_min in voxel computation"
    assert "mech.mechanics_voxel_size" in source, "Missing mechanics_voxel_size"

    # Should clamp to grid bounds
    assert "clamp" in source, "Missing clamp in voxel index computation"

    # Should use atomic_fetch_add for hash bucket insertion
    assert "atomic_fetch_add_explicit" in source, "Missing atomic hash insertion"

    return True, "Spatial hash: position→voxel with clamping and atomic insertion"


# ─────────────────────────────────────────────────────────────────────
# Test 14: Force computation uses Moore neighborhood
# ─────────────────────────────────────────────────────────────────────
def test_moore_neighborhood_search(ctx):
    """Verify compute_forces iterates over 3x3 (2D) or 3x3x3 (3D) neighborhood."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    shader_path = os.path.join(project_dir, "shaders", "mechanics.metal")

    with open(shader_path, "r") as f:
        source = f.read()

    # Should have triple-nested loops for di, dj, dk
    assert "for (int dk" in source, "Missing dk loop for 3D Moore neighborhood"
    assert "for (int dj" in source, "Missing dj loop"
    assert "for (int di" in source, "Missing di loop"

    # Should iterate -1 to +1
    assert "-1" in source and "<= 1" in source, "Loop bounds should be -1 to +1"

    return True, "Moore neighborhood: 3x3x3 with proper bounds"


# ─────────────────────────────────────────────────────────────────────
# Test 15: PhysiCell force formula (repulsion + adhesion)
# ─────────────────────────────────────────────────────────────────────
def test_force_formula_matches_physicell(ctx):
    """Verify force computation matches PhysiCell's add_potentials formula."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    shader_path = os.path.join(project_dir, "shaders", "mechanics.metal")

    with open(shader_path, "r") as f:
        source = f.read()

    # Repulsion: overlap^2 * sqrt(rep_i * rep_j)
    assert "sqrt(rep_i * rep_j)" in source, (
        "Repulsion should use geometric mean: sqrt(rep_i * rep_j)"
    )

    # Adhesion: -t^2 * sqrt(adh_i * adh_j)
    assert "sqrt(adh_i * adh_j)" in source, (
        "Adhesion should use geometric mean: sqrt(adh_i * adh_j)"
    )

    # Overlap = 1 - dist/R_sum
    assert "1.0f - dist / R_sum" in source, (
        "Overlap formula should be (1 - dist/R_sum)"
    )

    return True, "PhysiCell force: overlap²·√(rep_i·rep_j) − t²·√(adh_i·adh_j)"


# ─────────────────────────────────────────────────────────────────────
# Test 16: Cell mechanics uses unified dispatch (not 4 separate buffers)
# ─────────────────────────────────────────────────────────────────────
def test_cell_mechanics_uses_unified_dispatch(ctx):
    """Verify cell_mechanics.mm uses batched GPU dispatch, not 4 separate single-kernel buffers.

    After the motility fix, mechanics uses two batched dispatches:
      computeForces()        → dispatchForcesOnlyPipeline  (clear+build+forces in one buffer)
      integratePositions()   → dispatchIntegrate           (after CPU motility is applied)
    Forces must still share one command buffer (with memory barriers) to guarantee
    correct spatial-hash state when compute_forces reads it.
    """
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    mm_path = os.path.join(project_dir, "src", "cell_mechanics.mm")

    with open(mm_path, "r") as f:
        source = f.read()

    # Must use the forces-only or full unified pipeline (not 4 separate command buffers)
    assert ("dispatchForcesOnlyPipeline" in source or "dispatchMechanicsPipeline" in source), (
        "cell_mechanics.mm should call dispatchForcesOnlyPipeline or dispatchMechanicsPipeline"
    )

    # Should NOT bypass batching by calling the fine-grained individual dispatchers
    assert "dispatchClearHash" not in source, (
        "cell_mechanics.mm should not call dispatchClearHash separately"
    )
    assert "dispatchBuildHash" not in source, (
        "cell_mechanics.mm should not call dispatchBuildHash separately"
    )
    # Check for the exact individual forces dispatch (not the pipeline variant)
    assert "->dispatchForces(" not in source, (
        "cell_mechanics.mm should not call dispatchForces() individually"
    )

    return True, "Uses unified dispatchMechanicsPipeline"
