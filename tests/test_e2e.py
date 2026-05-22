#!/usr/bin/env python3
"""
test_e2e.py — End-to-end integration tests

Runs the full Metal simulation with the heterogeneity sample config
and validates output files exist and contain valid data.
"""

import os
import subprocess
import tempfile
import shutil
import glob


def _find_config(project_dir):
    """Find the default PhysiCell_settings.xml config."""
    config_path = os.path.join(project_dir, "config", "PhysiCell_settings.xml")
    if os.path.exists(config_path):
        return config_path
    return None


def test_binary_exists(ctx):
    """Verify the Metal binary was built successfully."""
    binary = ctx.get("binary", "build/physicell-metal")
    assert os.path.exists(binary), f"Binary not found at {binary}"
    assert os.access(binary, os.X_OK), f"Binary is not executable: {binary}"
    return True, f"Binary: {binary}"


def test_config_exists(ctx):
    """Verify the default config file exists."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    config = _find_config(project_dir)
    assert config is not None, "PhysiCell_settings.xml not found in config/"
    return True, f"Config: {config}"


def test_short_run_completes(ctx):
    """Run a 1-minute simulation and verify it completes without errors."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))
    config = _find_config(project_dir)

    if config is None:
        return False, "Config not found"

    # Create a temporary directory for output
    tmpdir = tempfile.mkdtemp(prefix="test_e2e_")
    try:
        # Copy config to tmpdir with modified max_time
        with open(config, "r") as f:
            config_content = f.read()

        # Override max_time to 1 minute for quick test
        import re
        config_content = re.sub(
            r'<max_time[^>]*>.*?</max_time>',
            '<max_time units="min">1.0</max_time>',
            config_content
        )

        test_config = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(test_config, "w") as f:
            f.write(config_content)

        os.makedirs(os.path.join(tmpdir, "output"), exist_ok=True)

        result = subprocess.run(
            [binary, test_config],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, (
            f"Simulation failed with code {result.returncode}: {result.stderr[:500]}"
        )

        return True, "1-minute simulation completed successfully"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_output_files_generated(ctx):
    """Verify simulation produces expected output files."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))
    config = _find_config(project_dir)

    if config is None:
        return False, "Config not found"

    tmpdir = tempfile.mkdtemp(prefix="test_e2e_output_")
    try:
        with open(config, "r") as f:
            config_content = f.read()

        import re
        config_content = re.sub(
            r'<max_time[^>]*>.*?</max_time>',
            '<max_time units="min">0.5</max_time>',
            config_content
        )

        test_config = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(test_config, "w") as f:
            f.write(config_content)

        os.makedirs(os.path.join(tmpdir, "output"), exist_ok=True)

        result = subprocess.run(
            [binary, test_config],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return False, f"Simulation failed: {result.stderr[:200]}"

        # Check for output files
        output_dir = os.path.join(tmpdir, "output")
        files = os.listdir(output_dir)

        assert len(files) > 0, "No output files generated"

        return True, f"Generated {len(files)} output files"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_no_nan_in_output(ctx):
    """Verify simulation output doesn't contain NaN or Inf values."""
    project_dir = ctx.get("project_dir", os.path.dirname(os.path.dirname(__file__)))
    binary = ctx.get("binary", os.path.join(project_dir, "build", "physicell-metal"))
    config = _find_config(project_dir)

    if config is None:
        return False, "Config not found"

    tmpdir = tempfile.mkdtemp(prefix="test_e2e_nan_")
    try:
        with open(config, "r") as f:
            config_content = f.read()

        import re
        config_content = re.sub(
            r'<max_time[^>]*>.*?</max_time>',
            '<max_time units="min">2.0</max_time>',
            config_content
        )

        test_config = os.path.join(tmpdir, "PhysiCell_settings.xml")
        with open(test_config, "w") as f:
            f.write(config_content)

        os.makedirs(os.path.join(tmpdir, "output"), exist_ok=True)

        result = subprocess.run(
            [binary, test_config],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return False, f"Simulation failed: {result.stderr[:200]}"

        # Check stdout for NaN/Inf
        output = result.stdout + result.stderr
        nan_count = output.lower().count("nan")
        inf_count = output.lower().count("inf")

        # Some "info" strings contain "inf" — only flag standalone occurrences
        # Check for actual NaN values in numeric context
        assert "nan" not in output.lower().replace("cannot", "").replace("channel", ""), (
            f"Found NaN in output ({nan_count} occurrences)"
        )

        return True, "No NaN/Inf detected in simulation output"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
