"""test_parity.py — runs PhysiCell ↔ Metal parity scenarios.

Each scenario in parity/scenarios/ is run end-to-end:
  1. PhysiCell reference binary is invoked with the scenario config.
  2. The Metal binary is invoked with the same config.
  3. Per-snapshot .mat outputs are diffed under the scenario's tolerances.

A scenario passes when all paired snapshots are within tolerance.
Run with: python3 tests/run_tests.py --parity
"""

import os
import sys

# Add parity package to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from parity import harness as parity_harness


def _scenarios():
    scen_dir = os.path.join(_PROJECT_ROOT, "parity", "scenarios")
    if not os.path.isdir(scen_dir):
        return []
    return sorted([
        os.path.join(scen_dir, f)
        for f in os.listdir(scen_dir)
        if f.endswith(".json")
    ])


def _run_one(scenario_path):
    """Load scenario, run it, return (passed, message)."""
    try:
        sc = parity_harness.load_scenario(scenario_path)
    except Exception as e:
        return False, f"failed to load scenario: {e}"

    # Pre-flight: skip cleanly if reference binary is missing.
    if not os.path.isfile(sc["ref_binary"]):
        return True, f"SKIPPED (ref binary missing: {sc['ref_binary']})"
    if not os.path.isfile(sc["metal_binary"]):
        return False, f"metal binary missing: {sc['metal_binary']}"

    ok, report = parity_harness.run_scenario(sc, keep=False)
    if ok:
        return True, "all snapshots within tolerance"
    # On failure, surface the summary line + any FAIL diagnostics.
    lines = report.splitlines()
    summary = next((l for l in reversed(lines) if l.startswith("===")), "no summary")
    # Plus a few sample FAIL field lines for context.
    fail_lines = [l for l in lines if l.strip().endswith("FAIL")][:6]
    msg = summary + ("\n      " + "\n      ".join(fail_lines) if fail_lines else "")
    return False, msg


# Dynamically register test_<scenario_name> functions so the test runner
# discovers them via the standard test_* convention.
def _make_test(scenario_path):
    name = os.path.splitext(os.path.basename(scenario_path))[0]
    def _t(ctx=None):
        return _run_one(scenario_path)
    _t.__name__ = f"test_{name}"
    _t.__doc__ = f"Parity scenario: {name}"
    return _t


# Inject one test_* function per scenario into module globals.
for _sp in _scenarios():
    _fn = _make_test(_sp)
    globals()[_fn.__name__] = _fn
