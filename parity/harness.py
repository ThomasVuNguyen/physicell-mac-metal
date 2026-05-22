"""Parity harness orchestrator.

Usage:
    python3 -m parity.harness <scenario.json> [--keep]
    python3 -m parity.harness --list

A scenario JSON looks like:

    {
        "name": "heterogeneity_short",
        "description": "Heterogeneity sample, 5 minutes",
        "base_config": "../PhysiCell/config/PhysiCell_settings.xml",
        "ref_binary": "../PhysiCell/heterogeneity",
        "ref_project_root": "../PhysiCell",
        "metal_binary": "build/physicell-metal",
        "metal_project_root": ".",
        "overrides": {
            "max_time": 5,
            "save_interval": 1,
            "omp_num_threads": 1,
            "random_seed": 0
        },
        "tolerances": {
            "position": 1e-6,
            "volume": 1e-3,
            "phase": 0,
            "substrate_linf": 1e-3,
            "substrate_rmse": 1e-4,
            "voxel_coord": 1e-4,
            "cell_count_abs": 0
        },
        "timeout_sec": 600,
        "fail_on_count_drift": true
    }

Paths in the scenario are resolved relative to the scenario file's directory.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

from . import runner
from . import diff as diff_mod


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def _resolve(scenario_dir, p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(scenario_dir, p))


def load_scenario(scenario_path):
    with open(scenario_path) as f:
        sc = json.load(f)
    sd = os.path.dirname(os.path.abspath(scenario_path))
    sc["_path"] = scenario_path
    sc["_dir"] = sd
    sc["base_config"] = _resolve(sd, sc["base_config"])
    sc["ref_binary"] = _resolve(sd, sc["ref_binary"])
    sc["ref_project_root"] = _resolve(sd, sc.get("ref_project_root",
                                                  os.path.dirname(sc["ref_binary"])))
    sc["metal_binary"] = _resolve(sd, sc.get("metal_binary",
                                              os.path.join(PROJECT_ROOT, "build", "physicell-metal")))
    sc["metal_project_root"] = _resolve(sd, sc.get("metal_project_root", PROJECT_ROOT))
    sc.setdefault("overrides", {})
    sc.setdefault("tolerances", {})
    sc.setdefault("timeout_sec", 900)
    sc.setdefault("fail_on_count_drift", False)
    return sc


def run_scenario(sc, workdir=None, keep=False, verbose=False):
    """Run one scenario end-to-end. Returns (ok, report_text)."""
    name = sc.get("name", "scenario")
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix=f"parity_{name}_")
        created_workdir = True
    else:
        os.makedirs(workdir, exist_ok=True)
        created_workdir = False

    ref_out = os.path.join(workdir, "ref_output")
    metal_out = os.path.join(workdir, "metal_output")
    os.makedirs(ref_out, exist_ok=True)
    os.makedirs(metal_out, exist_ok=True)

    report = []
    report.append(f"=== {name} ===")
    report.append(f"description : {sc.get('description', '')}")
    report.append(f"base_config : {sc['base_config']}")
    report.append(f"ref_binary  : {sc['ref_binary']}")
    report.append(f"metal_binary: {sc['metal_binary']}")
    report.append(f"workdir     : {workdir}")
    report.append(f"overrides   : {json.dumps(sc['overrides'])}")
    report.append(f"tolerances  : {json.dumps(sc['tolerances'])}")
    report.append("")

    overall_ok = True

    # --- Reference run ---
    report.append("--- reference (PhysiCell) ---")
    ref_log = os.path.join(workdir, "ref.log")
    try:
        rres = runner.run_reference(
            ref_binary=sc["ref_binary"],
            base_config=sc["base_config"],
            output_dir=ref_out,
            log_path=ref_log,
            overrides=sc["overrides"],
            config_dir=sc["ref_project_root"],
            timeout=sc["timeout_sec"],
        )
    except Exception as e:
        report.append(f"  ERROR launching reference: {e}")
        return False, "\n".join(report)
    report.append(f"  rc={rres.returncode}  elapsed={rres.elapsed:.2f}s  log={rres.log_path}")
    if not rres.ok:
        report.append("  reference run FAILED — stderr tail:")
        for line in rres.stderr_tail.splitlines()[-20:]:
            report.append(f"    | {line}")
        overall_ok = False

    # --- Metal run ---
    report.append("")
    report.append("--- metal (physicell-metal) ---")
    metal_log = os.path.join(workdir, "metal.log")
    try:
        mres = runner.run_metal(
            metal_binary=sc["metal_binary"],
            metal_project_root=sc["metal_project_root"],
            base_config=sc["base_config"],
            output_dir=metal_out,
            log_path=metal_log,
            overrides=sc["overrides"],
            timeout=sc["timeout_sec"],
        )
    except Exception as e:
        report.append(f"  ERROR launching metal: {e}")
        return False, "\n".join(report)
    report.append(f"  rc={mres.returncode}  elapsed={mres.elapsed:.2f}s  log={mres.log_path}")
    if not mres.ok:
        report.append("  metal run FAILED — stderr tail:")
        for line in mres.stderr_tail.splitlines()[-20:]:
            report.append(f"    | {line}")
        overall_ok = False

    if not overall_ok:
        return False, "\n".join(report)

    # --- Diff snapshots ---
    report.append("")
    report.append("--- snapshots ---")
    ref_frames = runner.discover_snapshots(ref_out)
    metal_frames = runner.discover_snapshots(metal_out)
    common = sorted(set(ref_frames) & set(metal_frames))
    only_ref = sorted(set(ref_frames) - set(metal_frames))
    only_metal = sorted(set(metal_frames) - set(ref_frames))
    report.append(f"  ref frames   : {len(ref_frames)} {ref_frames[:5]}…")
    report.append(f"  metal frames : {len(metal_frames)} {metal_frames[:5]}…")
    report.append(f"  common       : {len(common)}")
    if only_ref:
        report.append(f"  only_ref     : {only_ref[:10]}")
    if only_metal:
        report.append(f"  only_metal   : {only_metal[:10]}")
    if not common:
        report.append("  no overlapping snapshots — cannot diff")
        return False, "\n".join(report)

    speedup = (rres.elapsed / mres.elapsed) if mres.elapsed > 0 else float("inf")
    report.append(f"  perf         : ref={rres.elapsed:.2f}s  metal={mres.elapsed:.2f}s  "
                  f"speedup={speedup:.2f}x")
    report.append("")

    n_fail = 0
    for idx in common:
        snap = diff_mod.diff_snapshot_pair(ref_out, metal_out, idx, sc["tolerances"])
        if not snap.ok:
            n_fail += 1
        report.append(diff_mod.format_snapshot(snap))

    report.append("")
    report.append(f"=== {name}: {len(common) - n_fail}/{len(common)} snapshots passed ===")

    if n_fail > 0:
        overall_ok = False

    if created_workdir and not keep:
        shutil.rmtree(workdir, ignore_errors=True)

    return overall_ok, "\n".join(report)


def main():
    ap = argparse.ArgumentParser(description="PhysiCell ↔ Metal parity harness")
    ap.add_argument("scenario", nargs="?", help="path to scenario JSON")
    ap.add_argument("--list", action="store_true", help="list bundled scenarios")
    ap.add_argument("--keep", action="store_true", help="keep tmp workdir after run")
    ap.add_argument("--workdir", help="explicit workdir (kept by default)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    scen_dir = os.path.join(HERE, "scenarios")

    if args.list:
        if not os.path.isdir(scen_dir):
            print("(no scenarios directory)")
            return 0
        for f in sorted(os.listdir(scen_dir)):
            if f.endswith(".json"):
                p = os.path.join(scen_dir, f)
                try:
                    sc = load_scenario(p)
                    print(f"  {f:35s}  {sc.get('description', '')}")
                except Exception as e:
                    print(f"  {f:35s}  (load error: {e})")
        return 0

    if not args.scenario:
        ap.print_help()
        return 2

    # If a bare name was passed, try resolving it in scenarios/
    sp = args.scenario
    if not os.path.isfile(sp):
        cand = os.path.join(scen_dir, sp)
        if os.path.isfile(cand):
            sp = cand
        elif os.path.isfile(cand + ".json"):
            sp = cand + ".json"
        else:
            print(f"scenario not found: {args.scenario}")
            return 2

    sc = load_scenario(sp)
    ok, report = run_scenario(sc, workdir=args.workdir,
                              keep=args.keep or bool(args.workdir),
                              verbose=args.verbose)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
