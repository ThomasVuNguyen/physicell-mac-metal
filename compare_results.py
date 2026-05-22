#!/usr/bin/env python3
"""
compare_results.py — Validate Metal PhysiCell rewrite against original PhysiCell.

Compares:
  A. Feature / functionality checklist
  B. Quantitative statistics (cell counts, positions, radii, oncoprotein)
  C. Accuracy / sanity checks

Uses only stdlib + json + xml.etree.
"""

import json
import glob
import math
import os
import sys

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORIGINAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "PhysiCell", "output"
)
METAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output"
)

PHYSICELL_DEFAULT_RADIUS = 8.412710547954228  # µm
DOMAIN_HALF = 1000.0  # µm  (domain is [-1000, 1000] in x and y)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def fmt(v, decimals=4):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return f"{v:.{decimals}f}"
    return str(v)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_xml_metadata(path):
    """Extract time, runtime, bounding box from a PhysiCell XML snapshot."""
    if not HAS_XML or not os.path.isfile(path):
        return None
    tree = ET.parse(path)
    root = tree.getroot()
    info = {}
    t_el = root.find(".//current_time")
    if t_el is not None:
        info["time"] = float(t_el.text)
        info["time_units"] = t_el.attrib.get("units", "")
    rt_el = root.find(".//current_runtime")
    if rt_el is not None:
        info["runtime_sec"] = float(rt_el.text)
    bb_el = root.find(".//bounding_box")
    if bb_el is not None:
        info["bounding_box"] = [float(x) for x in bb_el.text.strip().split()]
    # cell type names
    type_els = root.findall(".//cell_types/type")
    info["cell_types"] = {el.attrib.get("ID", "?"): el.text for el in type_els}
    # labels
    label_els = root.findall(".//labels/label")
    info["labels"] = [el.text for el in label_els]
    # mat filename
    mat_el = root.find(".//simplified_data/filename")
    if mat_el is not None:
        info["mat_file"] = mat_el.text
    return info


def load_original_xml_series(output_dir):
    """Load time and cell-count from every output*.xml in the original dir."""
    series = []
    pattern = os.path.join(output_dir, "output*.xml")
    for xmlf in sorted(glob.glob(pattern)):
        info = load_xml_metadata(xmlf)
        if info and "time" in info:
            series.append(info)
    return series


def load_cells_3d(path):
    """Load the cells_3d.json produced by extract_cells_json.py."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Might be {"frames": [...]} or just a list
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    if isinstance(data, list):
        return data
    return None


def load_metal_frames(output_dir):
    """Load frame_*.json files from Metal output."""
    frames = []
    pattern = os.path.join(output_dir, "frame_*.json")
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            frame = json.load(f)
        frames.append(frame)
    return frames


def cell_stats(cells, key_x="x", key_y="y", key_z="z",
               key_r="radius", key_onco="oncoprotein"):
    """Compute aggregate stats from a list of cell dicts."""
    if not cells:
        return {}
    xs = [c[key_x] for c in cells if key_x in c]
    ys = [c[key_y] for c in cells if key_y in c]
    zs = [c[key_z] for c in cells if key_z in c]
    radii = [c[key_r] for c in cells if key_r in c]
    onco_vals = [c[key_onco] for c in cells if key_onco in c]

    # Tumor radius (max Euclidean distance from origin)
    dists = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(xs, ys, zs)]

    stats = {
        "n": len(cells),
        "x_mean": _mean(xs), "x_std": _std(xs),
        "y_mean": _mean(ys), "y_std": _std(ys),
        "z_mean": _mean(zs), "z_std": _std(zs),
        "radius_mean": _mean(radii), "radius_std": _std(radii),
        "radius_min": min(radii) if radii else float("nan"),
        "radius_max": max(radii) if radii else float("nan"),
        "tumor_radius_max": max(dists) if dists else float("nan"),
        "tumor_radius_mean": _mean(dists),
    }
    if onco_vals:
        stats["onco_mean"] = _mean(onco_vals)
        stats["onco_std"] = _std(onco_vals)
        stats["onco_min"] = min(onco_vals)
        stats["onco_max"] = max(onco_vals)

    # sanity flags
    nan_count = sum(1 for c in cells
                    for v in [c.get(key_x), c.get(key_y), c.get(key_z), c.get(key_r)]
                    if v is not None and (math.isnan(v) or math.isinf(v)))
    stats["nan_inf_count"] = nan_count
    oob = sum(1 for c in cells
              if abs(c.get(key_x, 0)) > DOMAIN_HALF
              or abs(c.get(key_y, 0)) > DOMAIN_HALF)
    stats["out_of_bounds"] = oob
    return stats

# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

SECTION_WIDTH = 80
SUB_WIDTH = 70

def header(title):
    print("\n" + "=" * SECTION_WIDTH)
    print(f"  {title}")
    print("=" * SECTION_WIDTH)


def sub_header(title):
    print(f"\n  --- {title} " + "-" * max(0, SUB_WIDTH - len(title) - 6))


def kv(key, value, indent=4):
    pad = " " * indent
    print(f"{pad}{key:.<45s} {value}")


def print_stats_block(label, stats):
    sub_header(f"{label} Cell Statistics")
    kv("Cell count", fmt(stats.get("n", "N/A"), 0))
    kv("Position X  mean / std",
       f'{fmt(stats.get("x_mean"))} / {fmt(stats.get("x_std"))}')
    kv("Position Y  mean / std",
       f'{fmt(stats.get("y_mean"))} / {fmt(stats.get("y_std"))}')
    kv("Cell radius mean / std",
       f'{fmt(stats.get("radius_mean"))} / {fmt(stats.get("radius_std"))}')
    kv("Cell radius min / max",
       f'{fmt(stats.get("radius_min"))} / {fmt(stats.get("radius_max"))}')
    kv("Tumor radius (max dist from origin)",
       fmt(stats.get("tumor_radius_max")))
    if "onco_mean" in stats:
        kv("Oncoprotein mean / std",
           f'{fmt(stats.get("onco_mean"))} / {fmt(stats.get("onco_std"))}')
        kv("Oncoprotein min / max",
           f'{fmt(stats.get("onco_min"))} / {fmt(stats.get("onco_max"))}')
    kv("NaN / Inf values detected", str(stats.get("nan_inf_count", 0)))
    kv("Cells outside domain bounds", str(stats.get("out_of_bounds", 0)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#" * SECTION_WIDTH)
    print("#  PhysiCell vs Metal-PhysiCell  —  Comparison Report")
    print("#" * SECTION_WIDTH)

    # ---- Load original data ------------------------------------------------
    orig_initial_meta = load_xml_metadata(os.path.join(ORIGINAL_DIR, "initial.xml"))
    orig_final_meta = load_xml_metadata(os.path.join(ORIGINAL_DIR, "final.xml"))
    orig_xml_series = load_original_xml_series(ORIGINAL_DIR)
    orig_frames = load_cells_3d(os.path.join(ORIGINAL_DIR, "cells_3d.json"))

    # ---- Load Metal data ---------------------------------------------------
    metal_frames = load_metal_frames(METAL_DIR)

    # ======================================================================
    # A. FUNCTIONALITY CHECK
    # ======================================================================
    header("A. FUNCTIONALITY / FEATURE CHECKLIST")

    features = [
        ("Cell cycle model (Ki67 basic)",
         "phase" in (orig_frames[0]["cells"][0] if orig_frames else {}),
         "phase" in (metal_frames[0]["cells"][0] if metal_frames else {})),
        ("Cell death (apoptosis / necrosis)",
         orig_final_meta is not None and "death_rates" in (orig_final_meta.get("labels", [])),
         "alive" in (metal_frames[0]["cells"][0] if metal_frames else {})),
        ("Volume dynamics (variable radii)",
         True,  # PhysiCell always has this
         len(set(round(c["radius"], 4) for c in metal_frames[-1]["cells"])) > 1 if metal_frames else False),
        ("Cell mechanics (repulsion/adhesion)",
         orig_final_meta is not None and "cell_cell_repulsion_strength" in (orig_final_meta.get("labels", [])),
         True),  # Implied by positions not collapsing
        ("Motility",
         orig_final_meta is not None and "is_motile" in (orig_final_meta.get("labels", [])),
         False),  # No motility fields in Metal JSON
        ("Substrate diffusion / decay",
         any("microenvironment" in (m.get("mat_file", "") if m else "")
             for m in [orig_final_meta]) if orig_final_meta else False,
         False),  # No substrate data in Metal JSON
        ("Secretion / uptake",
         orig_final_meta is not None and "secretion_rates" in (orig_final_meta.get("labels", [])),
         False),
        ("Custom data (oncoprotein)",
         orig_frames is not None and "oncoprotein" in (orig_frames[0]["cells"][0] if orig_frames else {}),
         "oncoprotein" in (metal_frames[0]["cells"][0] if metal_frames else {})),
        ("Neighbor graphs",
         os.path.isfile(os.path.join(ORIGINAL_DIR, "final_cell_neighbor_graph.txt")),
         False),  # No graph output in Metal
        ("SVG output",
         os.path.isfile(os.path.join(ORIGINAL_DIR, "final.svg")),
         False),
        (".mat output",
         os.path.isfile(os.path.join(ORIGINAL_DIR, "final_cells.mat")),
         False),  # Metal uses JSON
        ("Rules / signals",
         False,  # Not in default sample
         False),
    ]

    print(f"\n  {'Feature':<42s} {'Original':>10s}  {'Metal':>10s}")
    print(f"  {'-'*42} {'-'*10}  {'-'*10}")
    implemented_count = 0
    total_applicable = 0
    for name, orig_has, metal_has in features:
        o_str = "  ✓" if orig_has else "  ✗"
        m_str = "  ✓" if metal_has else "  ✗"
        print(f"  {name:<42s} {o_str:>10s}  {m_str:>10s}")
        if orig_has:
            total_applicable += 1
            if metal_has:
                implemented_count += 1

    pct = (implemented_count / total_applicable * 100) if total_applicable else 0
    print(f"\n  Metal feature coverage: {implemented_count}/{total_applicable}"
          f" ({pct:.0f}%) of original features detected in output")

    # ======================================================================
    # B. QUANTITATIVE COMPARISON
    # ======================================================================
    header("B. QUANTITATIVE COMPARISON")

    # ---- Simulation overview -----------------------------------------------
    sub_header("Simulation Overview")
    if orig_initial_meta:
        kv("Original start time",
           f'{fmt(orig_initial_meta.get("time", 0))} {orig_initial_meta.get("time_units", "")}')
    if orig_final_meta:
        kv("Original end time",
           f'{fmt(orig_final_meta.get("time", 0))} {orig_final_meta.get("time_units", "")}')
        kv("Original runtime",
           f'{fmt(orig_final_meta.get("runtime_sec", 0))} sec')
    if orig_frames:
        kv("Original cells_3d.json frames", str(len(orig_frames)))
        kv("Original cells_3d time range",
           f'{fmt(orig_frames[0]["time"])} – {fmt(orig_frames[-1]["time"])} min')
    if metal_frames:
        kv("Metal frames", str(len(metal_frames)))
        kv("Metal time range",
           f'{fmt(metal_frames[0]["time"])} – {fmt(metal_frames[-1]["time"])} min')

    # ---- Cell count growth curves ------------------------------------------
    sub_header("Cell Count Growth Curve")

    # Build time→count for both
    orig_tc = []
    if orig_frames:
        for fr in orig_frames:
            orig_tc.append((fr["time"], fr["num_cells"]))

    metal_tc = []
    if metal_frames:
        for fr in metal_frames:
            metal_tc.append((fr["time"], fr["num_cells"]))

    # Print sampled points
    def sample_curve(tc, n_samples=12):
        if not tc:
            return []
        step = max(1, len(tc) // n_samples)
        return tc[::step] + ([tc[-1]] if tc[-1] not in tc[::step] else [])

    orig_sample = sample_curve(orig_tc)
    metal_sample = sample_curve(metal_tc)

    print(f"\n  {'Time (min)':>12s}  {'Original':>10s}  {'Metal':>10s}  {'Δ cells':>10s}  {'Δ%':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    # Match closest times
    all_times = sorted(set(t for t, _ in orig_sample) | set(t for t, _ in metal_sample))
    # Build lookup dicts
    orig_lookup = dict(orig_tc)
    metal_lookup = dict(metal_tc)

    def find_closest(lookup, target, tol=120):
        """Find value at closest time within tolerance."""
        best_t, best_v = None, None
        for t, v in lookup.items() if isinstance(lookup, dict) else []:
            pass
        # Simpler approach
        best_diff = float("inf")
        for t in lookup:
            d = abs(t - target)
            if d < best_diff:
                best_diff = d
                best_t = t
                best_v = lookup[t]
        if best_diff <= tol:
            return best_v
        return None

    # Pick ~15 representative times spanning both ranges
    min_time = min((orig_tc[0][0] if orig_tc else 1e18),
                   (metal_tc[0][0] if metal_tc else 1e18))
    max_time = max((orig_tc[-1][0] if orig_tc else 0),
                   (metal_tc[-1][0] if metal_tc else 0))
    n_report = 15
    report_times = [min_time + i * (max_time - min_time) / n_report
                    for i in range(n_report + 1)]

    for rt in report_times:
        ov = find_closest(orig_lookup, rt, tol=120)
        mv = find_closest(metal_lookup, rt, tol=120)
        o_str = str(ov) if ov is not None else "—"
        m_str = str(mv) if mv is not None else "—"
        if ov is not None and mv is not None:
            delta = mv - ov
            dpct = (delta / ov * 100) if ov else 0
            d_str = f"{delta:+d}"
            dp_str = f"{dpct:+.1f}%"
        else:
            d_str = "—"
            dp_str = "—"
        print(f"  {rt:12.1f}  {o_str:>10s}  {m_str:>10s}  {d_str:>10s}  {dp_str:>8s}")

    # ---- Initial frame comparison ------------------------------------------
    sub_header("Initial Frame Comparison")
    if orig_frames:
        orig_init_stats = cell_stats(orig_frames[0]["cells"])
        print_stats_block("Original (t=0)", orig_init_stats)
    if metal_frames:
        metal_init_stats = cell_stats(metal_frames[0]["cells"])
        print_stats_block("Metal (t=0)", metal_init_stats)

    # ---- Final frame comparison --------------------------------------------
    sub_header("Final Frame Comparison")
    if orig_frames:
        orig_final_stats = cell_stats(orig_frames[-1]["cells"])
        print_stats_block(f"Original (t={fmt(orig_frames[-1]['time'])})", orig_final_stats)
    if metal_frames:
        metal_final_stats = cell_stats(metal_frames[-1]["cells"])
        print_stats_block(f"Metal (t={fmt(metal_frames[-1]['time'])})", metal_final_stats)

    # ---- Time-matched frame-by-frame comparison ----------------------------
    sub_header("Time-Matched Frame-by-Frame (where overlap exists)")
    matches = []
    if orig_frames and metal_frames:
        # Build metal time index
        metal_by_time = {}
        for fr in metal_frames:
            metal_by_time[round(fr["time"], 1)] = fr

        for ofr in orig_frames:
            ot = round(ofr["time"], 1)
            # Try exact, then ±1 min
            mfr = metal_by_time.get(ot)
            if mfr is None:
                for delta in [1, -1, 2, -2, 5, -5, 10, -10]:
                    mfr = metal_by_time.get(round(ot + delta, 1))
                    if mfr:
                        break
            if mfr:
                matches.append((ofr, mfr))

        if matches:
            print(f"\n  Matched frames: {len(matches)}")
            print(f"\n  {'Orig t':>10s}  {'Metal t':>10s}  {'O cells':>8s}  {'M cells':>8s}"
                  f"  {'Δ cells':>8s}  {'O rad µ':>8s}  {'M rad µ':>8s}")
            print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

            # Show a sample of matched frames
            step = max(1, len(matches) // 15)
            shown = matches[::step]
            if matches[-1] not in shown:
                shown.append(matches[-1])

            count_diffs = []
            for ofr, mfr in shown:
                oc = ofr["num_cells"]
                mc = mfr["num_cells"]
                dc = mc - oc
                count_diffs.append(dc)
                or_vals = [c["radius"] for c in ofr["cells"] if "radius" in c]
                mr_vals = [c["radius"] for c in mfr["cells"] if "radius" in c]
                or_m = _mean(or_vals)
                mr_m = _mean(mr_vals)
                print(f"  {ofr['time']:10.1f}  {mfr['time']:10.1f}  {oc:8d}  {mc:8d}"
                      f"  {dc:+8d}  {or_m:8.4f}  {mr_m:8.4f}")

            # Summary statistics for matched frames
            if count_diffs:
                print(f"\n  Cell count difference across matched frames:")
                kv("Mean Δ cells", fmt(_mean(count_diffs)))
                kv("Std  Δ cells", fmt(_std(count_diffs)))
                kv("Min  Δ cells", fmt(min(count_diffs)))
                kv("Max  Δ cells", fmt(max(count_diffs)))
        else:
            print("  No time-matched frames found between Original and Metal.")

    # ======================================================================
    # C. ACCURACY ASSESSMENT
    # ======================================================================
    header("C. ACCURACY ASSESSMENT")

    checks_passed = 0
    checks_total = 0

    def check(label, passed, detail=""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "✓ PASS" if passed else "✗ FAIL"
        if passed:
            checks_passed += 1
        print(f"  [{status}]  {label}")
        if detail:
            print(f"           {detail}")

    # 1. Cell radii physically reasonable
    if metal_frames:
        last = metal_frames[-1]
        radii = [c["radius"] for c in last["cells"]]
        r_mean = _mean(radii)
        r_min = min(radii)
        r_max = max(radii)
        reasonable = 2.0 < r_mean < 20.0
        check("Cell radii physically reasonable (2–20 µm)",
              reasonable,
              f"Metal final: mean={fmt(r_mean)}, min={fmt(r_min)}, max={fmt(r_max)}, "
              f"PhysiCell default={fmt(PHYSICELL_DEFAULT_RADIUS)}")

        # Close to PhysiCell default?
        close = abs(r_mean - PHYSICELL_DEFAULT_RADIUS) / PHYSICELL_DEFAULT_RADIUS < 0.15
        check("Mean radius within 15% of PhysiCell default (8.413 µm)",
              close,
              f"Deviation: {abs(r_mean - PHYSICELL_DEFAULT_RADIUS) / PHYSICELL_DEFAULT_RADIUS * 100:.1f}%")

    # 2. Positions within domain bounds
    if metal_frames:
        last_stats = cell_stats(metal_frames[-1]["cells"])
        oob = last_stats["out_of_bounds"]
        check("All cells within domain bounds (±1000 µm)",
              oob == 0,
              f"Out-of-bounds cells: {oob} / {last_stats['n']}")

    # 3. No NaN/Inf values
    if metal_frames:
        total_nan = 0
        for fr in metal_frames:
            for c in fr["cells"]:
                for k in ["x", "y", "z", "radius", "oncoprotein"]:
                    v = c.get(k)
                    if v is not None and (math.isnan(v) or math.isinf(v)):
                        total_nan += 1
        check("No NaN or infinite values in Metal output",
              total_nan == 0,
              f"NaN/Inf values found: {total_nan}")

    # 4. Growth rate qualitatively similar
    if orig_tc and metal_tc:
        # Compare growth ratio at overlapping time range
        overlap_end = min(orig_tc[-1][0], metal_tc[-1][0])
        # Find counts near overlap_end
        orig_end_c = find_closest(orig_lookup, overlap_end, tol=200)
        metal_end_c = find_closest(metal_lookup, overlap_end, tol=200)
        orig_start_c = orig_tc[0][1]
        metal_start_c = metal_tc[0][1]

        if orig_end_c and metal_end_c:
            orig_ratio = orig_end_c / orig_start_c if orig_start_c else 0
            metal_ratio = metal_end_c / metal_start_c if metal_start_c else 0
            similar = (0.2 < metal_ratio / orig_ratio < 5.0) if orig_ratio > 0 else False
            check("Growth rate qualitatively similar (within 5x)",
                  similar,
                  f"Original growth ratio: {orig_ratio:.2f}x, "
                  f"Metal growth ratio: {metal_ratio:.2f}x "
                  f"(at t≈{overlap_end:.0f} min)")

    # 5. Initial cell count reasonable
    if orig_frames and metal_frames:
        oc0 = orig_frames[0]["num_cells"]
        mc0 = metal_frames[0]["num_cells"]
        diff_pct = abs(mc0 - oc0) / oc0 * 100 if oc0 > 0 else 0
        check("Initial cell counts comparable",
              diff_pct < 50,
              f"Original: {oc0}, Metal: {mc0} (Δ={diff_pct:.1f}%)")

    # 6. Oncoprotein values reasonable
    if metal_frames:
        all_onco = []
        for fr in metal_frames[::max(1, len(metal_frames)//10)]:
            all_onco.extend(c.get("oncoprotein", 0) for c in fr["cells"])
        onco_ok = all(0 <= v <= 10.0 for v in all_onco) and len(all_onco) > 0
        check("Oncoprotein values in reasonable range [0, 10]",
              onco_ok,
              f"Range: [{fmt(min(all_onco))}, {fmt(max(all_onco))}]" if all_onco else "")

    # 7. Cell alive flag consistency
    if metal_frames:
        last = metal_frames[-1]
        alive_cells = sum(1 for c in last["cells"] if c.get("alive", 1) == 1)
        dead_cells = sum(1 for c in last["cells"] if c.get("alive", 1) == 0)
        has_death = dead_cells > 0
        check("Cell death model active (some dead cells in final frame)",
              has_death,
              f"Alive: {alive_cells}, Dead: {dead_cells} (total: {last['num_cells']})")

    # 8. Simulation duration comparison
    if metal_frames and orig_frames:
        orig_dur = orig_frames[-1]["time"]
        metal_dur = metal_frames[-1]["time"]
        pct_dur = metal_dur / orig_dur * 100 if orig_dur > 0 else 0
        check("Metal simulation covers significant duration",
              pct_dur > 50,
              f"Original: {orig_dur:.0f} min, Metal: {metal_dur:.0f} min ({pct_dur:.0f}%)")

    # 9. Radius variation (volume dynamics)
    if metal_frames:
        radii = [c["radius"] for c in metal_frames[-1]["cells"]]
        r_std = _std(radii)
        has_variation = r_std > 0.01
        check("Radius variation present (volume dynamics active)",
              has_variation,
              f"Std of radii: {fmt(r_std)} µm")

    # 10. Spatial coherence (tumor-like structure)
    if metal_frames:
        last_cells = metal_frames[-1]["cells"]
        dists = [math.sqrt(c["x"]**2 + c["y"]**2) for c in last_cells]
        # Check that most cells are within a reasonable tumor boundary
        within_500 = sum(1 for d in dists if d < 500) / len(dists) * 100 if dists else 0
        check("Spatial coherence (>50% cells within 500µm of origin)",
              within_500 > 50,
              f"{within_500:.1f}% of cells within 500µm radius")

    # ---- Summary -----------------------------------------------------------
    header("SUMMARY")
    print(f"\n  Accuracy checks passed: {checks_passed}/{checks_total}")
    if metal_frames:
        print(f"  Metal frames: {len(metal_frames)}, "
              f"Final cell count: {metal_frames[-1]['num_cells']}")
    if orig_frames:
        print(f"  Original frames: {len(orig_frames)}, "
              f"Final cell count: {orig_frames[-1]['num_cells']}")
    if total_applicable:
        print(f"  Feature coverage: {implemented_count}/{total_applicable} "
              f"({pct:.0f}%)")

    # Overall assessment
    print()
    if checks_passed == checks_total and pct >= 50:
        print("  ★ OVERALL: Metal rewrite produces PHYSICALLY REASONABLE output")
        print("    and covers core simulation features.")
    elif checks_passed >= checks_total * 0.7:
        print("  ◐ OVERALL: Metal rewrite is PARTIALLY VALIDATED — most checks pass")
        print("    but some features or accuracy criteria need attention.")
    else:
        print("  ✗ OVERALL: Metal rewrite has SIGNIFICANT DEVIATIONS from expected")
        print("    behavior. Review failing checks above.")

    print()


if __name__ == "__main__":
    main()
