"""Snapshot-level numerical diffs for PhysiCell ↔ Metal parity.

Compares paired .mat outputs:
  - Cells: ID-matched per-cell scalar fields (position, volume, phase).
  - Microenvironment: voxel-aligned substrate densities (L∞ + L2 norms).

Tolerances are passed in by the caller (per scenario). Each comparator
returns a Result with .ok plus diagnostic numbers, never raises on
mismatch — the harness aggregates Results into a report.

Canonical PhysiCell BioFVM cell-row schema (used by both binaries for
the leading rows):
    row 0 : ID
    row 1 : pos_x
    row 2 : pos_y
    row 3 : pos_z
    row 4 : total_volume
    row 5 : cell_type
    row 6 : cycle_model
    row 7 : current_phase (index)

We only diff the leading rows we know are stable across both schemas;
total row counts are reported separately so a schema drift is visible.

Microenvironment row schema (both):
    row 0 : voxel_x
    row 1 : voxel_y
    row 2 : voxel_z
    row 3 : voxel_volume
    row 4+: substrate densities
"""

import math
from dataclasses import dataclass, field

from . import mat_reader


# --- Canonical row offsets (shared by PhysiCell + Metal ports) ---
CELL_ROW_ID = 0
CELL_ROW_X = 1
CELL_ROW_Y = 2
CELL_ROW_Z = 3
CELL_ROW_TOTAL_VOLUME = 4
CELL_ROW_CELL_TYPE = 5
CELL_ROW_CYCLE_MODEL = 6
CELL_ROW_CURRENT_PHASE = 7

ME_ROW_X = 0
ME_ROW_Y = 1
ME_ROW_Z = 2
ME_ROW_VOXEL_VOLUME = 3
ME_FIRST_SUBSTRATE_ROW = 4


@dataclass
class FieldStat:
    """Pairwise per-cell error stats for one scalar field."""
    name: str
    n_compared: int = 0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    rmse: float = 0.0
    ref_min: float = 0.0
    ref_max: float = 0.0
    ok: bool = True
    tolerance: float = 0.0

    def fmt(self):
        return (f"  {self.name:<15s}  n={self.n_compared:<6d}  "
                f"L∞={self.max_abs_err:.6g}  RMSE={self.rmse:.6g}  "
                f"ref∈[{self.ref_min:.4g}, {self.ref_max:.4g}]  "
                f"tol={self.tolerance:.4g}  {'PASS' if self.ok else 'FAIL'}")


@dataclass
class CellDiff:
    ok: bool = True
    n_ref: int = 0
    n_metal: int = 0
    matched: int = 0
    only_ref: int = 0       # IDs in ref but not in metal
    only_metal: int = 0     # IDs in metal but not in ref
    ref_rows: int = 0
    metal_rows: int = 0
    fields: list = field(default_factory=list)  # list[FieldStat]
    notes: list = field(default_factory=list)


@dataclass
class MicroDiff:
    ok: bool = True
    n_voxels_ref: int = 0
    n_voxels_metal: int = 0
    ref_rows: int = 0
    metal_rows: int = 0
    voxels_aligned: bool = False
    substrate_stats: list = field(default_factory=list)  # list[FieldStat]
    notes: list = field(default_factory=list)


@dataclass
class SnapshotDiff:
    label: str = ""
    cells: CellDiff = None
    micro: MicroDiff = None

    @property
    def ok(self):
        return ((self.cells is None or self.cells.ok) and
                (self.micro is None or self.micro.ok))


# --- Helpers ---

def _accumulate_field_errors(name, ref_vals, metal_vals, tol):
    """Compute L∞, mean, RMSE for paired sequences. Returns FieldStat."""
    n = len(ref_vals)
    assert n == len(metal_vals), "internal: paired sequences must align"
    if n == 0:
        return FieldStat(name=name, n_compared=0, tolerance=tol)
    max_e = 0.0
    sum_e = 0.0
    sum_sq = 0.0
    rmin = float("inf")
    rmax = float("-inf")
    for r, m in zip(ref_vals, metal_vals):
        e = abs(r - m)
        if e > max_e:
            max_e = e
        sum_e += e
        sum_sq += e * e
        if r < rmin:
            rmin = r
        if r > rmax:
            rmax = r
    return FieldStat(
        name=name,
        n_compared=n,
        max_abs_err=max_e,
        mean_abs_err=sum_e / n,
        rmse=math.sqrt(sum_sq / n),
        ref_min=rmin,
        ref_max=rmax,
        ok=(max_e <= tol),
        tolerance=tol,
    )


# --- Public diff entry points ---

def diff_cells(ref_path, metal_path, tolerances):
    """Compare two cells.mat files.

    `tolerances` keys (all absolute, in cell-data units):
        position : default 1e-6  (microns)
        volume   : default 1e-3  (cubic microns)
        phase    : default 0     (integer phase index, must match exactly)
    """
    out = CellDiff()
    try:
        ref = mat_reader.read_mat(ref_path)
    except Exception as e:
        out.ok = False
        out.notes.append(f"failed to read ref {ref_path}: {e}")
        return out
    try:
        metal = mat_reader.read_mat(metal_path)
    except Exception as e:
        out.ok = False
        out.notes.append(f"failed to read metal {metal_path}: {e}")
        return out

    out.n_ref = ref.cols
    out.n_metal = metal.cols
    out.ref_rows = ref.rows
    out.metal_rows = metal.rows

    if ref.rows != metal.rows:
        out.notes.append(
            f"row schema drift: ref has {ref.rows} fields, metal has {metal.rows} — "
            f"diffing leading canonical rows only")

    ref_idx = mat_reader.cell_rows_by_id(ref, id_row=CELL_ROW_ID)
    metal_idx = mat_reader.cell_rows_by_id(metal, id_row=CELL_ROW_ID)
    common_ids = sorted(set(ref_idx) & set(metal_idx))
    out.matched = len(common_ids)
    out.only_ref = len(set(ref_idx) - set(metal_idx))
    out.only_metal = len(set(metal_idx) - set(ref_idx))

    if out.matched == 0:
        out.ok = False
        out.notes.append("no shared cell IDs between ref and metal")
        return out

    def gather(row):
        rv = [ref.get(row, ref_idx[i]) for i in common_ids]
        mv = [metal.get(row, metal_idx[i]) for i in common_ids]
        return rv, mv

    tol_pos = tolerances.get("position", 1e-6)
    tol_vol = tolerances.get("volume", 1e-3)
    tol_phase = tolerances.get("phase", 0)

    for label, row, tol in [
        ("pos_x", CELL_ROW_X, tol_pos),
        ("pos_y", CELL_ROW_Y, tol_pos),
        ("pos_z", CELL_ROW_Z, tol_pos),
        ("volume", CELL_ROW_TOTAL_VOLUME, tol_vol),
    ]:
        rv, mv = gather(row)
        out.fields.append(_accumulate_field_errors(label, rv, mv, tol))

    # Phase index is integer — exact match unless tol > 0.
    if min(ref.rows, metal.rows) > CELL_ROW_CURRENT_PHASE:
        rv, mv = gather(CELL_ROW_CURRENT_PHASE)
        out.fields.append(_accumulate_field_errors("cycle_phase", rv, mv, tol_phase))

    # Cell-count parity is reported but doesn't fail unless requested.
    if out.n_ref != out.n_metal:
        tol_count = tolerances.get("cell_count_abs", None)
        msg = (f"cell count differs: ref={out.n_ref}, metal={out.n_metal} "
               f"(matched={out.matched}, only_ref={out.only_ref}, only_metal={out.only_metal})")
        out.notes.append(msg)
        if tol_count is not None and abs(out.n_ref - out.n_metal) > tol_count:
            out.ok = False

    if any(not f.ok for f in out.fields):
        out.ok = False
    return out


def diff_microenvironment(ref_path, metal_path, tolerances):
    """Compare two microenvironment0.mat files.

    `tolerances` keys:
        substrate_linf : default 1e-3   (max abs density error)
        substrate_rmse : default 1e-4   (rmse density error)
    """
    out = MicroDiff()
    try:
        ref = mat_reader.read_mat(ref_path)
    except Exception as e:
        out.ok = False
        out.notes.append(f"failed to read ref {ref_path}: {e}")
        return out
    try:
        metal = mat_reader.read_mat(metal_path)
    except Exception as e:
        out.ok = False
        out.notes.append(f"failed to read metal {metal_path}: {e}")
        return out

    out.n_voxels_ref = ref.cols
    out.n_voxels_metal = metal.cols
    out.ref_rows = ref.rows
    out.metal_rows = metal.rows

    if ref.cols != metal.cols:
        out.ok = False
        out.notes.append(
            f"voxel count mismatch: ref={ref.cols}, metal={metal.cols} — "
            f"substrate diff skipped")
        return out
    if ref.rows != metal.rows:
        out.notes.append(
            f"substrate count differs: ref rows={ref.rows}, metal rows={metal.rows}")

    # Check voxel coordinates align.
    coord_tol = tolerances.get("voxel_coord", 1e-4)
    misaligned = 0
    for c in range(ref.cols):
        for r in (ME_ROW_X, ME_ROW_Y, ME_ROW_Z):
            if abs(ref.get(r, c) - metal.get(r, c)) > coord_tol:
                misaligned += 1
                break
    if misaligned == 0:
        out.voxels_aligned = True
    else:
        out.ok = False
        out.notes.append(f"{misaligned}/{ref.cols} voxels have mismatched coordinates "
                         f"(tol={coord_tol})")
        return out

    n_subs = min(ref.rows, metal.rows) - ME_FIRST_SUBSTRATE_ROW
    if n_subs <= 0:
        out.notes.append("no substrate rows present")
        return out

    tol_linf = tolerances.get("substrate_linf", 1e-3)
    tol_rmse = tolerances.get("substrate_rmse", 1e-4)

    for s in range(n_subs):
        row = ME_FIRST_SUBSTRATE_ROW + s
        rv = ref.row(row)
        mv = metal.row(row)
        stat = _accumulate_field_errors(f"substrate_{s}", rv, mv, tol_linf)
        stat.ok = (stat.max_abs_err <= tol_linf) and (stat.rmse <= tol_rmse)
        # Capture RMSE in tolerance display by stashing both bounds
        stat.tolerance = tol_linf
        out.substrate_stats.append(stat)
        if not stat.ok:
            out.ok = False

    return out


# --- High-level snapshot pair diff ---

def diff_snapshot_pair(ref_dir, metal_dir, frame_index, tolerances):
    """Diff the cells.mat and microenvironment.mat for a frame index."""
    import os
    label = f"output{frame_index:08d}"
    ref_cells = os.path.join(ref_dir, f"{label}_cells.mat")
    metal_cells = os.path.join(metal_dir, f"{label}_cells.mat")
    ref_micro = os.path.join(ref_dir, f"{label}_microenvironment0.mat")
    metal_micro = os.path.join(metal_dir, f"{label}_microenvironment0.mat")

    snap = SnapshotDiff(label=label)
    if os.path.isfile(ref_cells) and os.path.isfile(metal_cells):
        snap.cells = diff_cells(ref_cells, metal_cells, tolerances)
    if os.path.isfile(ref_micro) and os.path.isfile(metal_micro):
        snap.micro = diff_microenvironment(ref_micro, metal_micro, tolerances)
    return snap


def format_snapshot(snap):
    """Pretty-print a SnapshotDiff for the report."""
    lines = []
    status = "PASS" if snap.ok else "FAIL"
    lines.append(f"[{status}] {snap.label}")
    if snap.cells is not None:
        c = snap.cells
        lines.append(f"  cells: ref={c.n_ref}  metal={c.n_metal}  "
                     f"matched={c.matched}  only_ref={c.only_ref}  only_metal={c.only_metal}  "
                     f"(rows ref={c.ref_rows}, metal={c.metal_rows})")
        for n in c.notes:
            lines.append(f"    note: {n}")
        for fs in c.fields:
            lines.append(fs.fmt())
    if snap.micro is not None:
        m = snap.micro
        lines.append(f"  microenv: voxels ref={m.n_voxels_ref}  metal={m.n_voxels_metal}  "
                     f"aligned={m.voxels_aligned}  "
                     f"(rows ref={m.ref_rows}, metal={m.metal_rows})")
        for n in m.notes:
            lines.append(f"    note: {n}")
        for fs in m.substrate_stats:
            lines.append(fs.fmt())
    return "\n".join(lines)
