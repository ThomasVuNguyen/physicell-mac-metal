"""Runs the reference PhysiCell binary and the Metal port for parity scenarios.

A scenario hands us:
  - a base config XML (PhysiCell_settings.xml)
  - optional XML overrides (max_time, save interval, omp_num_threads, random_seed)
  - the path to the reference PhysiCell binary

For each side we:
  1. Materialise a copy of the config with <save><folder> rewritten to an
     absolute path under the scenario's tmp dir.
  2. Apply overrides.
  3. Run the binary, capturing stdout/stderr.

Notes:
  * PhysiCell needs its working directory set to wherever its config/cells.csv
    references live (it uses relative paths). We cd to the directory containing
    the materialised config.
  * The Metal port needs cwd = its project root because it loads
    `build/shaders.metallib` relatively. We pass it the absolute config path.
  * Both binaries write outputs to whatever <folder> resolves to from cwd; we
    rewrite that to an absolute path so cwd doesn't matter for outputs.
"""

import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET


class RunResult:
    __slots__ = ("ok", "returncode", "elapsed", "log_path", "output_dir", "stderr_tail")

    def __init__(self, ok, returncode, elapsed, log_path, output_dir, stderr_tail=""):
        self.ok = ok
        self.returncode = returncode
        self.elapsed = elapsed
        self.log_path = log_path
        self.output_dir = output_dir
        self.stderr_tail = stderr_tail


def materialise_config(base_config_path, dst_config_path, output_dir, overrides):
    """Copy base config to dst, rewriting <save><folder> and applying overrides.

    `overrides` may contain:
        max_time: float (minutes)
        save_interval: float (minutes, for full_data + SVG)
        omp_num_threads: int
        random_seed: int
    """
    tree = ET.parse(base_config_path)
    root = tree.getroot()

    save = root.find("save")
    if save is None:
        save = ET.SubElement(root, "save")
    folder = save.find("folder")
    if folder is None:
        folder = ET.SubElement(save, "folder")
    folder.text = output_dir

    if "max_time" in overrides:
        mt = root.find("overall/max_time")
        if mt is not None:
            mt.text = str(overrides["max_time"])
    if "save_interval" in overrides:
        siv = overrides["save_interval"]
        fd = root.find("save/full_data/interval")
        if fd is not None:
            fd.text = str(siv)
        svg = root.find("save/SVG/interval")
        if svg is not None:
            svg.text = str(siv)
    if "omp_num_threads" in overrides:
        omp = root.find("parallel/omp_num_threads")
        if omp is not None:
            omp.text = str(overrides["omp_num_threads"])
    if "random_seed" in overrides:
        rs = root.find("options/random_seed")
        if rs is None:
            opts = root.find("options")
            if opts is None:
                opts = ET.SubElement(root, "options")
            rs = ET.SubElement(opts, "random_seed")
        rs.text = str(overrides["random_seed"])

    os.makedirs(os.path.dirname(dst_config_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    tree.write(dst_config_path, encoding="UTF-8", xml_declaration=True)


def _run(cmd, cwd, log_path, timeout):
    """Run a subprocess, tee stdout+stderr to log_path. Returns (rc, elapsed, stderr_tail)."""
    t0 = time.time()
    with open(log_path, "wb") as lf:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    elapsed = time.time() - t0
    # Grab tail for error reports
    tail = ""
    try:
        with open(log_path, "rb") as lf:
            data = lf.read()
            if len(data) > 4000:
                data = data[-4000:]
            tail = data.decode("utf-8", errors="replace")
    except Exception:
        pass
    return rc, elapsed, tail


def run_reference(ref_binary, base_config, output_dir, log_path, overrides,
                  config_dir=None, timeout=900):
    """Run the PhysiCell reference binary.

    `config_dir`: cwd for the run. PhysiCell uses relative paths for cells.csv etc.,
                  so this should be the dir housing the original config (i.e. the
                  PhysiCell project root). If None we use dirname(base_config) parent.
    """
    if not os.path.isfile(ref_binary):
        raise FileNotFoundError(f"reference binary missing: {ref_binary}")

    cwd = config_dir if config_dir else os.path.dirname(os.path.dirname(base_config))
    materialised = os.path.join(cwd, "_parity_settings.xml")
    materialise_config(base_config, materialised, output_dir, overrides)

    rc, elapsed, tail = _run(
        [os.path.abspath(ref_binary), materialised],
        cwd=cwd, log_path=log_path, timeout=timeout)
    return RunResult(rc == 0, rc, elapsed, log_path, output_dir, tail)


def run_metal(metal_binary, metal_project_root, base_config, output_dir, log_path,
              overrides, timeout=900):
    """Run the Metal port. cwd must be metal_project_root so shaders load."""
    if not os.path.isfile(metal_binary):
        raise FileNotFoundError(f"metal binary missing: {metal_binary}")

    materialised = os.path.join(output_dir, "_settings.xml")
    materialise_config(base_config, materialised, output_dir, overrides)

    rc, elapsed, tail = _run(
        [os.path.abspath(metal_binary), os.path.abspath(materialised)],
        cwd=metal_project_root, log_path=log_path, timeout=timeout)
    return RunResult(rc == 0, rc, elapsed, log_path, output_dir, tail)


def discover_snapshots(output_dir):
    """Return sorted list of frame indices present (by output*_cells.mat existence)."""
    if not os.path.isdir(output_dir):
        return []
    pat = re.compile(r"output(\d{8})_cells\.mat$")
    out = []
    for f in os.listdir(output_dir):
        m = pat.match(f)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out
