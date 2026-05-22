"""MATLAB v4 (.mat Level-4) reader for PhysiCell / BioFVM output.

Both PhysiCell and the Metal port write Level-4 .mat files via BioFVM_matlab:
column-major, little-endian, one matrix per file.

This reader is stdlib-only (no numpy). It returns a Matrix object exposing:
  - name:   matrix name (string)
  - rows:   number of rows
  - cols:   number of columns
  - data:   flat list of floats, length rows*cols, COLUMN-MAJOR
            (i.e. data[col*rows + row] == M[row, col])

For PhysiCell cells.mat: each column is one cell, each row is one field.
For PhysiCell microenvironment0.mat: each column is one voxel, rows are
[x, y, z, voxel_volume, substrate_0, substrate_1, ...].
"""

import os
import struct


class Matrix:
    __slots__ = ("name", "rows", "cols", "data")

    def __init__(self, name, rows, cols, data):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.data = data

    def get(self, row, col):
        return self.data[col * self.rows + row]

    def row(self, row):
        return [self.data[col * self.rows + row] for col in range(self.cols)]

    def col(self, col):
        base = col * self.rows
        return self.data[base:base + self.rows]


def read_mat(path):
    """Read a single-matrix Level-4 .mat file. Returns Matrix or raises."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        header = f.read(20)
        if len(header) != 20:
            raise ValueError(f"{path}: header truncated")

        typ, rows, cols, imagf, name_len = struct.unpack("<5I", header)

        # Decompose type: MOPT
        M = (typ // 1000) % 10   # 0=LE, 1=BE
        P = (typ // 10) % 10     # 0=double, 1=float, 2=int32, 3=int16, 4=uint16, 5=uint8
        T = typ % 10             # 0=numeric

        if M != 0:
            raise ValueError(f"{path}: only little-endian (M=0) supported, got M={M}")
        if T != 0:
            raise ValueError(f"{path}: only numeric matrices (T=0) supported, got T={T}")
        if imagf != 0:
            raise ValueError(f"{path}: complex matrices not supported")

        name_bytes = f.read(name_len)
        name = name_bytes.rstrip(b"\x00").decode("latin1", errors="replace")

        n = rows * cols
        if P == 0:
            buf = f.read(8 * n)
            if len(buf) != 8 * n:
                raise ValueError(f"{path}: data truncated (expected {8*n} bytes, got {len(buf)})")
            data = list(struct.unpack(f"<{n}d", buf))
        elif P == 1:
            buf = f.read(4 * n)
            if len(buf) != 4 * n:
                raise ValueError(f"{path}: data truncated")
            data = list(struct.unpack(f"<{n}f", buf))
        else:
            raise ValueError(f"{path}: unsupported precision P={P}")

    return Matrix(name, rows, cols, data)


def cell_rows_by_id(mat, id_row=0):
    """Build {cell_id -> column_index} from a cells.mat. PhysiCell row 0 = ID."""
    out = {}
    for c in range(mat.cols):
        cid = int(mat.get(id_row, c))
        out[cid] = c
    return out


if __name__ == "__main__":
    # CLI smoke test
    import sys
    if len(sys.argv) < 2:
        print("usage: mat_reader.py FILE.mat")
        sys.exit(2)
    m = read_mat(sys.argv[1])
    print(f"name={m.name!r}  shape=({m.rows}, {m.cols})  size={len(m.data)}")
    print(f"first column: {m.col(0)[:10]}...")
