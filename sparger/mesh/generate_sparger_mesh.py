#!/usr/bin/env python3
"""Extrude the discharge section into the complete 3-D sparger fluid mesh."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np

from nekmeshpy import ElementTags, hexmesh, quadmesh, writer
from nekmeshpy.hexmesh import Seam

from generate_discharge_section import (
    INNER_SHELL_RADIUS,
    OUTER_SHELL_RADIUS,
    TUBE_CENTERS,
    TUBE_RADIUS,
    Z_SECTION,
    build_section,
)


# Axial dimensions, nondimensionalized by the 1.5875 mm tube diameter.
Z_BOTTOM = -0.0932434 / 0.0015875
Z_TOP = -0.0680034 / 0.0015875

# Approximately 0.2 tube diameters per axial macro element.  Every block uses
# the same discharge-plane section, so all three upward blocks remain conformal
# to the complete cylinder extruded downward.
TARGET_AXIAL_SPACING = 0.2
N_LOWER_LAYERS = math.ceil((Z_SECTION - Z_BOTTOM) / TARGET_AXIAL_SPACING)
N_UPPER_LAYERS = math.ceil((Z_TOP - Z_SECTION) / TARGET_AXIAL_SPACING)

HERE = Path(__file__).resolve().parent
RE2_OUTPUT = HERE / "sparger.re2"
VTU_OUTPUT = HERE / "sparger.vtu"

GROUPS = {
    "vessel_bottom": "W  ",
    "outer_shell": "W  ",
    "vessel_top": "O  ",
    "boring_bottom": "W  ",
    "inner_shell": "W  ",
    "tube_inlet": "v  ",
    "tube_wall": "W  ",
}

# These are the exact IDs consumed by ss1.usr and ss1.par. Dictionary order
# also makes the VTU bc_id field use this same 1-based numbering.
BOUNDARY_IDS = {
    "vessel_bottom": 1,
    "outer_shell": 2,
    "vessel_top": 3,
    "boring_bottom": 4,
    "inner_shell": 5,
    "tube_inlet": 6,
    "tube_wall": 7,
}


def _write_re2_with_boundary_ids(mesh: hexmesh.HexMesh, path: Path) -> None:
    """Write `.re2`, then populate Nek's numeric ``bc(5)`` slot.

    NekMeshPy writes the three-character boundary condition code but leaves
    ``bc(1:5)`` zero. The existing ss1.usr reads ``bc(5)`` as the surface ID,
    so place the stable 1--7 IDs there without changing the standard codes.
    """
    writer.to_re2(mesh, str(path), groups=GROUPS)

    physical_groups = writer._as_groups(mesh, GROUPS)
    rows = writer._export_rows(mesh, physical_groups)
    bytes_per_element = 25 * 8
    bytes_per_boundary = 8 * 8
    first_boundary = 80 + 4 + mesh.n_hexes * bytes_per_element + 8 + 8

    with path.open("r+b") as stream:
        for row_index, (_element, _face, name, _code, _pe, _pf) in enumerate(rows):
            boundary_id = BOUNDARY_IDS[name]
            # Boundary record layout: element, face, bc(1:5), cbc. Thus
            # bc(5) is double number 7, i.e. zero-based slot 6.
            offset = first_boundary + row_index * bytes_per_boundary + 6 * 8
            stream.seek(offset)
            stream.write(struct.pack("<d", float(boundary_id)))


def _tag_exposed_radius(
    section: quadmesh.QuadMesh,
    center: tuple[float, float],
    radius: float,
    tag: str,
) -> quadmesh.QuadMesh:
    """Tag currently exposed section edges lying on a specified circle."""
    rows = quadmesh.boundary_edges(section)
    corners = section.corners
    edge_corners = section.EDGE_POINTS[rows[:, 1] - 1]
    point_ids = corners[rows[:, 0, None], edge_corners]
    points = section.points[point_ids]
    radial = np.linalg.norm(points[:, :, :2] - np.asarray(center)[None, None, :], axis=2)
    selected = np.all(np.abs(radial - radius) < 1.0e-9, axis=1)
    if not np.any(selected):
        raise RuntimeError(f"no exposed edges found for boundary {tag!r}")
    return quadmesh.tag_edges(section, rows[selected], tag)


def _cap_tags(region_names: np.ndarray) -> ElementTags:
    """Name the discharge cap according to the block that attaches above it."""
    names = np.empty(region_names.shape, dtype=object)
    names[region_names == "surrounding"] = "boring_bottom"
    names[region_names == "outer_annulus"] = "join_outer"
    names[region_names == "tube_1"] = "join_tube_1"
    names[region_names == "tube_2"] = "join_tube_2"
    return ElementTags.from_dense(np.asarray(names, dtype=np.str_))


def build_mesh() -> hexmesh.HexMesh:
    section = build_section()
    regions = section.element_tags.dense(section.n_quads)

    # Complete cylinder below the discharge plane.
    lower = hexmesh.extrude(
        section,
        length=Z_SECTION - Z_BOTTOM,
        layers=N_LOWER_LAYERS,
        axis=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, Z_BOTTOM - Z_SECTION),
        element_tags=section.element_tags,
        first_tag="vessel_bottom",
        last_tag=_cap_tags(regions),
    )

    # Outer annulus above the discharge plane.
    outer_section = quadmesh.select(section, regions == "outer_annulus")
    outer_section = _tag_exposed_radius(
        outer_section, (0.0, 0.0), INNER_SHELL_RADIUS, "inner_shell"
    )
    upper_outer = hexmesh.extrude(
        outer_section,
        length=Z_TOP - Z_SECTION,
        layers=N_UPPER_LAYERS,
        axis=(0.0, 0.0, 1.0),
        element_tags=outer_section.element_tags,
        first_tag="join_outer",
        last_tag="vessel_top",
    )

    # Two circular inlet tubes above the discharge plane.
    tube_blocks = []
    for index, center in enumerate(TUBE_CENTERS, start=1):
        tube_section = quadmesh.select(section, regions == f"tube_{index}")
        tube_section = _tag_exposed_radius(
            tube_section, center, TUBE_RADIUS, "tube_wall"
        )
        tube_blocks.append(
            hexmesh.extrude(
                tube_section,
                length=Z_TOP - Z_SECTION,
                layers=N_UPPER_LAYERS,
                axis=(0.0, 0.0, 1.0),
                element_tags=tube_section.element_tags,
                first_tag=f"join_tube_{index}",
                last_tag="tube_inlet",
            )
        )

    blocks = [lower, upper_outer, *tube_blocks]
    mesh = hexmesh.attach(
        blocks,
        [
            Seam(0, "join_outer", 1, "join_outer"),
            Seam(0, "join_tube_1", 2, "join_tube_1"),
            Seam(0, "join_tube_2", 3, "join_tube_2"),
        ],
    )
    return mesh


def main() -> None:
    mesh = build_mesh()
    quality = hexmesh.quality_summary(mesh, order=8)

    _write_re2_with_boundary_ids(mesh, RE2_OUTPUT)
    writer.to_vtu(mesh, str(VTU_OUTPUT), groups=GROUPS)

    print(f"wrote: {RE2_OUTPUT}")
    print(f"wrote: {VTU_OUTPUT}")
    print(f"z range: {Z_BOTTOM:.12g} to {Z_TOP:.12g}")
    print(f"lower axial layers: {N_LOWER_LAYERS}")
    print(f"upper axial layers: {N_UPPER_LAYERS}")
    print(f"points: {mesh.n_points}")
    print(f"hexahedra: {mesh.n_hexes}")
    print(f"boundary groups: {sorted(mesh.face_group_tags)}")
    print(f"quality at solver order 8: {quality}")


if __name__ == "__main__":
    main()
