#!/usr/bin/env python3
"""Generate the sparger mesh with a smooth converging annular outlet snout.

The mesh is identical to ``generate_sparger_mesh.py`` below the adjustable
snout length. Over that final axial section, the full outer annulus contracts
smoothly to a narrow annular slot centered midway between the inner and outer
shell radii. The slot area equals the combined area of both inlet tubes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np

from nekmeshpy import hexmesh, quadmesh, writer
from nekmeshpy.hexmesh import Seam

import generate_sparger_mesh as base
from generate_discharge_section import (
    INNER_SHELL_RADIUS,
    OUTER_SHELL_RADIUS,
    TUBE_CENTERS,
    TUBE_RADIUS,
    Z_SECTION,
    build_section,
)


# Axial length of the smooth converging outlet section, in tube diameters.
# The snout begins this distance below Z_VESSEL_TOP.
SNOUT_LENGTH_DIAMETERS = 5.0
NUMBER_OF_INLET_TUBES = len(TUBE_CENTERS)
SNOUT_MID_RADIUS = 0.5 * (INNER_SHELL_RADIUS + OUTER_SHELL_RADIUS)
TOTAL_INLET_AREA = NUMBER_OF_INLET_TUBES * math.pi * TUBE_RADIUS**2
# Final annular outlet area relative to the combined area of both inlet tubes.
OUTLET_TO_INLET_AREA_RATIO = 10.0
SNOUT_OUTLET_AREA = OUTLET_TO_INLET_AREA_RATIO * TOTAL_INLET_AREA
# For radii Rmid +/- width/2, annular area is exactly 2*pi*Rmid*width.
SNOUT_WIDTH = SNOUT_OUTLET_AREA / (2.0 * math.pi * SNOUT_MID_RADIUS)
SNOUT_INNER_RADIUS = SNOUT_MID_RADIUS - 0.5 * SNOUT_WIDTH
SNOUT_OUTER_RADIUS = SNOUT_MID_RADIUS + 0.5 * SNOUT_WIDTH
SNOUT_START_Z = base.Z_VESSEL_TOP - SNOUT_LENGTH_DIAMETERS

HERE = Path(__file__).resolve().parent
RE2_OUTPUT = HERE / "sparger_snout.re2"
VTU_OUTPUT = HERE / "sparger_snout.vtu"


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _snout_top_section(section: quadmesh.QuadMesh) -> quadmesh.QuadMesh:
    """Contract an annular section to the final concentric-ring outlet."""
    radial_span = OUTER_SHELL_RADIUS - INNER_SHELL_RADIUS

    def contract(points: np.ndarray) -> np.ndarray:
        mapped = points.copy()
        radius = np.linalg.norm(mapped[:, :2], axis=1)
        fraction = (radius - INNER_SHELL_RADIUS) / radial_span
        target_radius = (
            SNOUT_INNER_RADIUS + fraction * SNOUT_WIDTH
        )
        mapped[:, 0] *= target_radius / radius
        mapped[:, 1] *= target_radius / radius
        return mapped

    return quadmesh.transform_fn(section, contract)


def _snout_profiles(
    outer_section: quadmesh.QuadMesh,
) -> tuple[list[quadmesh.QuadMesh], np.ndarray]:
    """Return conformal profiles spanning the straight annulus and snout."""
    length = base.Z_VESSEL_TOP - Z_SECTION
    fractions = base.UPPER_OUTER_LAYER_POSITIONS.copy()
    snout_start_fraction = (SNOUT_START_Z - Z_SECTION) / length
    # Do not insert another profile next to an already nearby graded station:
    # that creates a very thin axial element at the start of the snout. Move
    # the nearest existing station to the exact start instead. Its displacement
    # is at most half one local layer, so the neighboring sizes remain smooth.
    nearest = int(np.argmin(np.abs(fractions - snout_start_fraction)))
    fractions[nearest] = snout_start_fraction
    fractions = np.sort(fractions)
    z_positions = Z_SECTION + fractions * length

    eta = (z_positions - SNOUT_START_Z) / SNOUT_LENGTH_DIAMETERS
    contraction = _smoothstep(eta)
    top_section = _snout_top_section(outer_section)
    profiles = quadmesh.blend(outer_section, top_section, contraction)
    profiles = [
        quadmesh.translate(profile, (0.0, 0.0, z - Z_SECTION))
        for profile, z in zip(profiles, z_positions)
    ]
    return profiles, z_positions


def build_mesh() -> hexmesh.HexMesh:
    section = build_section()
    regions = section.element_tags.dense(section.n_quads)

    lower = hexmesh.extrude(
        section,
        length=Z_SECTION - base.Z_BOTTOM,
        layers=base.N_LOWER_LAYERS,
        axis=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, base.Z_BOTTOM - Z_SECTION),
        element_tags=section.element_tags,
        first_tag="vessel_bottom",
        last_tag=base._cap_tags(regions),
    )

    outer_section = quadmesh.select(section, regions == "outer_annulus")
    outer_section = base._tag_exposed_radius(
        outer_section,
        (0.0, 0.0),
        INNER_SHELL_RADIUS,
        "inner_shell",
    )
    profiles, _ = _snout_profiles(outer_section)
    upper_outer = hexmesh.loft(
        profiles,
        element_tags=outer_section.element_tags,
        first_tag="join_outer",
        last_tag="vessel_top",
    )

    tube_blocks = []
    for index, center in enumerate(TUBE_CENTERS, start=1):
        tube_section = quadmesh.select(section, regions == f"tube_{index}")
        tube_section = base._tag_exposed_radius(
            tube_section, center, TUBE_RADIUS, "tube_wall"
        )
        tube_blocks.append(
            hexmesh.extrude(
                tube_section,
                length=base.Z_TUBE_INLET - Z_SECTION,
                layers=base.N_TUBE_LAYERS,
                axis=(0.0, 0.0, 1.0),
                element_tags=tube_section.element_tags,
                first_tag=f"join_tube_{index}",
                last_tag="tube_inlet",
            )
        )

    return hexmesh.attach(
        [lower, upper_outer, *tube_blocks],
        [
            Seam(0, "join_outer", 1, "join_outer"),
            Seam(0, "join_tube_1", 2, "join_tube_1"),
            Seam(0, "join_tube_2", 3, "join_tube_2"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-order",
        type=int,
        default=1,
        metavar="N",
        help="quality sampling order; use 8 for a full solver-order audit",
    )
    parser.add_argument(
        "--skip-vtu",
        action="store_true",
        help="write only sparger_snout.re2",
    )
    args = parser.parse_args()
    if args.quality_order < 0:
        parser.error("--quality-order must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    mesh = build_mesh()
    built = time.perf_counter()

    quality = None
    if args.quality_order:
        quality = hexmesh.quality_summary(mesh, order=args.quality_order)
    checked = time.perf_counter()

    base._write_re2_with_boundary_ids(mesh, RE2_OUTPUT)
    if not args.skip_vtu:
        writer.to_vtu(mesh, str(VTU_OUTPUT), groups=base.GROUPS)
    written = time.perf_counter()

    print(f"wrote: {RE2_OUTPUT}")
    if not args.skip_vtu:
        print(f"wrote: {VTU_OUTPUT}")
    print(f"snout z range: {SNOUT_START_Z:.12g} to {base.Z_VESSEL_TOP:.12g}")
    print(f"snout axial length [D]: {SNOUT_LENGTH_DIAMETERS:.12g}")
    print(f"snout midpoint radius: {SNOUT_MID_RADIUS:.12g}")
    print(f"snout inner/outer radii: {SNOUT_INNER_RADIUS:.12g} / {SNOUT_OUTER_RADIUS:.12g}")
    print(f"snout width: {SNOUT_WIDTH:.12g}")
    print(f"outlet/inlet area ratio: {OUTLET_TO_INLET_AREA_RATIO:.12g}")
    print(f"snout area: {math.pi * (SNOUT_OUTER_RADIUS**2 - SNOUT_INNER_RADIUS**2):.12g}")
    print(f"combined tube inlet area: {TOTAL_INLET_AREA:.12g}")
    print(f"points: {mesh.n_points}")
    print(f"hexahedra: {mesh.n_hexes}")
    if quality is None:
        print("quality check: skipped")
    else:
        print(f"quality at sampling order {args.quality_order}: {quality}")
    print(
        "timing [s] build/quality/write/total: "
        f"{built - start:.2f} / {checked - built:.2f} / "
        f"{written - checked:.2f} / {written - start:.2f}"
    )


if __name__ == "__main__":
    main()
