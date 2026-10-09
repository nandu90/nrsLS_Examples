#!/usr/bin/env python3
"""Generate the conforming all-quad sparger section at the tube discharge.

The geometry is nondimensionalized by the tube diameter.  The section covers
the complete disk inside the inner-shell radius and contains two circular tube
regions embedded in the surrounding region.  It is intended to become the
common interface between the lower chamber and the separately extruded tubes.

Install NekMeshPy before running this script:

    python3 -m pip install "NekMeshPy[plot] @ git+https://github.com/khanhn201/NekMeshPy.git"
    python3 generate_discharge_section.py

The script writes ``sparger_discharge_section.vtu`` for inspection in ParaView
and both PNG and SVG previews in this script's directory.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from nekmeshpy import ElementTags, linemesh, quadmesh, writer
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.quadmesh import QuadMesh


# Geometry: tube diameter is the reference length.
INNER_SHELL_RADIUS = 2.0
OUTER_SHELL_RADIUS = 0.0102112534 / 0.0015875
TUBE_RADIUS = 0.5
TUBE_CENTERS = ((-1.0, 0.0), (1.0, 0.0))
Z_SECTION = -52.4367874015748

# Mesh controls. N_CIRC must equal 4*N_SIDE, and N_SIDE must be even. Keep the
# original tube O-grid resolution; refine only the surrounding radial region.
N_SIDE = 12
N_CIRC = 4 * N_SIDE
N_TUBE_RADIAL = 5
N_SURROUNDING_RADIAL = 8
# Blend uniform and cosine spacing. This makes radial spacing grow smoothly
# away from each tube and shrink again near the outer interface, avoiding one
# oversized terminal element. Zero is uniform; one is full cosine clustering.
SURROUNDING_END_CLUSTERING = 0.25
# Redistribute only the surrounding rays that terminate on the x=0 divider.
# A value of one makes their meeting locations uniformly spaced from -R to R;
# zero retains the original tan(angle) distribution. Tube points are unchanged.
CENTER_DIVIDER_SPREAD = 0.75
N_OUTER_SHELL_RADIAL = 14
FIRST_OUTER_SHELL_LAYER = 0.15
ORDER = 1

OUTPUT = Path(__file__).with_name("sparger_discharge_section.vtu")
PREVIEW = Path(__file__).with_name("sparger_discharge_section.svg")
PNG_PREVIEW = Path(__file__).with_name("sparger_discharge_section.png")


def _tagged_loop(points: np.ndarray, tags: list[str]) -> LineMesh:
    """Closed linear loop with one tag per segment."""
    loop = linemesh.loft(points, loop=True, order=ORDER)
    return LineMesh(
        loop.point_mesh,
        loop.lines,
        loop.interior,
        ElementTags.from_dense(np.asarray(tags, dtype=np.str_)),
    )


def _with_region(mesh: QuadMesh, name: str) -> QuadMesh:
    """Return ``mesh`` with every quadrilateral assigned to one region."""
    return QuadMesh(
        mesh.line_mesh,
        mesh.quads,
        mesh.orient,
        mesh.interior,
        ElementTags.uniform(mesh.n_quads, name),
    )


def _geometric_ratio(total: float, first: float, layers: int) -> float:
    """Growth ratio whose ``layers`` geometric widths sum to ``total``."""
    uniform_total = first * layers
    if total <= uniform_total * (1.0 + 1.0e-12):
        return 1.0

    def series(ratio: float) -> float:
        return first * (ratio**layers - 1.0) / (ratio - 1.0)

    lower, upper = 1.0, 2.0
    while series(upper) < total:
        upper *= 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if series(middle) < total:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _layered_surrounding(
    tube_points: np.ndarray,
    outer_points: np.ndarray,
    outer_loop: LineMesh,
) -> QuadMesh:
    """Mesh the tube-to-half-disk region with smooth two-sided grading."""
    delta = outer_points - tube_points

    uniform = np.linspace(0.0, 1.0, N_SURROUNDING_RADIAL + 1)
    cosine = 0.5 * (1.0 - np.cos(math.pi * uniform))
    fractions = (
        (1.0 - SURROUNDING_END_CLUSTERING) * uniform
        + SURROUNDING_END_CLUSTERING * cosine
    )

    rings: list[LineMesh] = []
    for fraction in fractions:
        points = tube_points + fraction * delta
        # Radial lofting winds the annular quads opposite to an O-grid built
        # from a CCW boundary, so reverse each ring to keep the final section
        # consistently +z-oriented for later hexahedral extrusion.
        rings.append(linemesh.loft(points[::-1], loop=True, order=ORDER))

    outer_tags = outer_loop.element_tags.dense(outer_loop.n_lines)
    reversed_outer_tags = ElementTags.from_dense(
        np.roll(outer_tags[::-1], -1)
    )
    return quadmesh.loft(
        rings,
        element_tags="surrounding",
        last_tag=reversed_outer_tags,
    )


def _ordered_tag_points(section: QuadMesh, tag: str) -> np.ndarray:
    """Return the corner points of one closed tagged boundary in CCW order."""
    edge_ids = quadmesh.tagged_edges(section, tag)
    edges = np.asarray(section.line_mesh.lines[edge_ids], dtype=np.int64)
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError(f"boundary tag {tag!r} is not one closed loop")

    start = min(adjacency)
    ordered = [start]
    previous = -1
    current = start
    while True:
        neighbors = adjacency[current]
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == start:
            break
        ordered.append(following)
        previous, current = current, following
        if len(ordered) > len(adjacency):
            raise RuntimeError(f"failed to order boundary tag {tag!r}")

    points = section.points[np.asarray(ordered, dtype=np.int64)]
    signed_area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    )
    return points if signed_area > 0.0 else points[::-1].copy()


def _outer_shell_annulus(section: QuadMesh) -> QuadMesh:
    """Add the annulus from the inner-shell circle to the outer-shell wall."""
    inner_points = _ordered_tag_points(section, "inner_shell")
    outer_points = inner_points.copy()
    outer_points[:, :2] *= OUTER_SHELL_RADIUS / INNER_SHELL_RADIUS

    inner_points = inner_points[::-1].copy()
    outer_points = outer_points[::-1].copy()
    inner_loop = _tagged_loop(inner_points, [""] * inner_points.shape[0])
    outer_loop = _tagged_loop(
        outer_points, ["outer_shell"] * outer_points.shape[0]
    )

    ratio = _geometric_ratio(
        OUTER_SHELL_RADIUS - INNER_SHELL_RADIUS,
        FIRST_OUTER_SHELL_LAYER,
        N_OUTER_SHELL_RADIAL,
    )
    widths = FIRST_OUTER_SHELL_LAYER * ratio ** np.arange(N_OUTER_SHELL_RADIAL)
    fractions = np.concatenate(([0.0], np.cumsum(widths)))
    fractions /= fractions[-1]

    shell = quadmesh.annulus(
        inner_loop,
        outer_loop,
        radial=fractions,
    )
    return _with_region(shell, "outer_annulus")


def _distance_to_half_disk_boundary(
    center_x: float, dx: np.ndarray, dy: np.ndarray
) -> np.ndarray:
    """Ray distance from a tube center to its containing half disk.

    Each ray first meets either the circular inner-shell boundary or the
    center divider x=0.  Taking the nearer positive intersection produces a
    closed, star-shaped outer loop that pairs point-for-point with the tube
    circle, so the region between them can be filled with structured quads.
    """
    # Positive intersection with x^2+y^2=R^2 for p=(center_x,0)+r*(dx,dy).
    cdot = center_x * dx
    circle = -cdot + np.sqrt(cdot * cdot + INNER_SHELL_RADIUS**2 - center_x**2)

    divider = np.full_like(circle, np.inf)
    toward_divider = dx * center_x < 0.0
    divider[toward_divider] = -center_x / dx[toward_divider]
    return np.minimum(circle, divider)


def _spread_center_divider_points(points: np.ndarray) -> np.ndarray:
    """Redistribute the x=0 divider endpoints toward uniform spacing.

    Only the outer endpoints of rays that connect the two tube half-sections
    are moved. The tube circumference is untouched; `_layered_surrounding`
    interpolates from that fixed circumference to these endpoints, so the
    lateral displacement develops gradually along each radial mesh line.
    """
    spread_points = points.copy()
    divider_ids = np.flatnonzero(np.abs(spread_points[:, 0]) < 1.0e-12)
    order = np.argsort(spread_points[divider_ids, 1])
    sorted_ids = divider_ids[order]
    original_y = spread_points[sorted_ids, 1]
    target_y = np.linspace(
        -INNER_SHELL_RADIUS,
        INNER_SHELL_RADIUS,
        sorted_ids.size,
    )
    spread_points[sorted_ids, 1] = (
        (1.0 - CENTER_DIVIDER_SPREAD) * original_y
        + CENTER_DIVIDER_SPREAD * target_y
    )
    return spread_points


def _tube_angles(center_x: float) -> np.ndarray:
    """CCW tube angles with exact half-disk corner rays.

    Relative angle zero points from either tube center toward the divider.  The
    ray reaches the two half-disk corners at +/-atan2(Rinner, |xcenter|); these
    angles must be explicit nodes or the two half meshes leave wedge-shaped
    gaps where their polygonal boundaries meet.
    """
    alpha = math.atan2(INNER_SHELL_RADIUS, abs(center_x))
    n_to_corner = int(round(alpha / (0.5 * math.pi) * N_SIDE))
    if not 1 <= n_to_corner < N_SIDE:
        raise ValueError("angular resolution cannot represent half-disk corners")

    q0 = np.concatenate(
        (
            np.linspace(0.0, alpha, n_to_corner + 1),
            np.linspace(alpha, 0.5 * math.pi, N_SIDE - n_to_corner + 1)[1:],
        )
    )
    q1 = np.linspace(0.5 * math.pi, math.pi, N_SIDE + 1)
    q2 = np.linspace(math.pi, 1.5 * math.pi, N_SIDE + 1)
    q3 = 2.0 * math.pi - q0[::-1]
    relative = np.concatenate((q0[:-1], q1[:-1], q2[:-1], q3[:-1]))

    # The right tube points toward the divider along global angle pi.
    return relative if center_x < 0.0 else relative + math.pi


def _half_section(center_x: float, tube_region: str) -> tuple[QuadMesh, QuadMesh]:
    """Build one tube O-grid and its surrounding half-disk ring."""
    theta = _tube_angles(center_x)
    dx = np.cos(theta)
    dy = np.sin(theta)

    tube_points = np.column_stack(
        (
            center_x + TUBE_RADIUS * dx,
            TUBE_RADIUS * dy,
            np.full(N_CIRC, Z_SECTION),
        )
    )
    tube_loop = _tagged_loop(tube_points, [""] * N_CIRC)

    distance = _distance_to_half_disk_boundary(center_x, dx, dy)
    outer_points = np.column_stack(
        (
            center_x + distance * dx,
            distance * dy,
            np.full(N_CIRC, Z_SECTION),
        )
    )
    outer_points = _spread_center_divider_points(outer_points)

    # The center-divider edges are internal and deliberately untagged.  Every
    # remaining edge lies on the inner-shell circle.
    next_points = np.roll(outer_points, -1, axis=0)
    on_divider = (
        np.abs(outer_points[:, 0]) < 1.0e-12
    ) & (np.abs(next_points[:, 0]) < 1.0e-12)
    outer_tags = np.where(on_divider, "", "inner_shell").tolist()
    outer_loop = _tagged_loop(outer_points, outer_tags)

    tube_radial = np.linspace(0.0, 1.0, N_TUBE_RADIAL + 1)
    tube = quadmesh.ogrid(
        tube_loop,
        n_side=N_SIDE,
        radial=tube_radial,
        center_scale=0.60,
        quadrant_scale=0.60,
    )
    tube = _with_region(tube, tube_region)

    surrounding = _layered_surrounding(tube_points, outer_points, outer_loop)
    return tube, surrounding


def build_section() -> QuadMesh:
    """Assemble and validate the complete two-tube discharge section."""
    left_tube, left_surrounding = _half_section(-1.0, "tube_1")
    right_tube, right_surrounding = _half_section(1.0, "tube_2")

    # Coordinate welding joins each tube to its surrounding ring and joins the
    # two half disks along x=0.  All welded interfaces are untagged, so only the
    # physical inner-shell boundary remains named.
    inner_section = quadmesh.merge(
        [left_tube, left_surrounding, right_tube, right_surrounding]
    )

    outer_annulus = _outer_shell_annulus(inner_section)
    inner_section = quadmesh.retag_edge(inner_section, {"inner_shell": ""})
    section = quadmesh.merge([inner_section, outer_annulus])

    quality = quadmesh.quality_summary(section, order=8)
    if quality.n_inverted:
        raise RuntimeError(
            f"generated section contains {quality.n_inverted} inverted elements"
        )
    return section


def _edge_aspect_ratios(section: QuadMesh) -> np.ndarray:
    """Corner-edge max/min ratio for every quadrilateral."""
    corners = section.points[section.corners]
    lengths = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    return np.max(lengths, axis=1) / np.min(lengths, axis=1)


def _write_svg_preview(section: QuadMesh, path: Path) -> None:
    """Write a lightweight region-colored mesh preview without matplotlib."""
    width = 900
    margin = 35
    scale = (width - 2 * margin) / (2.0 * OUTER_SHELL_RADIUS)
    colors = {
        "surrounding": "#dbeafe",
        "outer_annulus": "#bfdbfe",
        "tube_1": "#fdba74",
        "tube_2": "#fca5a5",
    }
    region = section.element_tags.dense(section.n_quads)

    def xy(point: np.ndarray) -> tuple[float, float]:
        return (
            margin + (point[0] + OUTER_SHELL_RADIUS) * scale,
            margin + (OUTER_SHELL_RADIUS - point[1]) * scale,
        )

    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{width}" viewBox="0 0 {width} {width}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for quad, name in zip(section.corners, region):
        coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in map(xy, section.points[quad]))
        rows.append(
            f'<polygon points="{coords}" fill="{colors[str(name)]}" '
            'stroke="#334155" stroke-width="0.55"/>'
        )
    rows.extend(
        [
            '<text x="35" y="25" font-family="sans-serif" font-size="17">'
            'Sparger section: Dtube = 1, Rinner = 2, Router = 6.4323</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_png_preview(section: QuadMesh, path: Path) -> None:
    """Write a region-colored PNG preview using NekMeshPy's plot dependency."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires matplotlib. Install it with "
            "`python -m pip install -e \"$HOME/NekMeshPy[plot]\"`."
        ) from exc

    colors = {
        "surrounding": "#dbeafe",
        "outer_annulus": "#bfdbfe",
        "tube_1": "#fdba74",
        "tube_2": "#fca5a5",
    }
    region = section.element_tags.dense(section.n_quads)
    polygons = section.points[section.corners][:, :, :2]

    figure, axes = plt.subplots(figsize=(8.0, 8.0), dpi=150)
    collection = PolyCollection(
        polygons,
        facecolors=[colors[str(name)] for name in region],
        edgecolors="#334155",
        linewidths=0.25,
    )
    axes.add_collection(collection)
    axes.autoscale_view()
    axes.set_aspect("equal")
    axes.set_xlabel("x / D_tube")
    axes.set_ylabel("y / D_tube")
    axes.set_title("Sparger discharge cross-section")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    section = build_section()
    quality = quadmesh.quality_summary(section, order=8)
    aspect = _edge_aspect_ratios(section)
    region_names = section.element_tags.dense(section.n_quads)
    inside = quadmesh.select(
        section, np.isin(region_names, ("tube_1", "tube_2"))
    )
    outside = quadmesh.select(
        section, ~np.isin(region_names, ("tube_1", "tube_2"))
    )
    inner_surrounding = quadmesh.select(section, region_names == "surrounding")
    outer_annulus = quadmesh.select(section, region_names == "outer_annulus")
    jacobian_inside = quadmesh.scaled_jacobian(inside, order=8)
    jacobian_outside = quadmesh.scaled_jacobian(outside, order=8)
    jacobian_inner_surrounding = quadmesh.scaled_jacobian(
        inner_surrounding, order=8
    )
    jacobian_outer_annulus = quadmesh.scaled_jacobian(outer_annulus, order=8)
    aspect_inside = _edge_aspect_ratios(inside)
    aspect_outside = _edge_aspect_ratios(outside)
    writer.quad_to_vtu(section, str(OUTPUT))
    _write_svg_preview(section, PREVIEW)
    _write_png_preview(section, PNG_PREVIEW)

    regions, counts = np.unique(
        section.element_tags.dense(section.n_quads), return_counts=True
    )
    print(f"wrote: {OUTPUT}")
    print(f"wrote: {PREVIEW}")
    print(f"wrote: {PNG_PREVIEW}")
    print(f"points: {section.n_points}")
    print(f"quadrilaterals: {section.n_quads}")
    print("regions: " + ", ".join(f"{r}={n}" for r, n in zip(regions, counts)))
    print(f"quality at solver order 8: {quality}")
    print(
        "scaled Jacobian inside tubes:  "
        f"min={jacobian_inside.min():.6g}, "
        f"mean={jacobian_inside.mean():.6g}, "
        f"max={jacobian_inside.max():.6g}"
    )
    print(
        "scaled Jacobian outside tubes: "
        f"min={jacobian_outside.min():.6g}, "
        f"mean={jacobian_outside.mean():.6g}, "
        f"max={jacobian_outside.max():.6g}"
    )
    print(
        "  inner surrounding region:     "
        f"min={jacobian_inner_surrounding.min():.6g}, "
        f"mean={jacobian_inner_surrounding.mean():.6g}, "
        f"max={jacobian_inner_surrounding.max():.6g}"
    )
    print(
        "  outer annulus region:          "
        f"min={jacobian_outer_annulus.min():.6g}, "
        f"mean={jacobian_outer_annulus.mean():.6g}, "
        f"max={jacobian_outer_annulus.max():.6g}"
    )
    print(
        "corner-edge aspect ratio overall: "
        f"min={aspect.min():.6g}, mean={aspect.mean():.6g}, max={aspect.max():.6g}"
    )
    print(
        "corner-edge aspect ratio inside tubes:  "
        f"min={aspect_inside.min():.6g}, "
        f"mean={aspect_inside.mean():.6g}, "
        f"max={aspect_inside.max():.6g}"
    )
    print(
        "corner-edge aspect ratio outside tubes: "
        f"min={aspect_outside.min():.6g}, "
        f"mean={aspect_outside.mean():.6g}, "
        f"max={aspect_outside.max():.6g}"
    )


if __name__ == "__main__":
    main()
