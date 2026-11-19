"""Turn the raw generated mesh into a bed-ready, watertight print.

Units are millimetres. The longest horizontal span becomes TARGET_MM so the
sofa, helmet ridges, and boot laces stay large enough for a 0.4 mm nozzle.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
TARGET_MM = 140.0  # longest horizontal extent


def largest_component(mesh):
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.faces) > 50]
    if not parts:
        return mesh
    return max(parts, key=lambda p: p.area)


def end_spread(vertices, axis, at_min):
    coords = vertices[:, axis]
    lo, hi = coords.min(), coords.max()
    span = hi - lo
    if span <= 0:
        return 0.0
    band = 0.08 * span
    if at_min:
        pts = vertices[coords <= lo + band]
    else:
        pts = vertices[coords >= hi - band]
    if len(pts) < 30:
        return 0.0
    others = [i for i in range(3) if i != axis]
    return float(np.ptp(pts[:, others[0]]) * np.ptp(pts[:, others[1]]))


def orient_base_down(mesh):
    """Put the wide oval base on Z = 0. Image-to-3D up-axis is not fixed."""
    best = None
    for axis in range(3):
        for at_min in (True, False):
            score = end_spread(mesh.vertices, axis, at_min)
            if best is None or score > best[0]:
                best = (score, axis, at_min)
    _, axis, at_min = best
    # Rotate `axis` onto Z. If the wide end is the high end, flip afterwards.
    if axis == 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    if not at_min:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    # The axis rotation can reverse the up direction. Keep the wider end down.
    if end_spread(mesh.vertices, 2, at_min=False) > end_spread(mesh.vertices, 2, at_min=True):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return mesh


def flatten_base(mesh, band_mm=0.6):
    """Squash a slightly domed underside onto the bed without opening the shell."""
    z = mesh.vertices[:, 2]
    floor = np.quantile(z, 0.01)
    mesh.vertices[:, 2] -= floor
    low = mesh.vertices[:, 2] < band_mm
    mesh.vertices[low, 2] = 0.0
    return mesh


def repair(mesh):
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)
    mesh.merge_vertices()
    if mesh.is_watertight and mesh.volume > 0:
        return mesh

    from manifold3d import Manifold, Mesh

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    solid = Manifold(Mesh(vert_properties=verts, tri_verts=faces))
    if solid.is_empty():
        raise RuntimeError("mesh could not be made manifold")
    out = solid.to_mesh()
    fixed = trimesh.Trimesh(
        vertices=out.vert_properties[:, :3],
        faces=out.tri_verts,
        process=True,
    )
    if fixed.volume < 0:
        fixed.invert()
    return fixed


def render_views(mesh, path, views):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    show = mesh
    if len(mesh.faces) > 140000:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=120000)
        m = ms.current_mesh()
        show = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix())
    show.fix_normals()

    tris = show.vertices[show.faces]
    light = np.array([0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    shade = np.clip(show.face_normals @ light, 0, 1) * 0.55 + 0.4
    base = np.array([0.18, 0.62, 0.28])
    colors = np.clip(shade[:, None] * base, 0, 1)

    fig = plt.figure(figsize=(6 * len(views), 6), dpi=140)
    lo, hi = show.bounds
    center = (lo + hi) / 2
    radius = (hi - lo).max() / 2
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, linewidths=0, edgecolors="none"))
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        if title:
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "army_man_raw.glb"
    mesh = trimesh.load(src, force="mesh")
    mesh = largest_component(mesh)
    mesh = orient_base_down(mesh)

    horizontal = max(mesh.extents[0], mesh.extents[1])
    mesh.apply_scale(TARGET_MM / horizontal)
    mesh = flatten_base(mesh)
    mesh = repair(mesh)

    print(
        "watertight", mesh.is_watertight,
        "faces", len(mesh.faces),
        "size_mm", np.round(mesh.extents, 1).tolist(),
        "volume_cm3", round(abs(mesh.volume) / 1000, 1),
    )
    mesh.export(ROOT / "army_man_sofa.stl")
    mesh.export(ROOT / "army_man_sofa.3mf")
    render_views(mesh, ROOT / "preview_front.png", [(16, -68, "")])
    render_views(
        mesh,
        ROOT / "preview_views.png",
        [(16, -68, "Front 3/4"), (8, -90, "Front"), (18, 24, "Back 3/4")],
    )
    print("wrote army_man_sofa.stl army_man_sofa.3mf")


if __name__ == "__main__":
    main()
