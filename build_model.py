"""Procedural stand-in: spheres and capsules on a sofa.

The printable files army_man_sofa.stl / .3mf come from the reference photo
via generate_from_reference.py. This script writes procedural_* files so it
does not replace that mesh.
"""
import numpy as np
import trimesh
from manifold3d import Manifold, OpType

SEG = 40


def sphere(c, r):
    return Manifold.sphere(r, SEG).translate(tuple(c))


def capsule(a, b, ra, rb=None):
    return Manifold.batch_hull([sphere(a, ra), sphere(b, rb if rb else ra)])


def chain(points, radii):
    return union([capsule(points[i], points[i + 1], radii[i], radii[i + 1])
                  for i in range(len(points) - 1)])


def rbox(lo, hi, r):
    """Box from lo to hi with rounded edges of radius r."""
    lo, hi = np.array(lo, float) + r, np.array(hi, float) - r
    corners = [sphere((x, y, z), r) for x in (lo[0], hi[0])
               for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return Manifold.batch_hull(corners)


def cyl(a, b, r):
    """Cylinder between points a and b."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    h = np.linalg.norm(d)
    m = Manifold.cylinder(h, r, r, SEG)
    d /= h
    pitch = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return m.rotate((0, pitch, 0)).rotate((0, 0, yaw)).translate(tuple(a))


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add)


def build_base():
    w, d, h, r = 108, 66, 4, 10
    cy = -4
    pts = [Manifold.cylinder(h, r, r, SEG).translate((sx * (w / 2 - r), cy + sy * (d / 2 - r), 0))
           for sx in (-1, 1) for sy in (-1, 1)]
    return Manifold.batch_hull(pts)


def build_sofa():
    parts = []
    for x in (-34, 34):
        for y in (-17, 17):
            parts.append(Manifold.cylinder(5, 2.4, 2.0, SEG).translate((x, y, 3.5)))
    parts.append(rbox((-31, -20, 8), (31, 18, 22), 1.5))           # seat frame
    parts.append(rbox((-30.5, -21, 21), (-0.4, 14, 28), 2.5))      # left cushion
    parts.append(rbox((0.4, -21, 21), (30.5, 14, 28), 2.5))        # right cushion
    parts.append(rbox((-37, 12, 8), (37, 22, 46), 3))              # backrest
    parts.append(cyl((-35, 17, 44), (35, 17, 44), 5.5))            # backrest roll
    parts.append(rbox((-30, 9, 26), (30, 15, 44), 3))              # back cushion
    for sx in (-1, 1):
        x0 = sx * 35
        parts.append(rbox((x0 - 5, -22, 8), (x0 + 5, 22, 34), 2))  # arm body
        parts.append(cyl((x0, -22, 35), (x0, 22, 35), 6.5))        # rolled arm top
        parts.append(cyl((x0, -23.2, 35), (x0, -21, 35), 5.0))     # scroll disc
        parts.append(rbox((x0 - 4.2, -23.2, 9), (x0 + 4.2, -21, 33), 1))  # front panel
    return union(parts)


def build_soldier():
    p = []
    # Torso reclined against the left arm, chest facing +X/up.
    hip_l, hip_r = (-6, 4, 31), (-6, -5, 31)
    sh_l, sh_r = (-24, 8, 47), (-24, -8, 47)
    p.append(Manifold.batch_hull([sphere(hip_l, 6.5), sphere(hip_r, 6.5),
                                  sphere(sh_l, 5.5), sphere(sh_r, 5.5),
                                  sphere((-17, 0, 43), 8)]))
    # Belt with ammo pouches.
    p.append(capsule((-9.5, 8.5, 34), (-9.5, -9.5, 34), 3.2))
    for y in (-6, -1.5, 3):
        p.append(rbox((-6.5, y - 2, 32), (-2.5, y + 2, 38), 0.8))
    # Collar / neck / head.
    p.append(capsule((-26, 0, 50), (-28, 1, 54), 3.6))
    head_c = np.array((-28.5, 1, 58.5))
    p.append(sphere(head_c, 6.0))
    p.append(sphere(head_c + (5.3, 0, -0.5), 1.6))                 # nose
    # Helmet: dome + brim.
    dome = sphere(head_c + (0, 0, 1.5), 8.2) ^ Manifold.cube((30, 30, 15)).translate(
        tuple(head_c + (-15, -15, 0.5)))
    p.append(dome)
    p.append(Manifold.cylinder(1.3, 9.5, 9.0, SEG).translate(tuple(head_c + (0, 0, 0.3))))
    # Arms raised to hold the phone in front of the face.
    p.append(chain([sh_r, (-15, -11, 40), (-9, -5, 52)], [4.0, 3.4, 3.0]))
    p.append(chain([sh_l, (-14, 9, 41), (-9, 2, 53)], [4.0, 3.4, 3.0]))
    p.append(sphere((-8.5, -4.5, 53.5), 3.3))
    p.append(sphere((-8.5, 1.5, 54.5), 3.3))
    phone = rbox((-1.3, -3.4, -6), (1.3, 3.4, 6), 0.8).rotate((0, -25, 0))
    p.append(phone.translate((-8.0, -1.5, 58)))
    # Raised leg crossed over the right armrest.
    up_leg = [(-4, -5, 31), (14, -8, 43), (32, -8, 45.5)]
    p.append(chain(up_leg, [6.0, 4.8, 4.0]))
    p.append(capsule((32, -8, 45.5), (36, -8, 47), 4.6))           # boot shaft
    p.append(Manifold.batch_hull([sphere((37, -8, 47), 4.4),
                                  sphere((43, -8, 53), 3.4),
                                  sphere((45, -8, 50.5), 3.0)]))   # boot foot
    # Lowered leg, boot planted on the base in front of the sofa.
    low_leg = [(-4, 3, 30), (16, -15, 33), (20, -27, 11)]
    p.append(chain(low_leg, [6.0, 4.8, 4.0]))
    p.append(capsule((20, -27, 11), (20, -27, 7.5), 4.6))          # boot shaft
    p.append(Manifold.batch_hull([sphere((20, -27, 6.8), 4.0),
                                  sphere((29, -27.5, 6.5), 3.0)]))  # boot foot
    p.append(rbox((15, -31.5, 3.5), (32, -23, 5.5), 1))            # sole
    # Cargo pocket on the lowered thigh.
    p.append(rbox((4, -11, 34), (11, -5, 38), 1))
    return union(p)


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=mesh.vert_properties[:, :3],
                           faces=mesh.tri_verts, process=True)


def render(mesh, path, views):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(6 * len(views), 6), dpi=150)
    tris = mesh.vertices[mesh.faces]
    light = np.array([0.4, -0.6, 0.8])
    light /= np.linalg.norm(light)
    shade = np.clip(mesh.face_normals @ light, 0, 1) * 0.65 + 0.35
    base = np.array([0.20, 0.62, 0.30])
    colors = np.clip(shade[:, None] * base, 0, 1)
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, linewidths=0))
        lo, hi = mesh.bounds
        c, s = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - s, c[0] + s)
        ax.set_ylim(c[1] - s, c[1] + s)
        ax.set_zlim(c[2] - s, c[2] + s)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    model = union([build_base(), build_sofa(), build_soldier()])
    mesh = to_trimesh(model)
    print("watertight:", mesh.is_watertight, "| genus:", model.genus(),
          "| tris:", len(mesh.faces))
    print("size (mm):", np.round(mesh.extents, 1), "| volume (cm3):",
          round(mesh.volume / 1000, 1))
    mesh.export("procedural_army_man.stl")
    mesh.export("procedural_army_man.3mf")
    render(mesh, "procedural_preview_front.png", [(18, -62, "Front 3/4")])
    render(mesh, "procedural_preview_views.png",
           [(18, -62, "Front 3/4"), (10, -90, "Front"), (25, 35, "Back 3/4")])


if __name__ == "__main__":
    main()
