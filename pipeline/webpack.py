"""Build web sim assets: physics-only .mjb + compact visual mesh pack per robot.
Visual pack (.bin, little-endian):
  header json (utf8, length-prefixed u32) describing geoms, then concatenated buffers.
"""
import mujoco, numpy as np, json, os, struct, re, gzip
OUT = "/home/tanay/superhumanoids-site/assets/sim"
os.makedirs(OUT, exist_ok=True)
MAT = {"quadruped": {"black": "black_plastic", "white": "white_paint", "gray": "grey_shell", "metal": "dark_anodized"},
       "arm": {"white": "white_paint", "off_white": "white_paint", "black": "black_plastic", "light_blue": "led", "green": "led"}}
SKIP = set()  # quadruped markings are erased upstream (erase_lettering.py)
SAME_AS_SHELL = {"base_3"}  # quadruped lettering infill, painted like the shell so the wordmark disappears
for name, d in [("quadruped", "quadruped"), ("arm", "arm")]:
    root = f"/home/tanay/sh-build/web_models/{d}/sh_scene.xml"
    # visuals: original-resolution meshes for the arm (its overlapping shells break when decimated)
    full = mujoco.MjModel.from_xml_path(root.replace("web_models", "models") if name == "arm" else root)
    # physics: same xml with discardvisual
    xml = open(root).read().replace("<include", '<compiler discardvisual="true"/>\n  <include', 1)
    tmp = root.replace("sh_scene.xml", "sh_phys.xml"); open(tmp, "w").write(xml)
    phys = mujoco.MjModel.from_xml_path(tmp)
    mujoco.mj_saveModel(phys, f"{OUT}/{name}.mjb", None)
    geoms, bufs, off = [], [], 0
    for g in range(full.ngeom):
        if full.geom_group[g] != 2 or full.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = full.geom_dataid[g]; mname = mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_MESH, mid)
        if mname in SKIP: continue
        va, vn = int(full.mesh_vertadr[mid]), int(full.mesh_vertnum[mid])
        fa, fn = int(full.mesh_faceadr[mid]), int(full.mesh_facenum[mid])
        v = full.mesh_vert[va:va + vn].astype("<f4").tobytes()
        f = full.mesh_face[fa:fa + fn].astype("<u2" if vn < 65536 else "<u4").tobytes()
        mat = full.geom_matid[g]
        matname = mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else ""
        geoms.append({"body": mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_BODY, full.geom_bodyid[g]),
                      "pos": full.geom_pos[g].round(6).tolist(), "quat": full.geom_quat[g].round(6).tolist(),
                      "mat": "grey_shell" if mname in SAME_AS_SHELL else MAT[name].get(matname, "black_plastic"), "v": [off, vn], "f": [off + len(v), fn, 2 if vn < 65536 else 4]})
        pad = (-(len(v) + len(f))) % 4
        bufs += [v, f, b"\0" * pad]; off += len(v) + len(f) + pad
    head = json.dumps({"geoms": geoms}).encode()
    head += b" " * ((-len(head)) % 4)
    with open(f"{OUT}/{name}.vis.bin", "wb") as fh:
        fh.write(struct.pack("<I", len(head))); fh.write(head); fh.write(b"".join(bufs))
    print(name, "phys", os.path.getsize(f"{OUT}/{name}.mjb") // 1024, "KB  vis", os.path.getsize(f"{OUT}/{name}.vis.bin") // 1024, "KB  geoms", len(geoms), "bodies", phys.nbody, "nu", phys.nu)

# ship pre-gzipped (the page inflates them with DecompressionStream)
for f in os.listdir(OUT):
    if f.endswith((".mjb", ".bin")):
        src = os.path.join(OUT, f)
        with open(src, "rb") as a, gzip.open(src + ".gz", "wb", 9) as b: b.write(a.read())
        os.remove(src)
