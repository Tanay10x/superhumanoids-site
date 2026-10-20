"""Remove vendor markings from the quadruped shell. The lettering is cut through the shell (base_4) and
filled by separate pieces (base_3). Merge the side pieces into the shell, weld the rims, then
Laplacian-fill each marking zone with its rim pinned. Writes a cleaned model copy."""
import numpy as np, shutil, os
SRC = "/home/tanay/sh-build/models/quadruped"; DST = "/home/tanay/sh-build/models/quadruped_clean"
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(SRC, DST)

def read(path):
    V, F = [], []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
    return np.array(V), F
def write(path, V, F):
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in V)
        fh.writelines("f " + " ".join(str(i + 1) for i in f) + "\n" for f in F)

S, SF = read(f"{SRC}/assets/base_4.obj")
L, LF = read(f"{SRC}/assets/base_3.obj")
# side wordmarks + "02" (|y| > 0.05), and the emblem on the nose flank (x 0.26-0.29, y > 0.03)
marking = ((np.abs(L[:, 1]) > 0.05) & (L[:, 0] < 0.30)) | ((L[:, 0] > 0.26) & (L[:, 0] <= 0.29) & (L[:, 1] > 0.03))
printed = L[:, 0] > 0.29          # wordmark printed on the front sensor strip: dropped outright
take = [f for f in LF if all(marking[i] for i in f)]
keep = [f for f in LF if not all(marking[i] for i in f) and not any(printed[i] for i in f)]
print("marking faces", len(take), "kept infill faces", len(keep))

# merge marking pieces into the shell and weld coincident rims (0.05 mm)
# merged pieces keep their own winding; flip any face that points into the trunk
def outward(f):
    a, b, c = L[f[0]], L[f[1]], L[f[2]]
    n = np.cross(b - a, c - a); d = (a + b + c) / 3 - np.array([0.1, 0.0, 0.0])
    return f if n @ d >= 0 else f[::-1]
take = [outward(f) for f in take]
V = np.vstack([S, L]); F = SF + [[i + len(S) for i in f] for f in take]
key = np.round(V / 5e-5).astype(np.int64)
_, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
P = V[rep].copy()
F = [[inv[i] for i in f] for f in F]
F = [f for f in F if len(set(f)) == 3]
nbr = [set() for _ in range(len(P))]
for f in F:
    for a in range(3):
        b = f[(a + 1) % 3]; nbr[f[a]].add(b); nbr[b].add(f[a])

sel = np.zeros(len(P), bool)
used = np.zeros(len(P), bool); used[np.unique(np.array(F).ravel())] = True
pad = 0.003
for side in (1, -1):
    for x0, x1 in ((-0.12, 0.11), (0.23, 0.295)):
        pts = L[marking & (np.sign(L[:, 1]) == side) & (L[:, 0] > x0) & (L[:, 0] < x1)]
        if not len(pts): continue
        lo, hi = pts.min(0) - pad, pts.max(0) + pad
        sel |= np.all((P >= lo) & (P <= hi), axis=1) & (np.sign(P[:, 1]) == side)
sel &= used
idx = np.where(sel)[0]; nb = [np.fromiter(nbr[i], int) for i in idx]
print("healed vertices", len(idx))
for _ in range(800):
    P[idx] = np.array([P[n].mean(0) if len(n) else P[i] for i, n in zip(idx, nb)])

# compact and write
usedv = np.unique(np.array(F).ravel()); remap = -np.ones(len(P), int); remap[usedv] = np.arange(len(usedv))
write(f"{DST}/assets/base_4.obj", P[usedv], [[remap[i] for i in f] for f in F])
if keep:
    usedl = np.unique(np.array(keep).ravel()); rl = -np.ones(len(L), int); rl[usedl] = np.arange(len(usedl))
    write(f"{DST}/assets/base_3.obj", L[usedl], [[rl[i] for i in f] for f in keep])
else:  # nothing of the infill survives: drop the mesh from the model
    os.remove(f"{DST}/assets/base_3.obj")
    xml = open(f"{DST}/quadruped.xml").read()
    xml = xml.replace('<mesh file="base_3.obj"/>', "").replace('<geom mesh="base_3" material="white" class="visual"/>', "")
    open(f"{DST}/quadruped.xml", "w").write(xml)
print("wrote", DST)
