"""Build a photoreal studio scene from a MuJoCo model and render it with Cycles.

Usage:
  blender -b -P render.py -- <robot> <out_dir> [--frames N] [--still F] [--samples S] [--res WxH]
"""
import sys, os, math, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import bpy, bmesh
import numpy as np
import mujoco
from mathutils import Matrix, Quaternion, Vector
import motion

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("robot"); ap.add_argument("out")
ap.add_argument("--frames", type=int, default=90)
ap.add_argument("--still", type=int, default=-1)
ap.add_argument("--samples", type=int, default=160)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--save", action="store_true")
ap.add_argument("--portrait", action="store_true")  # phone framing: centred, every other frame
A = ap.parse_args(argv)
W, H = map(int, A.res.split("x"))
HDRI = "/home/tanay/sh-build/hdri/studio.exr"

# --- per-robot staging -------------------------------------------------------
STAGE = {
    #        cam target       dist  focal  az0   az1   elev  key yaw
    # shift: lens shift (sensor widths) that slides the subject right of centre for copy space
    "humanoid":    dict(target=(0, 0, 0.66), dist=7.0, focal=70, az=(112, 74), elev=(4, 7), shift=-0.14),
    "quadruped":   dict(target=(0.02, 0, 0.2), dist=3.9, focal=70, az=(122, 52), elev=(16, 8), shift=-0.13),
    "arm": dict(target=(0.3, 0, 0.42), dist=4.6, focal=70, az=(-34, 18), elev=(12, 18), shift=-0.13),
}[A.robot]

# --- scene reset -------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            break
    except Exception:
        pass
for d in prefs.devices:
    d.use = d.type != "CPU"
scn.cycles.device = "GPU"
scn.cycles.samples = A.samples
scn.cycles.adaptive_threshold = 0.015
scn.cycles.use_denoising = True
scn.cycles.denoiser = "OPENIMAGEDENOISE"
scn.cycles.max_bounces = 8
scn.cycles.glossy_bounces = 4
scn.cycles.transparent_max_bounces = 4
scn.render.resolution_x, scn.render.resolution_y = W, H
scn.render.resolution_percentage = 100
scn.render.film_transparent = False
scn.render.use_persistent_data = True
scn.view_settings.view_transform = "AgX"
scn.view_settings.look = "AgX - Medium High Contrast"
scn.view_settings.exposure = -0.6
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_depth = "8"

# --- materials ---------------------------------------------------------------
def nodes_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
    return m, nt, bsdf

def micro_rough(nt, bsdf, base, var, scale=180.0, bump=0.04):
    """Break up perfect CG roughness with fine noise + faint bump (handling marks)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = base - var; mr.inputs["To Max"].default_value = base + var
    nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = bump
    bn.inputs["Distance"].default_value = 0.0005
    nz2 = nt.nodes.new("ShaderNodeTexNoise"); nz2.inputs["Scale"].default_value = scale * 6
    nt.links.new(tc.outputs["Object"], nz2.inputs["Vector"])
    nt.links.new(nz2.outputs["Fac"], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], bsdf.inputs["Normal"])

def make_mat(kind, rgba):
    m, nt, b = nodes_mat(kind)
    c = list(rgba[:3]) + [1.0]
    if kind == "aluminium":          # machined / bead-blasted aluminium
        b.inputs["Base Color"].default_value = (0.62, 0.63, 0.64, 1)
        b.inputs["Metallic"].default_value = 1.0
        micro_rough(nt, b, 0.40, 0.035, scale=260, bump=0.012)
    elif kind == "dark_anodized":
        b.inputs["Base Color"].default_value = (0.035, 0.037, 0.04, 1)
        b.inputs["Metallic"].default_value = 0.85
        micro_rough(nt, b, 0.38, 0.06)
    elif kind == "black_plastic":
        b.inputs["Base Color"].default_value = (0.018, 0.018, 0.02, 1)
        b.inputs["Metallic"].default_value = 0.0
        micro_rough(nt, b, 0.48, 0.08, scale=90)
        b.inputs["Coat Weight"].default_value = 0.08
    elif kind == "white_paint":       # satin powder-coat / painted ABS shell
        b.inputs["Base Color"].default_value = (0.78, 0.785, 0.79, 1)
        micro_rough(nt, b, 0.36, 0.05, scale=120, bump=0.02)
        b.inputs["Coat Weight"].default_value = 0.25
        b.inputs["Coat Roughness"].default_value = 0.18
        b.inputs["Subsurface Weight"].default_value = 0.02
    elif kind == "grey_shell":
        b.inputs["Base Color"].default_value = (0.42, 0.44, 0.48, 1)
        micro_rough(nt, b, 0.34, 0.05, scale=120, bump=0.02)
        b.inputs["Coat Weight"].default_value = 0.3
        b.inputs["Coat Roughness"].default_value = 0.15
    elif kind == "accent":
        b.inputs["Base Color"].default_value = c
        micro_rough(nt, b, 0.3, 0.04)
    elif kind == "led":
        b.inputs["Base Color"].default_value = c
        b.inputs["Emission Color"].default_value = c
        b.inputs["Emission Strength"].default_value = 2.0
    return m

def classify(robot, matname):
    n = (matname or "").lower()
    if robot == "humanoid":
        return "aluminium" if n == "metal" else "black_plastic"
    if robot == "quadruped":
        return {"black": "black_plastic", "white": "white_paint", "gray": "grey_shell", "metal": "dark_anodized"}.get(n, "black_plastic")
    if robot == "arm":
        return {"white": "white_paint", "off_white": "white_paint", "black": "black_plastic", "light_blue": "led", "green": "led"}.get(n, "black_plastic")
    return "black_plastic"

MATS = {}
def get_mat(kind, rgba):
    key = (kind, tuple(np.round(rgba, 2))) if kind in ("accent", "led") else kind
    if key not in MATS:
        MATS[key] = make_mat(kind, rgba)
    return MATS[key]

# --- robot from mujoco -------------------------------------------------------
model, frames, extras = motion.ROBOTS[A.robot](A.frames)
data = mujoco.MjData(model)
SKIP_MESHES = {"logo_link"}  # vendor wordmark (humanoid)
# quadruped wordmark: base_3 is the lettering infill; painting it like the shell fills the engraving flush
SAME_AS_SHELL = {"base_3"}
geoms = [g for g in range(model.ngeom) if model.geom_group[g] == 2 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
         and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) not in SKIP_MESHES]
objs = {}
mesh_cache = {}
for g in geoms:
    mid = model.geom_dataid[g]
    if mid not in mesh_cache:
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn].astype(float)
        faces = model.mesh_face[fa:fa + fn].astype(int)
        me = bpy.data.meshes.new(f"mesh{mid}")
        me.from_pydata(verts.tolist(), [], faces.tolist())
        me.validate(); me.update()
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me); bm.free()
        me.shade_smooth()
        try:
            me.set_sharp_from_angle(angle=math.radians(32))
        except Exception:
            pass
        mesh_cache[mid] = me
    me = mesh_cache[mid]
    ob = bpy.data.objects.new(f"g{g}", me)
    bpy.context.collection.objects.link(ob)
    mat = model.geom_matid[g]
    mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else None
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    # per-object material via object-linked slot so shared meshes can differ
    if len(me.materials) == 0:
        me.materials.append(None)
    ob.material_slots[0].link = "OBJECT"
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
    kind = "grey_shell" if gname in SAME_AS_SHELL else classify(A.robot, mname)
    ob.material_slots[0].material = get_mat(kind, rgba)
    ob.rotation_mode = "QUATERNION"
    objs[g] = ob

PATCHES = []
# prop cube for the arm
cube_ob = None
if "cube" in extras:
    bpy.ops.mesh.primitive_cube_add(size=0.05)
    cube_ob = bpy.context.active_object
    bev = cube_ob.modifiers.new("bev", "BEVEL"); bev.width = 0.002; bev.segments = 3
    cm, cnt, cb = nodes_mat("anodized_blue")
    cb.inputs["Base Color"].default_value = (0.09, 0.22, 0.62, 1)
    cb.inputs["Metallic"].default_value = 0.9
    micro_rough(cnt, cb, 0.32, 0.05)
    cube_ob.data.materials.append(cm)
    cube_ob.rotation_mode = "QUATERNION"

# plinth for the arm
if A.robot == "arm":
    bpy.ops.mesh.primitive_cube_add(size=1)
    pl = bpy.context.active_object
    pl.scale = (0.26, 0.26, 0.02); pl.location = (0, 0, -0.01)
    bv = pl.modifiers.new("bev", "BEVEL"); bv.width = 0.003; bv.segments = 3
    pl.data.materials.append(get_mat("aluminium", (1, 1, 1, 1)))
    # table the cube sits on is the floor; nothing else

def pose_at(i):
    data.qpos[:] = frames[i]
    mujoco.mj_kinematics(model, data)
    for g, ob in objs.items():
        p = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
        ob.location = Vector(p); ob.rotation_quaternion = Quaternion(q)
    for pl in PATCHES:  # everything in PATCHES is authored in the trunk's body frame
        pl.location = Vector(data.xpos[1]); pl.rotation_quaternion = Quaternion(data.xquat[1])
    if cube_ob:
        p, q = extras["cube"][i]
        cube_ob.location = Vector(p); cube_ob.rotation_quaternion = Quaternion(q)

# ground contact: lowest vertex over all frames sits on the floor
pose_at(0)
bpy.context.view_layer.update()
zmin = min((ob.matrix_world @ v.co).z for ob in objs.values() for v in ob.data.vertices)
floor_z = zmin if A.robot != "arm" else 0.0

# --- cyclorama ---------------------------------------------------------------
def cyclorama(z0):
    prof = []
    R = 2.6
    for y in np.linspace(-14, 3, 18): prof.append((y, z0))
    for a in np.linspace(-math.pi / 2, 0, 24)[1:]:
        prof.append((3 + R * math.cos(a), z0 + R + R * math.sin(a)))
    for z in np.linspace(z0 + R, z0 + 12, 8)[1:]: prof.append((3 + R, z))
    xs = np.linspace(-16, 16, 2)
    bm = bmesh.new()
    rows = [[bm.verts.new((x, y, z)) for (y, z) in prof] for x in xs]
    for i in range(len(xs) - 1):
        for j in range(len(prof) - 1):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    me = bpy.data.meshes.new("cyc"); bm.to_mesh(me); bm.free()
    me.shade_smooth()
    ob = bpy.data.objects.new("cyc", me); bpy.context.collection.objects.link(ob)
    ob.rotation_euler[2] = math.radians(sum(STAGE["az"]) / 2)  # backdrop faces the camera path
    # CC0 "smooth_concrete_floor" texture: sealed concrete infinity cove
    m, nt, b = nodes_mat("cyc_concrete")
    T = "/home/tanay/sh-build/tex/conc_"
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.45, 0.45, 0.45)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    def img(name, noncolor):
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(T + name)
        n.projection = "BOX"; n.projection_blend = 0.3
        if noncolor: n.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"]); return n
    dif = img("diff.jpg", False); rgh = img("rough.jpg", True); nor = img("nor_gl.jpg", True)
    tint = nt.nodes.new("ShaderNodeMix"); tint.data_type = "RGBA"; tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0; tint.inputs["B"].default_value = (0.42, 0.43, 0.44, 1)
    hs = nt.nodes.new("ShaderNodeHueSaturation"); hs.inputs["Saturation"].default_value = 0.08
    nt.links.new(dif.outputs["Color"], hs.inputs["Color"])
    nt.links.new(hs.outputs["Color"], tint.inputs["A"]); nt.links.new(tint.outputs["Result"], b.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.32; mr.inputs["To Max"].default_value = 0.62
    nt.links.new(rgh.outputs["Color"], mr.inputs["Value"]); nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.35
    nt.links.new(nor.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    me.materials.append(m)
cyclorama(floor_z)

# --- lighting ----------------------------------------------------------------
world = bpy.data.worlds.new("w"); scn.world = world; world.use_nodes = True
wn = world.node_tree; env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(HDRI)
mp = wn.nodes.new("ShaderNodeMapping"); tc = wn.nodes.new("ShaderNodeTexCoord")
mp.inputs["Rotation"].default_value[2] = math.radians(110 + sum(STAGE["az"]) / 2)
wn.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); wn.links.new(mp.outputs["Vector"], env.inputs["Vector"])
wn.nodes["Background"].inputs["Strength"].default_value = 0.25
wn.links.new(env.outputs["Color"], wn.nodes["Background"].inputs["Color"])

def area(name, loc, size, power, color=(1, 1, 1), aim=(0, 0, 0.6), shape="RECTANGLE", sy=None):
    rz = Matrix.Rotation(math.radians(sum(STAGE["az"]) / 2), 3, "Z")
    loc = tuple(rz @ Vector(loc))
    L = bpy.data.lights.new(name, "AREA"); L.energy = power; L.color = color
    L.shape = shape; L.size = size; L.size_y = sy or size
    o = bpy.data.objects.new(name, L); bpy.context.collection.objects.link(o)
    o.location = loc
    d = Vector(aim) - Vector(loc); o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return o

tz = STAGE["target"][2]
area("key", (-3.2, -3.0, 3.6 + floor_z), 2.4, 520, (1.0, 0.97, 0.93), aim=(0, 0, tz), sy=1.4)
area("fill", (3.8, -2.2, 1.6 + floor_z), 3.0, 70, (0.92, 0.96, 1.0), aim=(0, 0, tz))
area("rim", (1.8, 2.6, 2.8 + floor_z), 0.6, 700, (1, 1, 1), aim=(0, 0, tz), sy=3.0)
area("top", (0, 0, 5.5 + floor_z), 3.5, 160, (1, 1, 1), aim=(0, 0, 0))

# --- camera ------------------------------------------------------------------
cam_d = bpy.data.cameras.new("cam"); cam_d.lens = STAGE["focal"]; cam_d.sensor_width = 36
cam_d.shift_x = 0 if A.portrait else STAGE["shift"]; cam_d.shift_y = {"humanoid": 0.0, "quadruped": -0.13, "arm": -0.11}[A.robot] if A.portrait else 0.0; cam_d.dof.use_dof = True; cam_d.dof.aperture_fstop = 5.6
cam = bpy.data.objects.new("cam", cam_d); bpy.context.collection.objects.link(cam); scn.camera = cam

def cam_at(t):
    a = motion.smooth(t)
    az = math.radians(STAGE["az"][0] * (1 - a) + STAGE["az"][1] * a)
    el = math.radians(STAGE["elev"][0] * (1 - a) + STAGE["elev"][1] * a)
    dist = STAGE["dist"] * (1 - 0.06 * a)
    tx, ty, tz = STAGE["target"]; tz += floor_z
    pos = Vector((tx + dist * math.sin(az) * math.cos(el), ty - dist * math.cos(az) * math.cos(el), tz + dist * math.sin(el)))
    cam.location = pos
    cam.rotation_euler = (Vector((tx, ty, tz)) - pos).to_track_quat("-Z", "Y").to_euler()
    cam_d.dof.focus_distance = (Vector((tx, ty, tz)) - pos).length

# --- render ------------------------------------------------------------------
os.makedirs(A.out, exist_ok=True)
telemetry = []
idx = [A.still] if A.still >= 0 else range(0, A.frames, 2 if A.portrait else 1)
for i in idx:
    pose_at(i)
    cam_at(i / max(1, A.frames - 1))
    if A.still < 0 or True:
        telemetry.append({"f": i, "q": [round(float(x), 4) for x in frames[i][-min(len(frames[i]), 43):]]})
    scn.render.filepath = os.path.join(A.out, f"{i:04d}.png")
    if A.still < 0 and os.path.exists(scn.render.filepath):
        continue
    bpy.ops.render.render(write_still=True)
    print(f"FRAME {i} done", flush=True)
if A.save:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(A.out, "scene.blend"))
jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
json.dump({"joints": jn, "nq": model.nq, "frames": [[round(float(x), 4) for x in f] for f in frames]},
          open(os.path.join(A.out, "motion.json"), "w"))
