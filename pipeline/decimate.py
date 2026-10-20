"""Blender: decimate every mesh in a models model dir into a web copy.
blender -b -P decimate.py -- <src_model_dir> <dst_model_dir> <max_faces>"""
import bpy, bmesh, sys, os, shutil, glob
src, dst, maxf = sys.argv[sys.argv.index("--") + 1:]
maxf = int(maxf)
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.png", "*.md", "CHANGELOG*"))
tot0 = tot1 = 0
for f in sorted(glob.glob(os.path.join(dst, "assets", "*"))):
    ext = f.rsplit(".", 1)[-1].lower()
    if ext not in ("obj", "stl"): continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext == "obj":
        bpy.ops.wm.obj_import(filepath=f, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.wm.stl_import(filepath=f, forward_axis="Y", up_axis="Z")
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not obs: continue
    bpy.context.view_layer.objects.active = obs[0]
    for o in obs: o.select_set(True)
    if len(obs) > 1: bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    n = len(ob.data.polygons); tot0 += n
    if n > maxf:
        mod = ob.modifiers.new("d", "DECIMATE"); mod.ratio = maxf / n
        bpy.ops.object.modifier_apply(modifier="d")
    bpy.ops.object.modifier_apply(modifier="d") if ob.modifiers else None
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(ob.data); bm.free()
    tot1 += len(ob.data.polygons)
    if ext == "obj":
        bpy.ops.wm.obj_export(filepath=f, forward_axis="Y", up_axis="Z", export_materials=False,
                              export_uv=False, export_normals=False, apply_modifiers=True, export_selected_objects=False)
    else:
        bpy.ops.wm.stl_export(filepath=f, forward_axis="Y", up_axis="Z", apply_modifiers=True)
print(f"DECIMATE {src}: {tot0} -> {tot1} faces")
