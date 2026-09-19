import json
import sys
from pathlib import Path

import bpy


def export_bone_transforms(armature_obj, frame, output_path):
    bpy.context.scene.frame_set(frame)
    bone_data = {}
    for bone in armature_obj.pose.bones:
        mat = bone.matrix
        bone_data[bone.name] = [list(row) for row in mat]
    with open(output_path, 'w') as f:
        json.dump(bone_data, f, indent=2)


def sample_frames_and_bones(fbx_path, output_dir, frame_count):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    scene = bpy.context.scene

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    armature_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if not mesh_objs or not armature_objs:
        print(f"No mesh or armature found in {fbx_path}")
        return
    mesh = mesh_objs[0]
    target_obj_name = mesh.name

    armature = armature_objs[0]

    sampled_frames = list(range(scene.frame_start, frame_count + 1))

    for idx_frame, frame in enumerate(sampled_frames):
        scene.frame_set(frame)
        bpy.ops.object.select_all(action='DESELECT')
        obj = bpy.data.objects[target_obj_name]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.duplicate()
        bpy.ops.object.convert(target='MESH')

        output_path = output_dir / f"frame_{idx_frame:02d}.fbx"
        bpy.ops.export_scene.fbx(filepath=str(output_path), use_selection=True, apply_unit_scale=True,
                                 apply_scale_options='FBX_SCALE_ALL')

        bone_out = output_dir / f"frame_{idx_frame:02d}_bones.json"
        export_bone_transforms(armature, frame, bone_out)
    print(f"Processed {fbx_path}")


if __name__ == "__main__":
    assert len(sys.argv) > 3, "Usage: blender --background --python blender.py <input_fbx> <output_dir> <frame_count>"
    fbx_path = Path(sys.argv[-3])
    output_dir = Path(sys.argv[-2])
    frame_count = int(sys.argv[-1])
    sample_frames_and_bones(fbx_path, output_dir, frame_count)
