"""
Export all actors as FBX.
Split InstancedFoliageActor into separate FBX files based on static mesh assets.
"""
import json
import os
from collections import namedtuple, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import unreal

# ==================== Data Classes ====================
Label = namedtuple("Label", ["name", "id", "folder_names"])

# folder-semantic mapping
labels = [
    Label(name="None", id=0, folder_names=[]),
    Label(name="Road", id=1, folder_names=["Road", "Roads"]),
    Label(name="Sidewalk", id=2, folder_names=["SideWalk", "Sidewalks"]),
    Label(name="Building", id=3, folder_names=["Building", "Buildings"]),
    Label(name="Wall", id=4, folder_names=["Wall", "Walls"]),
    Label(name="Fence", id=5, folder_names=["Fence", "Fences"]),
    Label(name="Pole", id=6, folder_names=["Pole", "Poles", "StreetLight", "Streets lights"]),
    Label(name="TrafficLight", id=7, folder_names=["TrafficLight", "TrafficLights", "Traffic Light", "Traffic Lights"]),
    Label(name="TrafficSign", id=8, folder_names=["TrafficSign", "TrafficSigns", "Traffic Sign", "Traffic Signs"]),
    Label(name="Vegetation", id=9, folder_names=["Vegetation", "Bush"]),
    Label(name="Terrain", id=10, folder_names=["Terrain"]),
    Label(name="Sky", id=11, folder_names=["Sky"]),
    Label(name="Pedestrian", id=12, folder_names=["Pedestrian"]),
    Label(name="Rider", id=13, folder_names=["Rider"]),
    Label(name="Car", id=14, folder_names=["Car"]),
    Label(name="Truck", id=15, folder_names=["Truck"]),
    Label(name="Bus", id=16, folder_names=["Bus"]),
    Label(name="Train", id=17, folder_names=["Train"]),
    Label(name="Motorcycle", id=18, folder_names=["Motorcycle"]),
    Label(name="Bicycle", id=19, folder_names=["Bicycle"]),
    Label(name="Static", id=20, folder_names=["Static"]),
    Label(name="Dynamic", id=21, folder_names=["Dynamic"]),
    Label(name="Other", id=22, folder_names=["Other"]),
    Label(name="Water", id=23, folder_names=["Water"]),
    Label(name="RoadLine", id=24, folder_names=["RoadLine"]),
    Label(name="Ground", id=25, folder_names=["Ground"]),
    Label(name="Bridge", id=26, folder_names=["Bridge"]),
    Label(name="RailTrack", id=27, folder_names=["RailTrack"]),
    Label(name="GuardRail", id=28, folder_names=["GuardRail"]),
    Label(name="Rock", id=29, folder_names=["Rock", "Stone"]),
]

_FOLDER_LABEL_LOOKUP: Dict[str, "Label"] = {
    name.lower(): label
    for label in labels
    for name in label.folder_names
}


# ==================== Configuration ====================
@dataclass
class ExportConfig:
    """Configuration for scene export"""
    export_dir: str
    target_actors: Set[str]
    invalid_actors: Set[str]
    invalid_actor_classes: Tuple
    skip_existing: bool
    show_progress: bool

    # Export options
    export_background: bool
    export_foreground: bool

    # FBX options
    fbx_export_morph_targets: bool
    fbx_export_preview_mesh: bool
    fbx_level_of_detail: bool
    fbx_collision: bool
    fbx_vertex_color: bool
    fbx_ascii: bool

    # Semantic labels
    semantic_labels: List[Label]

    # Foreground actors
    foreground_vehicles: Dict[str, Dict[str, str]]
    fg_pedestrian_anim_path: str
    fg_animation_fps: int

    @staticmethod
    def _class_name_to_unreal_class(class_name: str):
        """Convert class name string to Unreal class object"""
        return getattr(unreal, class_name, None)

    @classmethod
    def from_json(cls, json_path: str) -> 'ExportConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert lists to sets
        data["target_actors"] = set(data["target_actors"])
        data["invalid_actors"] = set(data["invalid_actors"])

        # Convert class names to Unreal class objects
        data['invalid_actor_classes'] = tuple(
            cls._class_name_to_unreal_class(name)
            for name in data['invalid_actor_classes']
        )

        # Parse semantic labels
        data["semantic_labels"] = labels

        # Flatten nested export_options
        export_opts = data.pop('export_options')
        data.update(export_opts)

        # Flatten nested fbx_options
        fbx_opts = data.pop('fbx_options')
        data.update({f'fbx_{k}': v for k, v in fbx_opts.items()})

        # Flatten foreground_pedestrians
        fg_ped = data.pop('foreground_pedestrians')
        data['fg_pedestrian_anim_path'] = fg_ped['animation_path']
        data['fg_animation_fps'] = fg_ped['fps']

        Logger.info(f"Loaded configuration from: {json_path}")
        return cls(**data)


# ==================== Utility Functions ====================
class Logger:
    """Simple structured logger for export operations"""

    @staticmethod
    def info(message: str) -> None:
        """Log info message"""
        unreal.log(f"[INFO] {message}")

    @staticmethod
    def warning(message: str) -> None:
        """Log warning message"""
        unreal.log_warning(f"[WARNING] {message}")

    @staticmethod
    def error(message: str) -> None:
        """Log error message"""
        unreal.log_error(f"[ERROR] {message}")

    @staticmethod
    def section(title: str) -> None:
        """Log section separator"""
        unreal.log(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    @staticmethod
    def progress(current: int, total: int, item_name: str = "") -> None:
        """Log progress"""
        percentage = (current / total * 100) if total > 0 else 0
        unreal.log(f"[PROGRESS] {current}/{total} ({percentage:.1f}%) - {item_name}")


def file_exists_skip(file_path: str, skip_existing: bool = True) -> bool:
    """Check if file exists and should skip"""
    if os.path.exists(file_path) and skip_existing:
        Logger.info(f"File exists, skipping: {file_path}")
        return True
    return False


def transform_to_dict(transform: unreal.Transform) -> Dict[str, Any]:
    """Convert Unreal Transform to dictionary."""
    return {
        "location": vector_to_dict(transform.translation),
        "rotation": rotator_to_dict(transform.rotation.rotator()),
        "scale": vector_to_dict(transform.scale3d),
    }


def vector_to_dict(vec: unreal.Vector) -> Dict[str, float]:
    """Convert Unreal Vector to dictionary."""
    return {"x": vec.x, "y": vec.y, "z": vec.z}


def rotator_to_dict(rot: unreal.Rotator) -> Dict[str, float]:
    """Convert Unreal Rotator to dictionary."""
    return {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll}


# ==================== Semantic Label Functions ====================
def get_label_by_folder_name(folder_name: str, semantic_labels: List[Label]) -> Label:
    """Get semantic label based on folder name (O(1) lookup)."""
    return _FOLDER_LABEL_LOOKUP.get(folder_name.strip().lower(), semantic_labels[0])


def get_label_by_mesh_path(mesh_path: str, semantic_labels: List[Label]) -> Label:
    """
    Get semantic label based on mesh asset path (CARLA official implementation)
    
    Reference:
        https://github.com/carla-simulator/carla/blob/ue5-dev/Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Game/Tagger.h
    """
    if not mesh_path:
        return semantic_labels[0]

    path_components = mesh_path.split("/")
    idx = 5 if "UE5UseOnly" in mesh_path else 4

    if len(path_components) > idx:
        folder_name = path_components[idx]
        return get_label_by_folder_name(folder_name, semantic_labels)

    return semantic_labels[0]


def get_actor_semantic_label(actor: unreal.Actor, semantic_labels: List[Label]) -> str:
    """Get semantic label for an actor using CARLA official method"""
    # Special case: Landscape
    if isinstance(actor, unreal.Landscape):
        return "Terrain"

    # Try to get label from mesh paths
    components = actor.get_components_by_class(unreal.StaticMeshComponent)
    if components:
        label_names = []
        for component in components:
            static_mesh = component.static_mesh
            if static_mesh:
                mesh_path = static_mesh.get_path_name()
                label = get_label_by_mesh_path(mesh_path, semantic_labels)
                if label.id != 0:  # Not "None" label
                    label_names.append(label.name)

        # Fixed logic: check if we found any valid labels
        if label_names:
            # Return the most common label, or first one if all unique
            return Counter(label_names).most_common(1)[0][0]

    # Fallback: use project folder path
    folder_path = str(actor.get_folder_path())
    if folder_path and folder_path != "None":
        folder_name = folder_path.split("/")[0] if "/" in folder_path else folder_path
        label = get_label_by_folder_name(folder_name, semantic_labels)
        if label.id != 0:
            return label.name

    # Return "None" label as last resort
    return semantic_labels[0].name


# ==================== Export Functions ====================
def create_fbx_export_options(config: Optional[ExportConfig] = None) -> unreal.FbxExportOption:
    """Create standardized FBX export options."""
    fbx_option = unreal.FbxExportOption()
    fbx_option.export_local_time = False
    fbx_option.bake_material_inputs = unreal.FbxMaterialBakeMode.DISABLED
    if config:
        fbx_option.export_morph_targets = config.fbx_export_morph_targets
        fbx_option.export_preview_mesh = config.fbx_export_preview_mesh
        fbx_option.level_of_detail = config.fbx_level_of_detail
        fbx_option.collision = config.fbx_collision
        fbx_option.vertex_color = config.fbx_vertex_color
        fbx_option.ascii = config.fbx_ascii
    return fbx_option


def export_static_mesh(static_mesh: unreal.StaticMesh,
                       output_path: str,
                       skip_existing: bool = True,
                       fbx_options: Optional[unreal.FbxExportOption] = None) -> bool:
    """
    Export a static mesh to FBX format.

    Args:
        static_mesh: Unreal StaticMesh object to export
        output_path: Full path for the output FBX file
        skip_existing: Skip if file already exists
        fbx_options: Pre-created FBX export options

    Returns:
        True if export succeeded or skipped, False if failed
    """
    if not isinstance(static_mesh, unreal.StaticMesh):
        Logger.error(f"Invalid static_mesh type: {type(static_mesh)}")
        return False

    if file_exists_skip(output_path, skip_existing):
        return True

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if fbx_options is None:
        fbx_options = create_fbx_export_options()

    export_task = unreal.AssetExportTask()
    export_task.object = static_mesh
    export_task.filename = output_path
    export_task.automated = True
    export_task.prompt = False
    export_task.replace_identical = True
    export_task.options = fbx_options

    success = unreal.Exporter.run_asset_export_task(export_task)
    result_msg = "SUCCESS" if success else "FAILED"
    Logger.info(f"[{result_msg}] Exported static mesh to {output_path}")

    return success


def export_actor_to_fbx(actor: unreal.Actor,
                        output_path: str,
                        skip_existing: bool = True,
                        fbx_options: Optional[unreal.FbxExportOption] = None) -> bool:
    """
    Export an actor (with all its components) to FBX format.

    Args:
        actor: Actor to export
        output_path: Full path for the output FBX file
        skip_existing: Skip if file already exists
        fbx_options: Pre-created FBX export options

    Returns:
        True if export succeeded or skipped, False if failed
    """
    if file_exists_skip(output_path, skip_existing):
        return True

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if fbx_options is None:
        fbx_options = create_fbx_export_options()

    # Ungroup to prevent repeated export
    unreal.ActorGroupingUtils().ungroup_actors([actor])

    editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    editor_actor_subsystem.set_selected_level_actors([actor])

    # Create export task
    task = unreal.AssetExportTask()
    task.object = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    task.filename = output_path
    task.selected = True
    task.replace_identical = False
    task.prompt = False
    task.automated = True
    task.options = fbx_options

    success = unreal.Exporter.run_asset_export_task(task)
    result_msg = "SUCCESS" if success else "FAILED"
    Logger.info(f"[{result_msg}] Exported actor to {output_path}")

    return success


# ==================== Actor Filtering ====================
def filter_actors(actors: List[unreal.Actor], config: ExportConfig) -> List[unreal.Actor]:
    """
    Filter valid actors based on exclusion rules.
    
    Args:
        actors: List of all actors in the scene
        config: Export configuration
        
    Returns:
        Filtered list of valid actors to export
    """
    # If target actors specified, only export those
    if config.target_actors:
        filtered = [a for a in actors if a.get_name() in config.target_actors]
        Logger.info(f"Using target actors filter: {len(filtered)} actors")
        return filtered

    filtered = []

    for actor in actors:
        # Skip invalid actors
        if actor.get_name() in config.invalid_actors:
            continue

        # Skip invalid actor classes
        if isinstance(actor, config.invalid_actor_classes):
            continue

        # Check for valid mesh components (or Landscape)
        components = actor.get_components_by_class(unreal.StaticMeshComponent)
        has_valid_mesh = components and any(comp.static_mesh for comp in components)

        if not has_valid_mesh and not isinstance(actor, unreal.Landscape):
            continue

        filtered.append(actor)

    Logger.info(f"Filtered {len(filtered)}/{len(actors)} valid actors")
    return filtered


# ==================== Scene Exporter ====================
class SceneExporter:
    """Exports actor info and static meshes from the Unreal Engine scene."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.export_dir = Path(self.config.export_dir)

        # Cache editor subsystems to avoid repeated lookups
        self._editor_actor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        self._editor_subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)

        # Get current map name
        map_name = self._editor_subsys.get_editor_world().get_path_name()
        self.map_name = map_name.split("/")[-1].split(".")[0]

        # Setup export directories
        self.scene_export_dir = self.export_dir / self.map_name
        self.bg_actors_dir = self.scene_export_dir / "bg_actors"
        self.fg_actors_dir = self.export_dir / "fg_actors"

        self.bg_actors_dir.mkdir(parents=True, exist_ok=True)
        self.fg_actors_dir.mkdir(parents=True, exist_ok=True)

        # Cache FBX export options (created once, reused for all exports)
        self._fbx_options = create_fbx_export_options(config)

        self.actors: List[unreal.Actor] = []

    def export_actor(self, actor: unreal.Actor) -> Dict[str, Any]:
        """
        Export all relevant information for an actor.
        
        Args:
            actor: Actor to export
            
        Returns:
            Dictionary containing complete actor information
        """
        # Special handling for InstancedFoliageActor
        if actor.get_class().get_name() == "InstancedFoliageActor":
            info = {
                "name": actor.get_name(),
                "label": actor.get_actor_label(),
                "ue_class": actor.get_class().get_name(),
                "components": []
            }

            components = actor.get_components_by_class()
            for comp in components:
                if isinstance(comp, unreal.StaticMeshComponent) and comp.static_mesh:
                    comp_info = self.export_ism_component(actor, comp)
                    info["components"].append(comp_info)

            return info

        # Regular actor export
        output_path = (self.bg_actors_dir / f"{actor.get_name()}.fbx").as_posix()
        success = export_actor_to_fbx(
            actor, output_path, self.config.skip_existing, self._fbx_options
        )

        if not success:
            Logger.warning(f"Failed to export actor: {actor.get_name()}")

        return {
            "name": actor.get_name(),
            "label": actor.get_actor_label(),
            "semantic_label": get_actor_semantic_label(actor, self.config.semantic_labels),
            "ue_class": actor.get_class().get_name(),
        }

    def export_ism_component(self, actor: unreal.Actor,
                             ism_component: unreal.InstancedStaticMeshComponent) -> Dict[str, Any]:
        """
        Export data from InstancedStaticMeshComponent (ISM).
        HierarchicalInstancedStaticMeshComponent (HISM) inherits from ISM.
        FoliageInstancedStaticMeshComponent (FISM) inherits from HISM.
        
        Args:
            actor: The parent actor of the ISM component
            ism_component: The ISM component to export
            
        Returns:
            Dictionary containing component data and instance transforms
        """
        static_mesh = ism_component.static_mesh
        asset_path = static_mesh.get_path_name()
        asset_name = static_mesh.get_name()

        output_path = (self.bg_actors_dir / f"{actor.get_name()}_{asset_name}.fbx").as_posix()
        export_static_mesh(
            static_mesh, output_path, self.config.skip_existing, self._fbx_options
        )

        # Collect all instance transforms
        instance_count = ism_component.get_instance_count()
        instances = [
            {
                "index": idx,
                "transform": transform_to_dict(ism_component.get_instance_transform(idx, world_space=True)),
            }
            for idx in range(instance_count)
        ]

        return {
            "asset_path": asset_path,
            "instance_count": instance_count,
            "semantic_label": get_label_by_mesh_path(asset_path, self.config.semantic_labels).name,
            "instances": instances
        }

    def parse_scene(self) -> None:
        """Parse and filter scene actors."""
        Logger.section("Parsing Scene Actors")

        actors = self._editor_actor_subsys.get_all_level_actors()

        # Check for duplicate names
        name_counts = Counter(actor.get_name() for actor in actors)
        duplicate_names = [name for name, count in name_counts.items() if count > 1]
        if duplicate_names:
            raise ValueError(f"Duplicate actor names found: {duplicate_names}. Please rename them.")

        self.actors = filter_actors(actors, self.config)

        # Log actor type statistics
        actor_types = Counter(actor.get_class().get_name() for actor in self.actors)
        Logger.section("Actor Type Statistics")
        for actor_type, count in sorted(actor_types.items()):
            Logger.info(f"  {actor_type}: {count}")

    def export_background(self) -> Path:
        """Export background scene actors"""
        Logger.section("Exporting Background Scene")

        self.parse_scene()

        scene_data = {
            "map_name": self.map_name,
            "exported_actors_dir": self.bg_actors_dir.as_posix(),
            "valid_actor_number": len(self.actors),
            "actors": [],
        }

        # Export each actor with progress tracking
        total = len(self.actors)
        for idx, actor in enumerate(self.actors, 1):
            if self.config.show_progress:
                Logger.progress(idx, total, actor.get_name())

            actor_info = self.export_actor(actor)
            scene_data["actors"].append(actor_info)

        # Save scene summary
        json_path = self.scene_export_dir / "exported_scene_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=4, ensure_ascii=False)

        Logger.section(f"Scene info saved to: {json_path}")
        return json_path

    def export_foreground(self) -> None:
        """Export foreground actors (vehicles and pedestrians)"""
        Logger.section("Exporting Foreground Actors")
        self.export_fg_vehicles()
        self.export_fg_peds()

    def export_fg_vehicles(self) -> None:
        """Export foreground vehicle meshes."""
        Logger.info("Exporting vehicle meshes...")

        total_vehicles = sum(len(v) for v in self.config.foreground_vehicles.values())
        current = 0

        for sem_label, vehicle_dict in self.config.foreground_vehicles.items():
            for carla_id, bp_path in vehicle_dict.items():
                current += 1

                export_path = self.fg_actors_dir / sem_label / f"{carla_id.replace('.', '_')}.fbx"

                if self.config.show_progress:
                    Logger.progress(current, total_vehicles, carla_id)

                if export_path.exists() and self.config.skip_existing:
                    Logger.info(f"Vehicle exists, skipping: {carla_id}")
                    continue

                export_path.parent.mkdir(parents=True, exist_ok=True)

                bp = unreal.load_asset(bp_path)
                spawned_vehicle = self._editor_actor_subsys.spawn_actor_from_object(
                    bp, location=[0, 0, 0]
                )
                export_actor_to_fbx(
                    spawned_vehicle, str(export_path),
                    skip_existing=False,  # already checked above
                    fbx_options=self._fbx_options,
                )
                self._editor_actor_subsys.destroy_actor(spawned_vehicle)

        Logger.info(f"Vehicle export completed: {current} vehicles")

    def export_fg_peds(self) -> None:
        """Export foreground pedestrian animations"""
        Logger.info("Exporting pedestrian animations...")

        asset_paths = unreal.EditorAssetLibrary.list_assets(
            self.config.fg_pedestrian_anim_path,
            recursive=True
        )
        name_frame_dict = {}
        exported_count = 0
        total_paths = len(asset_paths)

        for i, asset_path in enumerate(asset_paths, 1):
            anim_asset = unreal.EditorAssetLibrary.load_asset(asset_path)
            if not isinstance(anim_asset, unreal.AnimSequence):
                continue

            asset_name = os.path.basename(asset_path).split(".")[0]
            frame_num = int(anim_asset.sequence_length * self.config.fg_animation_fps)
            name_frame_dict[asset_name] = frame_num

            output_path = self.fg_actors_dir / "Pedestrians" / f"{asset_name}.fbx"

            if self.config.show_progress and i % 10 == 0:
                Logger.progress(i, total_paths, asset_name)

            if output_path.exists() and self.config.skip_existing:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_task = unreal.AssetExportTask()
            export_task.object = anim_asset
            export_task.filename = str(output_path)
            export_task.automated = True
            export_task.prompt = False
            export_task.replace_identical = True

            success = unreal.Exporter.run_asset_export_task(export_task)
            if success:
                exported_count += 1
                Logger.info(f"Exported animation: {asset_name}")

        # Save frame info
        frame_info_path = self.fg_actors_dir / "Pedestrians" / "frame_info.json"
        frame_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frame_info_path, "w", encoding="utf-8") as f:
            json.dump(name_frame_dict, f, indent=4, ensure_ascii=False)

        Logger.info(f"Pedestrian export completed: {exported_count} animations")
        Logger.info(f"Frame info saved to: {frame_info_path}")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config",
        "export_scene.json"
    )
    config = ExportConfig.from_json(config_path)

    exporter = SceneExporter(config)

    if config.export_foreground:
        exporter.export_foreground()

    if config.export_background:
        exporter.export_background()
