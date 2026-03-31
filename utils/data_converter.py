from typing import Union, Dict, List

import numpy as np
import trimesh


def decode_combined_id(global_id: Union[np.uint16, np.ndarray]):
    """
    Decode combined ID into semantic ID and instance ID. Supports scalar or ndarray.

    The combined ID is a 16-bit integer where the first 5 bits store the semantic ID (0-31),
    and the last 11 bits store the instance ID (0-2047).
    
    Args:
        global_id: uint16 or ndarray of uint16

    Returns:
        tuple: (semantic_id, instance_id) with same shape
    """
    global_id = np.asarray(global_id, dtype=np.uint16)

    semantic_id = (global_id >> 11) & 0x1F  # Extract high 5 bits, mask 0x1F = 31 = 0b11111
    instance_id = global_id & 0x7FF  # Extract low 11 bits, mask 0x7FF = 2047 = 0b11111111111
    return semantic_id, instance_id


def encode_combined_id(semantic_id: Union[np.uint16, np.ndarray],
                       instance_id: Union[np.uint16, np.ndarray]):
    """
    Encode semantic ID and instance ID into a combined ID. Supports scalar or ndarray.
    
    Combines semantic ID (5 bits) and instance ID (11 bits) into a single 16-bit integer.
    
    Args:
        semantic_id: uint16 or ndarray of uint16 (0-31)
        instance_id:  uint16 or ndarray of uint16 (0-2047)
        
    Returns:
        global_id: uint16 or ndarray of uint16
    """
    semantic_id = np.asarray(semantic_id, dtype=np.uint16)
    instance_id = np.asarray(instance_id, dtype=np.uint16)
    if np.any(semantic_id > 31):
        raise ValueError("Semantic ID must be in range [0, 31]")
    if np.any(instance_id > 2047):
        raise ValueError("Instance ID must be in range [0, 2047]")

    global_id = (semantic_id << 11) | instance_id
    return global_id.astype(np.uint16)


def get_fg_instance_id(fg_actor_id):
    """
    Remap foreground instance ID obtained in CARLA to avoid collision with background IDs.
    Args:
        fg_actor_id: foreground actor ID from CARLA, typically in range [1000, +inf)

    Returns:
        Remapped instance ID in range [1000, 1999]

    """
    return fg_actor_id % 1000 + 1000


def mesh_pano_to_sem(
        pano_mesh: Dict[int, Union[trimesh.Trimesh, List[tuple]]]
) -> Dict[int, trimesh.Trimesh]:
    sem_mesh = {}
    for sem_id, mesh_data in pano_mesh.items():
        if isinstance(mesh_data, list):
            # Instance class: merge all instances into one mesh
            meshes = [mesh for instance_id, mesh in mesh_data if mesh is not None]
            if len(meshes) == 0:
                continue
            sem_mesh[sem_id] = trimesh.util.concatenate(meshes)
        else:
            # Stuff class: single mesh
            if mesh_data is not None:
                sem_mesh[sem_id] = mesh_data
    return sem_mesh


def occ_pano_to_sem(occ: np.ndarray) -> np.ndarray:
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] == 4), \
        "Occupancy grid must be in shape of XxYxZ or Nx4"
    if occ.ndim == 3:
        # Dense occ grid
        semantic_id, _ = decode_combined_id(occ.astype(np.uint16))
        return semantic_id

    # Sparse occ grid
    semantic_id, _ = decode_combined_id(occ[:, 3].astype(np.uint16))
    semantic_id = semantic_id.astype(np.uint16).reshape(-1, 1)

    return np.hstack([occ[:, :3], semantic_id])


if __name__ == "__main__":
    print("===== Test decode/encode =====")
    gid = np.uint16(0x2A05)
    sem, inst = decode_combined_id(gid)
    print("decode:", gid, "-> sem:", sem, "inst:", inst)

    gid2 = encode_combined_id(sem, inst)
    print("encode:", sem, inst, "->", gid2)

    print("\n===== Test vectorized decode =====")
    arr = np.array([0x2A05, 0x1234, 0xFFFF], dtype=np.uint16)
    sem_v, inst_v = decode_combined_id(arr)
    print("input :", arr)
    print("semantic:", sem_v)
    print("instance:", inst_v)

    print("\n===== Test occ_pano_to_sem (Sparse Nx4) =====")
    # mock occ Nx4
    occ = np.array([
        [1, 2, 3, 0x2A05],
        [4, 5, 6, 0x1F80],
        [7, 8, 9, 0xFFFF]
    ], dtype=np.uint16)

    sem_occ = occ_pano_to_sem(occ)
    print("occ input:\n", occ)
    print("sem output:\n", sem_occ)

    print("\n===== Test occ_pano_to_sem (Dense X×Y×Z) =====")
    # Create a 3×3×3 dense occupancy grid with known combined IDs
    dense_occ = np.zeros((3, 3, 3), dtype=np.uint16)

    # Fill with some combined IDs for testing
    dense_occ[0, 0, 0] = 0x2A05  # semantic 5
    dense_occ[1, 1, 1] = 0x1F80  # semantic 31
    dense_occ[2, 2, 2] = 0xFFFF  # semantic 31 (0xFFFF >> 11 = 0b11111)

    print("dense grid input:\n", dense_occ)

    dense_sem = occ_pano_to_sem(dense_occ)
    print("dense semantic output:\n", dense_sem)

    # Check expected
    print("Check (0,0,0): expected 5  -> got", dense_sem[0, 0, 0])
    print("Check (1,1,1): expected 31 -> got", dense_sem[1, 1, 1])
    print("Check (2,2,2): expected 31 -> got", dense_sem[2, 2, 2])
