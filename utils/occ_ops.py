import numpy as np

from utils.labels import lut_sem_prior


def occ_sparse2dense(sparse_data,
                     empty_value: int = 0,
                     volume_size=(200, 200, 16)
                     ):
    """
    Convert sparse semantic occupancy data to dense voxel grid

    Args:
        sparse_data: [N, 4] array
        empty_value: Value for empty voxels (default: 0)
        volume_size: Dimensions of output voxel grid

    Returns:
        Dense voxel grid [X,Y,Z] with optional semantics
    """
    sparse = np.asarray(sparse_data)
    assert sparse.ndim == 2 and sparse.shape[1] == 4, \
        f"sparse_data must be [N,4], got shape {sparse.shape}"

    dense_grid = np.full(volume_size, empty_value, dtype=sparse.dtype)

    coords = sparse[:, :3]
    dense_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse[:, 3]
    return dense_grid


def occ_dense2sparse_smallest(dense_data: np.ndarray, empty_value: int = 0) -> np.ndarray:
    """
    Convert dense semantic occupancy grid to sparse.

    Args:
        dense_data: [X,Y,Z] semantic occupancy grid
        empty_value: Value considered as empty (default: 0)

    Returns:
        Sparse array [N, 4] (coordinates + semantics)
    """
    assert dense_data.ndim == 3, "Dense data must be a 3D semantic occupancy grid"

    # Get occupied positions
    mask = dense_data != empty_value

    # Use Fortran-style for better semantic compression
    flat_idx = np.nonzero(np.ravel(mask, order='F'))[0]
    if flat_idx.size == 0:
        return np.empty((0, 4), dtype=dense_data.dtype)

    X, Y, Z = dense_data.shape

    # F-order: flat_id=z⋅(Y⋅X)+y⋅X+x
    xy = X * Y
    z = flat_idx // xy
    r = flat_idx - z * xy
    y = r // X
    x = r - y * X
    sem = dense_data.ravel(order='F')[flat_idx]

    # C-order: flat_id=x⋅(Y⋅Z)+y⋅Z+z
    # xy = Y * Z
    # x = flat_idx // xy
    # r = flat_idx - x * xy
    # y = r // Z
    # z = r - y * Z
    # sem = dense_data.reshape(-1)[flat_idx]  # equal to: sem = dense_data.ravel(order='C')[flat_idx]

    out = np.empty((flat_idx.size, 4), dtype=dense_data.dtype, order='F')
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    out[:, 3] = sem
    return out


def occ_dense2sparse_fast(dense_data: np.ndarray, empty_value: int = 0) -> np.ndarray:
    """
    Convert dense semantic occupancy grid to sparse.

    Args:
        dense_data: [X,Y,Z] semantic occupancy grid
        empty_value: Value considered as empty (default: 0)

    Returns:
        Sparse array [N, 4] (coordinates + semantics)
    """
    assert dense_data.ndim == 3, "Dense data must be a 3D semantic occupancy grid"

    # Get occupied positions
    mask = dense_data != empty_value
    flat_idx = np.flatnonzero(mask)
    if flat_idx.size == 0:
        return np.empty((0, 4), dtype=dense_data.dtype)

    X, Y, Z = dense_data.shape

    # C-order: flat_id=x⋅(Y⋅Z)+y⋅Z+z
    x, y, z = np.unravel_index(flat_idx, (X, Y, Z), order='C')  # equal to the following:
    # xy = Y * Z
    # x = flat_idx // xy
    # r = flat_idx - x * xy
    # y = r // Z
    # z = r - y * Z

    sem = dense_data.reshape(-1)[flat_idx]

    out = np.empty((flat_idx.size, 4), dtype=dense_data.dtype, order='F')  # Use Fortran-style for better compression
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    out[:, 3] = sem
    return out


def occ_dense2sparse(dense_data,
                     empty_value: int = 0
                     ):
    """
    Convert dense voxel grid to sparse representation

    Args:
        dense_data: [X,Y,Z] voxel grid with occupancy values
        empty_value: Value considered as empty (default: 0)

    Returns:
        Sparse array [N, 3] or [N, 4] (coordinates or coordinates+semantics)
    """
    assert dense_data.ndim == 3, "Dense data must be a 3D array"

    # Get occupied indices
    occupied_mask = dense_data != empty_value
    indices = np.array(np.where(occupied_mask)).T

    semantics = dense_data[occupied_mask]
    sparse_data = np.hstack((indices, semantics[:, None]))

    return sparse_data


def occ_downsample(dense_pano_occ: np.ndarray,
                   factor: int = 2,
                   use_sem_prior: bool = True) -> np.ndarray:
    """
    Downsample a dense panoptic occupancy grid with semantic priority selection.

    Parameters
    ----------
    dense_pano_occ : np.ndarray
        Input dense grid [X, Y, Z] (dtype usually uint16) encoding
        semantic (high 5 bits) + instance id (low 11 bits).
    factor : int, optional
        Downsampling factor for each axis (default 2).
    use_sem_prior : bool, optional
        If False, fall back to simple nearest-neighbor (first voxel per block).

    Returns
    -------
    np.ndarray
        Downsampled dense grid with the same dtype as input.
    """
    dense = np.asarray(dense_pano_occ)
    if dense.ndim != 3:
        raise ValueError(f"dense_pano_occ must be 3D [X, Y, Z], got shape {dense.shape}")

    X, Y, Z = dense.shape

    if X % factor != 0 or Y % factor != 0 or Z % factor != 0:
        raise ValueError(
            f"All dimensions must be divisible by factor={factor}, "
            f"got shape {dense.shape}"
        )

    if not use_sem_prior:
        return dense[::factor, ::factor, ::factor].astype(dense.dtype, copy=False)

    x2, y2, z2 = X // factor, Y // factor, Z // factor

    # 1) Reshape [X,Y,Z] to [x2, factor, y2, factor, z2, factor],
    #    then transpose to [x2, y2, z2, factor, factor, factor] so each block maps to one coarse voxel
    dense_blocks = dense.reshape(x2, factor, y2, factor, z2, factor) \
        .transpose(0, 2, 4, 1, 3, 5)  # (x2, y2, z2, f, f, f)

    # 2) Extract semantic ID (high 5 bits) from panoptic ID and build priority map
    sem_blocks = (dense_blocks >> np.uint16(11)).astype(np.int32)  # (x2, y2, z2, f, f, f)
    pri_blocks = lut_sem_prior[sem_blocks]  # (x2, y2, z2, f, f, f)

    # 3) Within each block, find the voxel with the highest semantic priority
    pri_flat = pri_blocks.reshape(x2, y2, z2, -1)  # (x2, y2, z2, f^3)
    dense_flat = dense_blocks.reshape(x2, y2, z2, -1)  # (x2, y2, z2, f^3)

    max_idx = pri_flat.argmax(axis=-1)[..., None]  # (x2, y2, z2, 1)
    downsampled = np.take_along_axis(dense_flat, max_idx, axis=-1).squeeze(-1)

    return downsampled.astype(dense.dtype, copy=False)


def occ_downsample_sparse(sparse_data: np.ndarray,
                          factor: int = 2,
                          voxel_range=(768, 512, 130),
                          use_sem_prior: bool = True) -> np.ndarray:
    """
    Downsample a sparse voxel representation directly (recommended for pipelines).

    Args:
        sparse_data (np.ndarray): [N, 4]; last column is panoptic ID (uint16).
            Panoptic ID layout:
                - high 5 bits : semantic id (0-31)
                - low 11 bits : instance id (0-2047)
        factor (int): Downsampling factor.
        voxel_range (tuple): Original dense grid size (X, Y, Z).
        use_sem_prior (bool): Whether to select by semantic priority.

    Returns:
        np.ndarray: [M, 4] Sparse downsampled voxels with coords and panoptic IDs.
    """
    sparse = np.asarray(sparse_data)
    assert sparse.ndim == 2 and sparse.shape[1] == 4, \
        f"sparse_data must be [N, 4], got shape {sparse.shape}"

    coords = sparse[:, :3]
    values = sparse[:, 3].astype(np.uint16, copy=False)

    # 1) Pad voxel range so each dimension is divisible by factor
    voxel_range = list(voxel_range)
    for i in range(3):
        dim = voxel_range[i]
        r = dim % factor
        if r != 0:
            pad = factor - r
            voxel_range[i] = dim + pad
    voxel_range = tuple(voxel_range)

    X, Y, Z = voxel_range
    cX, cY, cZ = X // factor, Y // factor, Z // factor

    # 2) Downsample coordinates: coords // factor
    coarse_coords = coords // factor

    # Clip to valid range
    valid = (
            (0 <= coarse_coords[:, 0]) & (coarse_coords[:, 0] < cX) &
            (0 <= coarse_coords[:, 1]) & (coarse_coords[:, 1] < cY) &
            (0 <= coarse_coords[:, 2]) & (coarse_coords[:, 2] < cZ)
    )
    coarse_coords = coarse_coords[valid]
    values = values[valid]

    if coarse_coords.size == 0:
        return np.empty((0, 4), dtype=np.uint16)

    # 3) Compute semantic priority
    if use_sem_prior:
        sem_ids = (values >> np.uint16(11)).astype(np.uint16)  # high 5 bits
        pri = lut_sem_prior[sem_ids]
    else:
        pri = np.zeros(values.shape[0], dtype=np.uint16)

    # 4) Map (x,y,z) to 1D linear index for grouping
    coarse_coords_cp = coarse_coords.astype(np.int32)
    linear_idx = (
            coarse_coords_cp[:, 0] * (cY * cZ) +
            coarse_coords_cp[:, 1] * cZ +
            coarse_coords_cp[:, 2]
    )

    # 5) Sort by (linear_idx, priority); within each coarse voxel, highest priority comes last
    order = np.lexsort((pri, linear_idx))
    linear_sorted = linear_idx[order]
    coords_sorted = coarse_coords[order]
    values_sorted = values[order]

    # 6) Keep only the last entry per group (highest priority)
    keep = np.ones(linear_sorted.shape[0], dtype=bool)
    keep[:-1] = linear_sorted[1:] != linear_sorted[:-1]

    coords_unique = coords_sorted[keep]
    values_unique = values_sorted[keep]

    return np.column_stack((coords_unique.astype(np.uint16),
                            values_unique.astype(np.uint16)))


def occ_resample_sparse(
        occ_sparse: np.ndarray,
        src_voxel_size: float,
        src_voxel_origin: np.ndarray,
        tgt_voxel_size: float,
        tgt_voxel_origin: np.ndarray,
        tgt_volume_size: np.ndarray,
        use_sem_prior: bool = True
) -> np.ndarray:
    """
    Sparse occupancy resampling:
    downsample + crop + reindex + semantic resolve (all-in-one).

    Args:
        occ_sparse: [N, 4] sparse occupancy data with panoptic IDs
        src_voxel_size: Source voxel size in meters
        src_voxel_origin: Source voxel origin in LiDAR coordinate [x, y, z]
        tgt_voxel_size: Target voxel size in meters
        tgt_voxel_origin: Target voxel origin in LiDAR coordinate [x, y, z]
        tgt_volume_size: Target volume size [X, Y, Z]
        use_sem_prior: Whether to use semantic priority for conflict voxels

    Returns:
        [M, 4] Resampled sparse occupancy data
    """
    assert occ_sparse.ndim == 2 and occ_sparse.shape[1] == 4, "Occupancy data must be in shape of [N, 4]."

    coords = occ_sparse[:, :3]
    values = occ_sparse[:, 3]

    # Convert source voxel coordinates to world points
    offset = src_voxel_origin - tgt_voxel_origin
    pts = (coords.astype(np.float32) + 0.5) * float(src_voxel_size) + offset[None, :]

    # Crop to target volume
    tgt_voxel_range = np.array(tgt_volume_size) * tgt_voxel_size
    valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < tgt_voxel_range[0]) &
            (pts[:, 1] >= 0) & (pts[:, 1] < tgt_voxel_range[1]) &
            (pts[:, 2] >= 0) & (pts[:, 2] < tgt_voxel_range[2])
    )

    pts = pts[valid]
    values = values[valid]

    if pts.size == 0:
        # Empty result
        return np.empty((0, 4), dtype=np.uint16)

    # Convert world points to target voxel coordinates
    tgt_coords = np.floor(pts / tgt_voxel_size).astype(np.uint32)

    # Compute semantic priority
    if use_sem_prior:
        sem_ids = (values >> 11).astype(np.uint16)
        sem_prior = lut_sem_prior[sem_ids]
    else:
        sem_prior = np.zeros_like(values, dtype=np.uint16)

    # Group by voxel and resolve conflicts
    # Map (x, y, z) to 1D linear index for grouping
    X, Y, Z = tgt_volume_size
    linear_idx = (
            tgt_coords[:, 0] * (Y * Z) +
            tgt_coords[:, 1] * Z +
            tgt_coords[:, 2]
    )

    # Sort by linear_idx (primary) and semantic priority (secondary)
    # lexsort: keys[-1] is primary key, so (sem_prior, linear_idx) sorts by linear_idx first, then sem_prior
    order = np.lexsort((sem_prior, linear_idx))
    linear_sorted = linear_idx[order]
    coords_sorted = tgt_coords[order]
    values_sorted = values[order]

    # Keep only the last entry for each linear_idx (highest priority)
    keep = np.ones(len(order), dtype=bool)
    keep[:-1] = linear_sorted[1:] != linear_sorted[:-1]

    # Build output array with Fortran-style for better compression
    coords_kept = coords_sorted[keep].astype(np.uint16)
    values_kept = values_sorted[keep].astype(np.uint16)

    out = np.empty((len(coords_kept), 4), dtype=np.uint16, order='F')
    out[:, :3] = coords_kept
    out[:, 3] = values_kept
    return out
