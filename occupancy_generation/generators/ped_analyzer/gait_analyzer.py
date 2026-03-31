import json
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from scipy.signal import find_peaks


class PedGaitMatcher:
    """Pedestrian gait phase matcher.

    Matches observed pedestrian motion to a reference gait cycle database
    to determine the current animation frame.
    """

    def __init__(self, data_dir: Path, gait_cfg: Dict):
        """
        Args:
            data_dir: Data directory path
            gait_cfg: Gait configuration parameters
        """
        self.data_dir = Path(data_dir)

        # Configuration
        self.ped_dir = Path(gait_cfg.get('ped_dir', ''))
        self.gait_num_frames = gait_cfg.get('num_frames', 25)
        self.gait_speed_threshold = gait_cfg.get('speed_threshold', 0.1)
        self.seq_total_frames = gait_cfg.get('total_frames', 1000)
        self.database_frames = gait_cfg.get('database_frames', 36)
        self.target_bones = gait_cfg.get('target_bones', ['crl_leg__L', 'crl_leg__R'])

        self._traffic_info_cache = {}

        # Load and analyze reference gait database
        self.db_bones = {}
        self._init_database()

    def _init_database(self):
        """Load bone data for each frame in the reference gait database."""
        for frame_idx in range(self.database_frames):
            bones_path = self.ped_dir / f"frame_{frame_idx:02d}_bones.json"
            with open(bones_path, 'r') as f:
                bones_json = json.load(f)

            target_bones_data = {name: bones_json[name] for name in self.target_bones}
            self.db_bones[frame_idx] = {
                'bones': target_bones_data,
                'meta_info': self._extract_meta_info(target_bones_data, scale=100.0),
            }

    def _extract_meta_info(self, bone_data: Dict, scale: float = 1.0) -> Dict:
        """Extract positional and rotational meta information from bone transforms.

        Args:
            bone_data: Dictionary mapping bone names to 4x4 transform matrices
            scale: Scale factor for positions (e.g., 100.0 to convert cm to m)

        Returns:
            Dictionary with bone positions, rotation angle, and positional differences
        """
        if len(bone_data) < 2:
            return {
                f'{self.target_bones[0]}_pos': np.zeros(3),
                f'{self.target_bones[1]}_pos': np.zeros(3),
                'rot_angle': 0.0,
                'distance': 0.0,
                'x_difference': 0.0,
                'y_difference': 0.0,
                'z_difference': 0.0,
            }

        bone_name0, bone_name1 = bone_data.keys()
        bone0_transform = np.array(bone_data[bone_name0])
        bone1_transform = np.array(bone_data[bone_name1])

        bone0_loc = bone0_transform[:3, 3] / scale
        bone1_loc = bone1_transform[:3, 3] / scale

        # Rotation angle between the two bones
        cos_angle = (np.trace(bone0_transform[:3, :3].T @ bone1_transform[:3, :3]) - 1) / 2
        rot_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return {
            f'{bone_name0}_pos': bone0_loc,
            f'{bone_name1}_pos': bone1_loc,
            'rot_angle': rot_angle,
            'distance': np.linalg.norm(bone0_loc - bone1_loc),
            'x_difference': bone0_loc[0] - bone1_loc[0],
            'y_difference': bone0_loc[1] - bone1_loc[1],
            'z_difference': bone0_loc[2] - bone1_loc[2],
        }

    def load_traffic_info(self, frame_id: int) -> Dict:
        """Load traffic info with caching.

        Args:
            frame_id: Frame ID

        Returns:
            Traffic information dictionary
        """
        if frame_id not in self._traffic_info_cache:
            path = self.data_dir / "traffic_info" / f"{frame_id:04d}.yaml"
            with open(path) as f:
                self._traffic_info_cache[frame_id] = yaml.safe_load(f)
        return self._traffic_info_cache[frame_id]

    def _get_frame_range(self, current_frame: int) -> tuple:
        """Calculate the observation window around the current frame.

        Args:
            current_frame: Current frame index

        Returns:
            (start_frame, end_frame) inclusive
        """
        half = self.gait_num_frames // 2
        total = self.seq_total_frames

        if current_frame < half:
            return 0, self.gait_num_frames - 1
        elif current_frame + half > total - 1:
            return total - self.gait_num_frames, total - 1
        else:
            return current_frame - half, current_frame + half

    def analyze_pedestrian_gait(self, pedestrian_id: int, current_frame: int) -> int:
        """Determine the best-matching database gait frame for a pedestrian.

        Args:
            pedestrian_id: Pedestrian actor ID
            current_frame: Current simulation frame

        Returns:
            Matched database frame index (0 to database_frames-1)
        """
        start_frame, end_frame = self._get_frame_range(current_frame)

        # Collect pedestrian bone data over the observation window
        ped_data = {}
        for frame_idx in range(start_frame, end_frame + 1):
            traffic_info = self.load_traffic_info(frame_idx)
            if not traffic_info:
                continue
            for ped in traffic_info['pedestrians']:
                if ped['id'] == pedestrian_id:
                    if ped['bones'] and len(ped['bones']) >= 2:
                        ped_data[frame_idx] = {
                            'bones': ped['bones'],
                            'transform': np.array(ped['transform']),
                            'meta_info': self._extract_meta_info(ped['bones']),
                        }
                    break

        if not ped_data:
            return self.database_frames // 2  # Default to mid-cycle

        current_idx = current_frame - start_frame
        return self._match_phase(ped_data, current_idx)

    def _match_phase(self, ped_data: Dict, current_frame_idx: int) -> int:
        """Align observed gait signal to the reference gait cycle using phase correlation.

        The method aligns the observed x-direction leg-distance signal with
        the reference z-direction gait cycle from the database, and determines
        which frame in the gait cycle corresponds to the current observation.

        Args:
            ped_data: Pedestrian observation data (short sequence around current frame)
            current_frame_idx: Index of the current frame within the observation window

        Returns:
            Matched database frame index (0 to database_frames-1)
        """
        # Step 1: Extract signals
        # Reference: full gait cycle from database (z-direction)
        z_ref = np.array([a['meta_info']['z_difference'] for a in self.db_bones.values()])
        # Observation: current window (x-direction)
        x_obs = np.array([a['meta_info']['x_difference'] for a in ped_data.values()])

        N_ref = len(z_ref)
        N_obs = len(x_obs)

        # Step 2: Normalize signals
        def normalize(s):
            s = np.asarray(s, dtype=np.float32)
            s -= np.mean(s)
            s /= (np.std(s) + 1e-8)
            return s

        z_norm = normalize(z_ref)
        x_norm = normalize(x_obs)

        # Step 3: Detect extrema
        peak_z, _ = find_peaks(z_norm)
        valley_z, _ = find_peaks(-z_norm)
        peak_x, _ = find_peaks(x_norm)
        valley_x, _ = find_peaks(-x_norm)

        # Guard: if reference signal has no clear extrema, fall back to linear ratio
        if len(peak_z) == 0 or len(valley_z) == 0:
            phase_ratio = (current_frame_idx % N_obs) / N_obs
            return int(round(phase_ratio * N_ref)) % N_ref

        # Step 4: Determine reference half-period boundaries
        if peak_z[0] < valley_z[0]:
            start_ref, end_ref = peak_z[0], valley_z[0]
        else:
            start_ref, end_ref = valley_z[0], peak_z[0]

        half_period_ref = end_ref - start_ref
        full_period_ref = N_ref

        # Step 5: Collect and sort observation extrema
        all_extrema = sorted(
            [(i, 'peak') for i in peak_x] + [(i, 'valley') for i in valley_x],
            key=lambda x: x[0]
        )

        if len(all_extrema) < 2:
            # Not enough extrema for alignment — fall back to simple ratio
            phase_ratio = (current_frame_idx % N_obs) / N_obs
            return int(round(phase_ratio * N_ref)) % N_ref

        # Step 6: Find the half-period containing the current frame
        prev_extrema, next_extrema = all_extrema[0], all_extrema[1]
        for i in range(len(all_extrema) - 1):
            if all_extrema[i][0] <= current_frame_idx <= all_extrema[i + 1][0]:
                prev_extrema, next_extrema = all_extrema[i], all_extrema[i + 1]
                break

        # Step 7: Compute local phase ratio within the observation half-period
        half_period_obs = max(next_extrema[0] - prev_extrema[0], 1)
        local_phase_ratio = np.clip(
            (current_frame_idx - prev_extrema[0]) / half_period_obs, 0.0, 1.0
        )

        # Step 8: Map to the reference database half-period
        if prev_extrema[1] == 'peak' and next_extrema[1] == 'valley':
            # Peak-to-valley in observation → peak-to-valley in reference
            mapped_frame = start_ref + local_phase_ratio * half_period_ref
        elif prev_extrema[1] == 'valley' and next_extrema[1] == 'peak':
            # Valley-to-peak in observation → valley-to-peak in reference
            mapped_frame = end_ref + local_phase_ratio * half_period_ref
        else:
            # Same-type extrema (edge case) — linear mapping
            mapped_frame = local_phase_ratio * full_period_ref

        # Step 9: Clamp and return
        return int(N_ref - round(mapped_frame)) % N_ref
