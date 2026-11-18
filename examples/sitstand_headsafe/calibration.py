"""Explicit external calibration loading; no device files are read on import."""
from pathlib import Path
import math
import tomllib
NAMES=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
SIT={1:0.,2:-.4079,3:1.35,4:0.,10:0.,11:.4079,12:-1.35,13:0.}

def load_calibration(path):
    rows=tomllib.loads(Path(path).read_text(encoding='utf-8'))['joints']
    by_name={row['name']:row for row in rows}
    if len(by_name)!=len(rows):raise ValueError('Duplicate calibration joint names')
    selected=[by_name[name] for name in NAMES]
    for row in selected:
        if not all(isinstance(row[k],(int,float)) and not isinstance(row[k],bool) and math.isfinite(row[k]) for k in ('min_rad','max_rad','zero_raw','direction')):
            raise ValueError('Calibration values must be finite numbers')
        if row['min_rad']>row['max_rad'] or row['direction'] not in (-1,1):raise ValueError('Invalid calibration bounds or direction')
    return selected
