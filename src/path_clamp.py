#######################################
## PATH CLAMP & HELPER FUNCTIONS     ##
#######################################
import numpy as np

def compute_vectors_with_angle_clamp(raw_action: np.ndarray, max_diff_deg: float = 10.0) -> np.ndarray:
    """Convert raw action to path vectors with angle constraints"""
    vectors = raw_action.reshape(8, 2)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    
    clamped = np.zeros_like(vectors)
    clamped[0] = [1, 0]
    prev_angle = 0.0
    
    for i in range(1, 8):
        desired_angle = np.arctan2(vectors[i,1], vectors[i,0])
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)
        clamped[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle
        
    return clamped

def clamp_vector_angle_diff(prev_angle: float, desired_angle: float,
                          max_diff_deg: float) -> float:
    """Clamp angle difference between consecutive path segments"""
    max_diff_rad = np.deg2rad(max_diff_deg)
    angle_diff = (desired_angle - prev_angle + np.pi) % (2*np.pi) - np.pi
    return prev_angle + np.clip(angle_diff, -max_diff_rad, max_diff_rad)