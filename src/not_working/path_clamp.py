#######################################
## PATH CLAMP & HELPER FUNCTIONS     ##
#######################################
import numpy as np

def compute_vectors_with_angle_clamp(raw_action: np.ndarray, max_diff_deg: float = 10.0) -> np.ndarray:
    """
    Convert raw_action (16 floats) -> 8 path segments (x,y),
    each normalized to length 1, while clamping the turn angle 
    between consecutive segments to ±max_diff_deg.
    """

    # Reshape from (16,) to (8,2)
    vectors = raw_action.reshape(8, 2)

    # Normalize each segment so its length is 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms

    clamped = np.zeros_like(vectors)

    # Let the first segment be exactly what the agent predicted (direction is agent's choice)
    first_angle = np.arctan2(vectors[0, 1], vectors[0, 0])
    clamped[0] = vectors[0]
    prev_angle = first_angle

    # For each subsequent segment, clamp the angle relative to the previous one
    for i in range(1, 8):
        desired_angle = np.arctan2(vectors[i, 1], vectors[i, 0])
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)

        # Because we're enforcing length=1, the segment is just (cos, sin) of the clamped angle
        clamped[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle

    return clamped



def clamp_vector_angle_diff(prev_angle: float, desired_angle: float,
                          max_diff_deg: float) -> float:
    """Clamp angle difference between consecutive path segments"""
    max_diff_rad = np.deg2rad(max_diff_deg)
    angle_diff = (desired_angle - prev_angle + np.pi) % (2*np.pi) - np.pi
    return prev_angle + np.clip(angle_diff, -max_diff_rad, max_diff_rad)