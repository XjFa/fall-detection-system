# animation/stick_figure.py
import matplotlib.pyplot as plt
import numpy as np

SENSOR_LABELS = {
    "head": "HEAD",
    "chest": "CHEST",
    "belt": "BELT",
    "ankle_l": "ANKLE_LEFT",
    "ankle_r": "ANKLE_RIGHT"
}

def draw_stick_figure_3d(ax, head_pos, chest_pos, belt_pos, ankle_l_pos, ankle_r_pos, fall_prob):
    """
    Draw a stick figure using absolute joint positions.
    head_pos, chest_pos, belt_pos, ankle_l_pos, ankle_r_pos are 3D coordinates (x, y, z).
    Applies fall pose if fall_prob > 0.7.
    """
    ax.clear()

    # Copy positions
    head = np.array(head_pos)
    chest = np.array(chest_pos)
    belt = np.array(belt_pos)
    ankle_l = np.array(ankle_l_pos)
    ankle_r = np.array(ankle_r_pos)

    # Apply fall pose if necessary
    threshold = 0.7
    if fall_prob > threshold:
        collapse = np.clip((fall_prob - threshold) / (1 - threshold), 0, 1)
        chest[2] -= 0.6 * collapse
        head[2] = chest[2] + (head[2] - chest[2])
        chest[1] += 0.15 * collapse
        head[1] += 0.2 * collapse
        ankle_l[0] -= 0.1 * collapse
        ankle_r[0] += 0.1 * collapse

    # Color
    color = "red" if fall_prob > threshold else "steelblue"

    # Draw limbs
    ax.plot([head[0], chest[0]], [head[1], chest[1]], [head[2], chest[2]], lw=3, c=color)
    ax.plot([chest[0], belt[0]], [chest[1], belt[1]], [chest[2], belt[2]], lw=3, c=color)
    ax.plot([belt[0], ankle_l[0]], [belt[1], ankle_l[1]], [belt[2], ankle_l[2]], lw=3, c=color)
    ax.plot([belt[0], ankle_r[0]], [belt[1], ankle_r[1]], [belt[2], ankle_r[2]], lw=3, c=color)

    # Draw joints
    joints = {
        "head": head,
        "chest": chest,
        "belt": belt,
        "ankle_l": ankle_l,
        "ankle_r": ankle_r
    }
    for name, pos in joints.items():
        ax.scatter(pos[0], pos[1], pos[2], s=80, c=color)
        ax.text(pos[0], pos[1], pos[2]+0.05, SENSOR_LABELS.get(name, name),
                fontsize=8, color="black")

    # Axes settings
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 2)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Fall Probability: {fall_prob:.2f}")
