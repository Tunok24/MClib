import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Define simulation parameters
num_photons = 100
grid_size = (100, 100)
mesh_coarse = 10000
detector_x = 10
detector_sz = 10
step_sz = grid_size[0] / mesh_coarse

absorption_prob = 0.001
scattering_prob = 0.5

trajectories = []  # Store all photon paths

# Simulate photon transport
for i in range(num_photons):
    x, y = 0, 0
    path = [(x, y)]

    while x < detector_x:
        interaction = np.random.rand()
        if interaction < absorption_prob:
            break
        elif interaction < absorption_prob + scattering_prob:
            y += np.random.choice([-detector_sz/2, detector_sz/2]) * step_sz
            y = min(y, grid_size[1] - 1)
        x += step_sz
        path.append((x, y))
    
    trajectories.append(path)

# Plot trajectories
plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, num_photons))

# --- Draw material volume background ---
plt.gca().add_patch(
    plt.Rectangle(
        (0, 0),  # Bottom-left corner
        grid_size[0], grid_size[1],  # Width and height
        facecolor='lightgrey',
        edgecolor='black',
        alpha=0.4,
        label='Material Volume'
    )
)

for i, path in enumerate(trajectories):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color=colors[i], label=f'Photon {i+1}')

# Draw detector line
plt.axvline(x=detector_x, color='black', linestyle='--', label='Detector')

plt.title("Photon Trajectories in 2D Grid")
plt.xlabel("X (Grid Columns)")
plt.ylabel("Y (Grid Rows)")
# plt.legend(loc='upper left', fontsize=8)
plt.grid(True)
plt.xlim(0, detector_x + 1)
plt.ylim(-detector_sz/2, detector_sz/2)

# Save the figure
output_path = "./fig/photon_trajectories.jpg"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Trajectory image saved as: {os.path.abspath(output_path)}")
