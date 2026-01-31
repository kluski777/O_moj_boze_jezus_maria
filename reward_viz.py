import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from importlib import reload
from constants import *
import approximation

reload(approximation)

class MockCar:
        def __init__(self, vel):
                self.vel = vel
                self.to_plot_dict = defaultdict(list)

res_v=50
res_d=50
actions = ['forward', 'backward', 'stop', 'left', 'right']
vels = np.linspace(0, 8.0, res_v)
dists = np.linspace(0.01, 1.2, res_d)
policy_grid = np.zeros((res_d, res_v))
reward_grid = np.zeros((res_d, res_v))

for i, d in enumerate(dists):
        for j, v in enumerate(vels):
                car = MockCar(v)
                state = [
                        np.array([0.155, 0.24 , 1.44 , 0.27 , 0.205]), 
                        np.array([1.  , 1.  , 0.19, 1.  , 0.55]), 
                        np.float64(0.9780952801854349), 
                        -0.06249999999999998
                ]
                cos = 1 - 2 * (np.sign(state[2]) * abs(state[2])**(1/3))**2
                rewards = [approximation.action_rewards(state, a, cos, car, show=False) for a in actions]
                policy_grid[i, j], reward_grid[i, j] = np.argmax(rewards), np.max(rewards)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot - actions
n_actions = len(actions)
cmap = plt.get_cmap('viridis', n_actions)
im1 = ax1.imshow(policy_grid, origin='lower', aspect='auto', 
        extent=[vels[0], vels[-1], dists[0], dists[-1]],
        cmap=cmap, vmin=-0.5, vmax=n_actions - 0.5)
ax1.set_xlabel('Velocity', fontsize=12)
ax1.set_ylabel('Front Distance', fontsize=12)
ax1.set_title('Best Action', fontsize=14)
cbar1 = plt.colorbar(im1, ax=ax1, ticks=range(n_actions))
cbar1.ax.set_yticklabels(actions)

# Right plot - reward values
im2 = ax2.imshow(reward_grid, origin='lower', aspect='auto',
        extent=[vels[0], vels[-1], dists[0], dists[-1]],
        cmap='coolwarm')
ax2.set_xlabel('Velocity', fontsize=12)
ax2.set_ylabel('Front Distance', fontsize=12)
ax2.set_title('Reward Values', fontsize=14)
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()