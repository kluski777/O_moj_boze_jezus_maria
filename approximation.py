import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from abstract_car import AbstractCar

class FunctionApproximation(AbstractCar):
    def __init__(self, name, alpha: float, gamma: float, epsilon: float, epsilon_decay: float, min_epsilon: float):
        AbstractCar.__init__(self, name)
        self.alpha = alpha
        self.weights = np.ones(5)  # [vertical, horizontal, coming_in, mixed, ]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.to_plot_dict = defaultdict(list)
        self.possible_actions = ['forward', 'backward', 'stop', 'left', 'right']

    def get_best_action(self, state):
        qvalues = np.array([self.estimate_q(state, action) for action in self.possible_actions])
        best_indices = np.where(np.isclose(qvalues, qvalues.max()))[0]
        return self.possible_actions[np.random.choice(best_indices)]

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action 

    #! to jest tak mocno mniej wiecej jak to bedzie - slalomem bedzie to jezdzic mocno
    def middle_way_q(self, distances: np.ndarray, action: str) -> float:
        right, left = distances[0], distances[-1]
        if action == 'left':
            left -= 0.1 # to jest kategorycznie za duzo
            right += 0.1
        elif action == 'right':
            left += 0.1
            right -= 0.1

        return min(abs(right - left) - 0.25, 0.0)

    def velocity_q(self, vel: float, sin: float, action: str):
        if action == 'left':
            sin -= 0.1
        elif action == 'right':
            sin += 0.1
        elif action == 'forward':
            vel += 0.1
        elif action == 'backward':
            vel -= 0.1
        elif action == 'stop':
            vel -= 0.05

        return vel * sin

    def collision_q(self, distances: float, action: str):
        dx = np.zeros_like(distances)
        if action == 'left':
            dx += np.array([-0.2, -0.1, 0.0, 0.1, 0.2]) # od prawej do lewej tu leci
        elif action == 'right':
            dx += np.array([0.2, 0.1, 0.0, -0.1, -0.2])
        dx[2] += self.vel
        dx[1] += self.vel / 2
        dx[3] += self.vel / 2

        new_distances = distances + dx
        return int(np.any(new_distances < 0.15))

    def features(self, state: list, action: str) -> np.ndarray:
        distances, car_distances, sin, velocity = self.get_next_state(state, action)
        return np.array([])

    def estimate_q(self, state: list, action: str) -> float:
        return np.dot(self.features(state, action), self.weights)
    
    def update_weights(self, state: list, action: str, reward: float, next_state: list) -> None:
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        max_q_next = max(self.estimate_q(next_state, i) for i, a in enumerate(self.possible_actions))
        delta = reward + self.gamma * max_q_next - self.estimate_q(state, action)
        self.weights += self.alpha * delta * self.features(state, action)








    def record(self, car: int, game: int, TRACK):
        # Plot all metrics except positions
        for key, vals in self.to_plot_dict.items():
            if key in ('x', 'y'):
                continue
            plt.figure()
            plt.title(f'{key} for {car=}, {game=}')
            plt.plot(vals, 'o', markersize=1)
            plt.savefig(f'./last_{key}_{game}_{car}.png')
            plt.close()
            if key != 'loss':
                self.to_plot_dict[key] = []
        
        WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

        # Plot trajectory if position data exists
        if 'x' in self.to_plot_dict and 'y' in self.to_plot_dict:
            plt.figure()
            plt.scatter(self.to_plot_dict['x'], -np.array(self.to_plot_dict['y']), s=0.2)
            plt.axis('off')
            plt.xlim([0, WIDTH])
            plt.ylim([-HEIGHT, 0])
            plt.savefig(f'./last_positions_{game}_{car}.png')
            plt.close()
            self.to_plot_dict['x'] = []
            self.to_plot_dict['y'] = []
