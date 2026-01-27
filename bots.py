from abstract_car import AbstractCar
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pygame
import torch

class FunctionApproximationCar(AbstractCar, nn.Module):
    def __init__(
            self, 
            name, 
            epsilon: float, 
            gamma: float, 
            alpha: float, 
            epsilon_decay: float, 
            min_epsilon: float = 0.1, 
        ):
        AbstractCar.__init__(self, name)
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.to_plot_dict = defaultdict(list)

        self.gamma = gamma
        self.alpha = alpha

        self.all_possible_actions = ['forward', 'backward', 'stop', 'left', 'right']

        self.network = nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(5),
        )
        
        self.loss = nn.MSELoss() # ty no nie wiem czy MSE jest lepsze idk tbh
        # self.loss = nn.SmoothL1Loss() # nn.MSELoss - zeby walczyc z szumem
        self.optim = torch.optim.SGD(self.parameters(), alpha)

    # wszystko musi byc lokalne
    def prepare_state(self, state) -> np.ndarray:
        distances = state[0]
        # car_distance = state[1]
        sin_angle = state[2]
        # velocity = state[3]

        flat_state = np.concatenate([
            distances,
            # car_distance,
            [sin_angle],
            # [velocity]
        ])
        
        # indx = 0 jest na prawo, w praktyce potrzebuje indeksow 0, -1, -2, -3, -4 -> tylko 5 kierunkow
        for i, distance in enumerate(distances):
            self.to_plot_dict[f'{i}_distance'].append(distance)
        self.to_plot_dict['sin'].append(sin_angle)
        # self.to_plot_dict['vel'].append(velocity)

        return flat_state

    def get_best_action(self, state: list) -> str:
        with torch.no_grad():
            q_values = self.estimate_q(state)
            best_action_idx = torch.argmax(q_values).item()
        return self.all_possible_actions[best_action_idx]

    def choose_action(self, state: list) -> str:
        # keys = pygame.key.get_pressed()
    
        # if keys[pygame.K_UP]:
        #     return "forward"
        # elif keys[pygame.K_DOWN]:
        #     return "backward"
        # elif keys[pygame.K_LEFT]:
        #     return "left"
        # elif keys[pygame.K_RIGHT]:
        #     return "right"
        # else:
        #     return "stop"
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_possible_actions)
        else:
            return self.get_best_action(state)


    def estimate_q(self, state) -> torch.Tensor:
        state_prepared = self.prepare_state(state)
        input = torch.tensor(state_prepared, dtype=torch.float32)
        return self.network(input)
    
    def update_weights(self, state: np.ndarray, action: str, reward: float, next_state: list):
        if self.epsilon > self.min_epsilon:
            # jak bardzo epsilon spadnie maja problemy z odbijaniem sie od scian
            self.epsilon *= (1 - self.epsilon_decay)
        
        self.optim.zero_grad()
        
        with torch.no_grad():
            next_q = self.estimate_q(next_state)
            max_next_q = torch.max(next_q)
            expected_reward = torch.tensor(reward, dtype=torch.float32) + self.gamma * max_next_q
        self.update_counter += 1

        current_q = self.estimate_q(state)
        action_idx = self.all_possible_actions.index(action)
        previous_reward = current_q[action_idx]

        loss = self.loss(expected_reward, previous_reward)

        loss.backward()
        self.to_plot_dict['loss'].append(loss.item())
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
        self.optim.step()
        self.optim.zero_grad()  # Must clear here
        
        # if self.update_counter % self.target_update_freq == 0:
            # self.target_network.load_state_dict(self.network.state_dict())


    def load_weights(self, i: int):
        self.network.load_state_dict( torch.load(f"model_{i}.pth", weights_only=True) )

    def save_model(self, i: int):
        torch.save( self.network.state_dict(), f"model_{i}.pth" )

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
