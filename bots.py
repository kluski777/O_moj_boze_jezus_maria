from abstract_car import AbstractCar
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pygame
import torch
from constants import *

def action_rewards(state: list, action: str, cos: float, car, show: bool) -> float:
    reward = 0.0
    vel = car.vel * cos

    distances, sin = state
    right, right_front, front, left_front, left = distances

    # nie naprawilem tego bledu krecenia sie
    if action == 'left':
        if left_front > right_front + 0.05:  # wiecej miejsca z lewej do przodu
            reward += 2.0
        else:
            reward -= 2.0
        if left > right + 0.05:              # wiecej miejsca z lewej
            reward += 1.0
        else:
            reward -= 1.0
        
        if front > right:                   # Moglby jechac prosto
            reward -= 1.0
        if front > right_front:             # Moglby jechac prosto
            reward -= 1.0
        
        if left < 0.15:
            reward += 1.0
        elif left > 0.3:
            reward -= 2.5
        
        if sin > 0.4:
            reward -= 5.0
        elif sin < 0.0:
            reward += 0.5

    if action == 'right':
        if right_front > left_front + 0.05:  # Po prawej wiecej miejsca jest
            reward += 2.0
        else:
            reward -= 2.0
        if right > left + 0.05:              # Po prawej wiecej miejsca
            reward += 1.0
        else:
            reward -= 1.0

        if front > right:
            reward -= 1.0
        if front > right_front:
            reward -= 1.0
        
        if right < 0.15:
            reward += 1.0
        elif right > 0.3:
            reward -= 2.5

        if sin < -0.4:
            reward -= 5.0
        elif sin > 0.0:
            reward += 0.5
    
    if action == 'forward': # wydaje mi sie ze nie bedzie jezdzic do przodu
        if abs(sin) > 0.3:
            reward -= 2.5                   # COFA SIE
        if left_front > front:
            reward -= 0.2
        if right_front > front:
            reward -= 0.2
        if left > front:
            reward -= 2.5                   # stoi lub jedzie bokiem do toru
        if right > front:
            reward -= 2.5                   # stoi lub jedzie bokiem do toru
        if front > 0.55:
            reward += 2.0                   # jedz adam jedz
        reward += vel

    if action == 'stop':
        if front + 0.2 > left or front + 0.2 > right:
            reward += 0.5
        if vel < 1.0:
            reward -= 10.0                  # stoi i sie gapi
        if vel > 3.0:
            reward += 0.75

    if action == 'backward':
        if abs(sin) < 0.4:
            reward -= 5.0
        if front > 0.2:
            reward -= 10.0
        if abs(sin) > 0.4:
            reward += 0.25

    if show:
        print(f'{action:10}, {reward:10.2f}')

    return reward


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
            nn.LazyLinear(64), nn.ReLU(),
            nn.LazyLinear(64), nn.ReLU(),
            nn.LazyLinear(64), nn.ReLU(),
            nn.LazyLinear(len(self.all_possible_actions))
        )
        
        # self.loss = nn.MSELoss() # ty no nie wiem czy MSE jest lepsze idk tbh
        self.loss = nn.SmoothL1Loss() # nn.MSELoss - zeby walczyc z szumem
        self.optim = torch.optim.SGD(self.parameters(), alpha)

    # wszystko musi byc lokalne
    def prepare_state(self, state) -> np.ndarray:
        distances = state[0]
        # car_distance = state[1]
        sin_angle = state[1]
        # velocity = state[3]

        flat_state = np.concatenate([
            distances,
            # car_distance,
            [sin_angle],
            # [velocity]
        ])
        
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

        current_q = self.estimate_q(state)
        action_idx = self.all_possible_actions.index(action)
        previous_reward = current_q[action_idx]

        loss = self.loss(expected_reward, previous_reward)

        loss.backward()
        self.to_plot_dict['loss'].append(loss.item())
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
        self.optim.step()
        self.optim.zero_grad()  # Must clear here
        


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
