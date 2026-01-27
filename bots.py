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
    vel = 2 * car.vel * (cos - 0.5)

    distances, sin, _ = state
    right, right_front, front, left_front, left = distances

    # trzeba dodac not action == KONIECZNIE

    # nie naprawilem tego bledu krecenia sie
    if action == 'left': #! szczegolnie w lewo
        if left_front > right_front:  # wiecej miejsca z lewej do przodu
            reward += GOOD_SLIGHT_TURN
        elif right_front > left_front + TOLERATION:
            # print(f'left wpierdol: {left_front=}, {right_front=}')
            reward -= GOOD_SLIGHT_TURN * 1.5
        if left > right:              # wiecej miejsca z lewej
            reward += GOOD_SHARP_TURN
        elif right > left + TOLERATION:
            # print(f'right: {left_front=}, {right_front=}')
            reward -= GOOD_SHARP_TURN * 1.5
        
        if front > left:                   # Moglby jechac prosto
            reward -= GOOD_SHARP_TURN / 3
        if front > left_front:             # Moglby jechac prosto
            reward -= GOOD_SLIGHT_TURN / 3
        
        if right_front < 0.15:
            reward += CLOSE_TURN
        if right < 0.15:
            reward += CLOSE_TURN
        elif left < right - PROXIMITY:
            # print(f'right: {left=}, {right=}')
            reward -= 3 * CLOSE_TURN
        
        if sin > 0.4:
            reward -= 50.0
        elif sin < -0.4:
            reward += GOOD_SHARP_TURN * 5

    if action == 'right':
        if right_front > left_front:  # Po prawej wiecej miejsca jest
            reward += GOOD_SLIGHT_TURN
        elif left_front + 0.3 > right_front:
            # print(f'right: {left_front=}, {right_front=}')
            reward -= GOOD_SLIGHT_TURN * 2
        if right > left:              # Po prawej wiecej miejsca
            reward += GOOD_SHARP_TURN
        elif left > right + 0.3:
            reward -= GOOD_SHARP_TURN * 2

        if front > right:
            reward -= GOOD_SHARP_TURN / 3
        if front > right_front:
            reward -= GOOD_SLIGHT_TURN / 3
        
        if left_front < 0.15:
            reward += CLOSE_TURN
        if left < 0.15:
            reward += CLOSE_TURN
        elif right < left - PROXIMITY:
            # print(f'right: {left=}, {right=}')
            reward -= 3 * CLOSE_TURN

        if sin < -0.4:
            reward -= 50.0
        elif sin > 0.4:
            reward += GOOD_SHARP_TURN * 5
    
    if action == 'forward':
        if abs(sin) > 0.7: # odchylony o wiecej niz 60 stopni 
            reward -= 420                   # COFA SIE
        
        if left_front > front + 0.1 or right_front > front + 0.1:
            reward -= TURN_INSTEAD_OF_FORWARD
        else:
            reward += TURN_INSTEAD_OF_FORWARD

        if left > front + 0.1:
            reward -= TURN_INSTEAD_OF_FORWARD * 2                   # stoi lub jedzie bokiem do toru
        if right > front + 0.1:
            reward -= TURN_INSTEAD_OF_FORWARD * 2                   # stoi lub jedzie bokiem do toru
        if front < 0.2:
            reward -= car.vel * 2
        reward += vel

    if action == 'stop':
        if front + 0.2 > left or front + 0.2 > right:
            reward += 0.5
        if vel < 0.5:
            reward -= 10.0                  # stoi i sie gapi
        if car.vel > 1:
            reward += 3.0

    if action == 'backward':
        if abs(sin) < 0.4:
            reward -= 5.0
        if front > 0.2:
            reward -= 10.0
        if abs(sin) > 0.4:
            reward += 0.25

    if show:
        print(f'{action:10}, {reward:10.2f}, {car.vel:10.2f}')

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
        velocity = state[2]

        flat_state = np.concatenate([
            distances,
            # car_distance,
            [sin_angle],
            [velocity]
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
