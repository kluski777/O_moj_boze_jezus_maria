from abstract_car import AbstractCar
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pygame
import pickle
import torch
import copy
from constants import *


def action_rewards(state: list, action: str, cos: float, car, show: bool) -> float:
    reward = 0.0
    vel = 2 * car.vel * (cos - 0.5)
    distances, car_distances, sin, _ = state
    right, right_front, front, left_front, left, back = distances
    back_car, front_car = car_distances

    steering_weight = 150 * abs(sin)
    if action == 'left':
        if left_front > right_front:  # wiecej miejsca z lewej do przodu
            reward += GOOD_SLIGHT_TURN
        elif right_front > left_front + TOLERATION:
            reward -= GOOD_SLIGHT_TURN * 1.5
        if left > right:              # wiecej miejsca z lewej
            reward += GOOD_SHARP_TURN
        elif right > left + TOLERATION:
            reward -= GOOD_SHARP_TURN * 1.5
        if front > left:                   # Moglby jechac prosto
            reward -= GOOD_SHARP_TURN / 3
        if front > left_front:             # Moglby jechac prosto
            reward -= GOOD_SLIGHT_TURN / 3
        if right_front < 0.2:
            reward += CLOSE_TURN
        if right < 0.2:
            reward += CLOSE_TURN
        elif left < right - PROXIMITY:
            reward -= 3 * CLOSE_TURN
        
        reward -= sin * steering_weight
    if action != 'left' and (right < 0.2 or right_front < 0.2):
        reward -= 10.0
    
    if action == 'right':
        if right_front > left_front:  # Po prawej wiecej miejsca jest
            reward += GOOD_SLIGHT_TURN
        elif left_front + 0.3 > right_front:
            reward -= GOOD_SLIGHT_TURN * 2
        if right > left:              # Po prawej wiecej miejsca
            reward += GOOD_SHARP_TURN
        elif left > right + 0.3:
            reward -= GOOD_SHARP_TURN * 2
        if front > right:
            reward -= GOOD_SHARP_TURN / 3
        if front > right_front:
            reward -= GOOD_SLIGHT_TURN / 3
        if left_front < 0.2:
            reward += CLOSE_TURN
        if left < 0.2:
            reward += CLOSE_TURN
        elif right < left - PROXIMITY:
            reward -= 3 * CLOSE_TURN
        reward += sin * steering_weight
    if action != 'right' and (left < 0.2 or left_front < 0.2):
            reward -= 10.0
    
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
        if front < 0.2 or front_car < 0.1:
            reward -= car.vel * 2
        if back_car < 0.1:
            reward += 50
        elif front_car < 1e-3: # najprawdopodobniej zderzka
            reward -= 420
        else:
            reward += 4.5 / (car.vel + 1e-2)
    
    if action == 'stop':
        if front + 0.2 > left or front + 0.2 > right:
            reward += 0.5
        if vel < 0.5:
            reward -= 50.0                  # stoi i sie gapi
        if car.vel > 2.0:
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
    network = None
    optim = None
    target_network = None
    update_counter = 0
    
    def __init__(
            self, 
            name, 
            epsilon: float, 
            gamma: float, 
            alpha: float, 
            epsilon_decay: float, 
            min_epsilon: float = 0.1, 
            eval_flag: bool = False
        ):
        AbstractCar.__init__(self, name)
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.collision_counter = 0
        self.min_epsilon = min_epsilon
        self.to_plot_dict = defaultdict(list)
        self.eval_flag = eval_flag

        self.gamma = gamma
        self.alpha = alpha

        self.all_possible_actions = ['forward', 'backward', 'stop', 'left', 'right']

        if FunctionApproximationCar.network is None:
            FunctionApproximationCar.network = nn.Sequential(
                nn.Linear(10, 216), nn.Mish(),
                nn.Linear(216, 216), nn.Mish(),
                nn.Linear(216, len(self.all_possible_actions))
            )
            FunctionApproximationCar.target_network = copy.deepcopy(
                FunctionApproximationCar.network
            )
            FunctionApproximationCar.optim = torch.optim.SGD(
                FunctionApproximationCar.network.parameters(), 
                alpha
            )
            
        self.network = FunctionApproximationCar.network
        self.target_network = FunctionApproximationCar.target_network
        self.optim = FunctionApproximationCar.optim
        self.loss = nn.SmoothL1Loss()

    def prepare_state(self, state) -> np.ndarray:
        distances = state[0]
        car_distance = state[1]
        sin_angle = state[2]
        velocity = state[3]

        flat_state = np.concatenate([
            distances,
            car_distance,
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
        if self.eval_flag:
            return self.get_best_action(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_possible_actions)
        return self.get_best_action(state)

    def estimate_q(self, state) -> torch.Tensor:
        state_prepared = self.prepare_state(state)
        input = torch.tensor(state_prepared, dtype=torch.float32)
        return self.network(input)
    
    def update_weights(self, state: np.ndarray, action: str, reward: float, next_state: list):
        if self.eval_flag:
            return
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= (1 - self.epsilon_decay)
        
        # Use target network for stable Q-target
        with torch.no_grad():
            next_state_prepared = self.prepare_state(next_state)
            next_input = torch.tensor(next_state_prepared, dtype=torch.float32)
            next_q = self.target_network(next_input)
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
        self.optim.zero_grad()

        # Sync target network every 500 steps
        FunctionApproximationCar.update_counter += 1
        if FunctionApproximationCar.update_counter % 500 == 0:
            FunctionApproximationCar.target_network.load_state_dict(
                FunctionApproximationCar.network.state_dict()
            )

    def save_model(self, filename: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
        }, filename)

    def load_weights(self, filename: str):
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        if 'target_network_state_dict' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        else:
            self.target_network.load_state_dict(checkpoint['network_state_dict'])
        print(f'Wagi wczytane z {filename}')

    def record(self, car: int, game: int, TRACK):
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