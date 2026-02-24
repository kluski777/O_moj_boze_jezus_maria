from collections import deque, defaultdict
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch.nn as nn
import copy

import math
from abstract_car import AbstractCar
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pygame
import pickle
import os
import torch
import copy
import seaborn as sns
from constants import *




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
        self.rewards_dict = defaultdict(list)
        self.eval_flag = eval_flag
        self.checkpoints = None

        self.gamma = gamma
        self.alpha = alpha

        self.all_possible_actions = ['forward', 'backward', 'stop', 'left', 'right']

        if FunctionApproximationCar.network is None:
            FunctionApproximationCar.network = nn.Sequential(
                nn.Linear(17, 512), nn.LayerNorm(512), nn.Mish(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(),
                nn.Linear(512, len(self.all_possible_actions))
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
        checkpoint_distance = state[4]

        flat_state = np.concatenate([
            distances,
            car_distance,
            [sin_angle],
            [velocity],
            [checkpoint_distance]
        ])
        
        return flat_state

    def action_rewards(self, state: list, action: str, cos: float, car, show: bool) -> float:
        # Dodac kat do poprzedniego checkpointa
        prev_check_x, prev_check_y = self.checkpoints[self.checkpoint_index - 1]
        check_diff_x = self.x - prev_check_x
        check_diff_y = self.y - prev_check_y
        angle_to_prev_check = math.degrees(math.atan2(check_diff_x, check_diff_y))
        angle_diff = (self.angle - angle_to_prev_check + 180) % 360 - 180
        prev_sin = np.sin(np.radians(angle_diff))
        prev_cos = np.cos(np.radians(angle_diff)) 
        # obliczyc kat do tego potwora. Od x i y 

        reward = 0.0
        vel = 2 * car.vel * (cos - 0.5)
        distances, car_distances, sin, _, _ = state
        right, right_front, front, left_front, left, back = distances
        back_car, front_car = car_distances[[-2, -6]]
        components = []

        steering_weight = 100 * abs(sin)
        def add(delta, reason):
            nonlocal reward
            reward += delta
            components.append((reason, delta))

        if action == 'left':
            if left_front > right_front:
                add(GOOD_SLIGHT_TURN, "left: more space left_front")
            elif right_front > left_front + TOLERATION:
                add(-GOOD_SLIGHT_TURN * 1.5, "left: right_front >> left_front")
            if left > right:
                add(GOOD_SHARP_TURN, "left: more space left")
            elif right > left + TOLERATION:
                add(-GOOD_SHARP_TURN * 1.5, "left: right >> left")
            if front > left:
                add(-GOOD_SHARP_TURN / 3, "left: could go straight (front>left)")
            if front > left_front:
                add(-GOOD_SLIGHT_TURN / 3, "left: could go straight (front>left_front)")
            if right_front < 0.2:
                add(CLOSE_TURN, "left: right_front close")
            if right < 0.2:
                add(CLOSE_TURN, "left: right close")
            elif left < right - PROXIMITY:
                add(-3 * CLOSE_TURN, "left: left closer than right")
            add(-sin * steering_weight, f"left: steering penalty (sin={sin:.3f})")

        if action != 'left' and (right < 0.2 or right_front < 0.2):
            add(-10.0, f"not left but right/right_front close")

        if action == 'right':
            if right_front > left_front:
                add(GOOD_SLIGHT_TURN, "right: more space right_front")
            elif left_front + 0.3 > right_front:
                add(-GOOD_SLIGHT_TURN * 2, "right: left_front >> right_front")
            if right > left:
                add(GOOD_SHARP_TURN, "right: more space right")
            elif left > right + 0.3:
                add(-GOOD_SHARP_TURN * 2, "right: left >> right")
            if front > right:
                add(-GOOD_SHARP_TURN / 3, "right: could go straight (front>right)")
            if front > right_front:
                add(-GOOD_SLIGHT_TURN / 3, "right: could go straight (front>right_front)")
            if left_front < 0.2:
                add(CLOSE_TURN, "right: left_front close")
            if left < 0.2:
                add(CLOSE_TURN, "right: left close")
            elif right < left - PROXIMITY:
                add(-3 * CLOSE_TURN, "right: right closer than left")
            add(sin * steering_weight, f"right: steering bonus (sin={sin:.3f})")
        if action != 'right' and (left < 0.2 or left_front < 0.2):
            add(-10.0, "not right but left/left_front close")

        if action == 'forward':
            if abs(sin) > 0.7:
                add(-420, "forward: going backwards (sin>0.7)")
            if left_front > front + 0.1 or right_front > front + 0.1:
                add(-TURN_INSTEAD_OF_FORWARD, "forward: should turn instead")
            else:
                add(TURN_INSTEAD_OF_FORWARD, "forward: front is clear")
            if left > front + 0.1:
                add(-TURN_INSTEAD_OF_FORWARD * 2, "forward: sideways (left>front)")
            if right > front + 0.1:
                add(-TURN_INSTEAD_OF_FORWARD * 2, "forward: sideways (right>front)")
            if front < 0.2 or front_car < 0.1:
                add(-car.vel * 2, "forward: wall/car ahead")
            if back_car < 0.1:
                add(50, "forward: car behind pushing")
            elif front_car < 1e-3:
                add(-420, "forward: bumper collision")
            else:
                add(4.5 / (car.vel + 1e-2), "forward: speed reward")

        if action != 'forward' and abs(car.vel) < 0.5:
            add(-200, "not forward but nearly stopped")

        if action == 'stop':
            if front + 0.2 > left or front + 0.2 > right:
                add(0.5, "stop: tight space")
            if vel < 0.5:
                add(-50.0, "stop: standing still")
            if car.vel > 2.0:
                add(3.0, "stop: braking at speed")

        if action == 'backward':
            if abs(sin) < 0.4:
                add(-5.0, "backward: facing forward")
            if front > 0.2:
                add(-10.0, "backward: front clear, no need")
            if abs(sin) > 0.4:
                add(0.25, "backward: correcting orientation")

        self.rewards_dict[action].append(components)

        if show:
            print(f'  {"TOTAL":10} | {reward:+10.2f} | vel={car.vel:.2f}')
        
        return reward
    
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
            self.to_plot_dict['q_per_action'].append(next_q.detach().cpu().numpy())
            max_next_q = torch.max(next_q)
            expected_reward = torch.tensor(reward, dtype=torch.float32) + self.gamma * max_next_q

        current_q = self.estimate_q(state)
        action_idx = self.all_possible_actions.index(action)
        previous_reward = current_q[action_idx]

        loss = self.loss(expected_reward, previous_reward)

        loss.backward()
        self.to_plot_dict['loss'].append(loss.item())
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.01)
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


    def plot_records(self, player_num, game_num, TRACK):
        os.makedirs("plots", exist_ok=True)
        WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
        
        for key, values in self.to_plot_dict.items():
            fig, ax = plt.subplots(figsize=(10, 4))
            if key == 'position':
                cmap = cm.get_cmap('plasma')
                n_episodes = len(values)
                split = max(0, n_episodes - 5)
                for i, episode in enumerate(values[:split]):
                    if len(episode) < 2:
                        continue
                    episode = np.array(episode)
                    ax.scatter(episode[:, 0], -episode[:, 1], s=5, alpha=0.3, color='steelblue')
                for i, episode in enumerate(values[split:]):
                    if len(episode) < 2:
                        continue
                    episode = np.array(episode)
                    color = cmap(i / max(4, 1))
                    ax.plot(episode[:, 0], -episode[:, 1], alpha=0.8, color=color, linewidth=1.2)
                plt.xlim([0, WIDTH])
                plt.ylim([-HEIGHT, 0])
                plt.axis('off')
                sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 5))
                fig.colorbar(sm, ax=ax, label='last 5 episodes')
            elif key == 'q_per_action':
                values = np.array(values)  # (steps, 5)
                window = max(1, len(values) // 20)
                for i, name in enumerate(self.all_possible_actions):
                    roll = np.convolve(values[:, i], np.ones(window)/window, mode='valid')
                    ax.plot(roll, linewidth=1.5, label=name)
                ax.legend()
            elif key == 'actions_taken':
                values = np.array(values)
                n = len(values)
                recent = values[int(n * 0.8):]  # last 20%
                
                total = len(values)
                total_recent = len(recent)
                counts_all = [100 * np.sum(values == i) / total for i in range(len(self.all_possible_actions))]
                counts_recent = [100 * np.sum(recent == i) / total_recent for i in range(len(self.all_possible_actions))]
                
                x = np.arange(len(self.all_possible_actions))
                ax.bar(x - 0.2, counts_all, width=0.4, label='all', alpha=0.6)
                ax.bar(x + 0.2, counts_recent, width=0.4, label='last 20%', alpha=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(self.all_possible_actions, rotation=15)
                ax.set_ylabel('%')
                ax.legend()
                ax.grid(True, alpha=0.3)
            elif key == 'distances':
                labels = ['right', 'right back', 'back', 'left back', 'left', 'left front', 'front', 'right front']
                values = self.to_plot_dict['distances']
                for vals, label in zip(values, labels):
                    plt.plot(values, label=label)
                plt.ylabel('Znormalizowana')
            else:
                n = len(values)
                ax.plot(np.arange(n), values, linewidth=0.8, alpha=0.7)
                window = max(1, n // 20)
                if n > window:
                    roll = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(np.arange(window-1, n), roll, linewidth=2, label=f'avg (w={window})')
                    ax.legend(fontsize=8)
                ax.set_xlabel('step')
                ax.set_ylabel(key)
                ax.xaxis.set_major_locator(MaxNLocator(10))
                ax.grid(True, alpha=0.3)
            ax.set_title(f'{self.name} | {key} | game {game_num} | epsilon {self.epsilon:.2f}')
            sns.despine(ax=ax)
            fig.savefig(f"plots/{key}_{player_num}.png", dpi=120, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("tab10", len(self.rewards_dict))
        for (key, vals), color in zip(self.rewards_dict.items(), palette):
            vals = np.array(vals)
            window = max(1, len(vals) // 20)
            roll = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(roll, linewidth=1.5, label=key, color=color)
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        fig.savefig(f"plots/rewards_{player_num}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)