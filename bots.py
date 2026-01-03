import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import pygame
from abstract_car import AbstractCar

class FunctionApproximationCar(AbstractCar, nn.Module):
    def __init__(
            self, 
            name, 
            epsilon: float, 
            gamma: float, 
            alpha: float, 
            epsilon_decay: float, 
            min_epsilon: float = 0.1, 
            update_frequency: int = 10,
            digitize = True
        ):
        AbstractCar.__init__(self, name)
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.gamma = gamma
        self.alpha = alpha

        self.pos_record = []
        self.checkpoints_record = []
        self.sin_record = []
        self.loss_record = []
        self.collision_record = []

        self.all_possible_actions = ['forward', 'backward', 'stop', 'left', 'right']

        self.network = nn.Sequential(
            nn.LazyLinear(216),
            nn.ReLU(), 
            nn.Linear(216, 64),
            nn.ReLU(), 
            nn.Linear(64, 16),
            nn.ReLU(), 
            nn.Linear(16, 5),
        )

        self.loss = nn.SmoothL1Loss() # nn.MSELoss - zeby walczyc z szumem
        self.optim = torch.optim.SGD(self.parameters(), alpha)
        self.update_frequency = update_frequency
        self.update_counter = 0
        self.digitize = digitize

        self.sin_vals = np.linspace(-1, 1, num=9)
        self.distance_vals = np.linspace(0, 250, num=30)
        self.checkpoint_distance_vals = np.linspace(-100, 100, num=100)

    # wszystko musi byc lokalne
    def prepare_state(self, state) -> np.ndarray:
        if self.digitize:
            distances = np.digitize(state[0], self.distance_vals) / self.distance_vals.size
            car_distances = np.digitize(state[1], self.distance_vals) / self.distance_vals.size
            sin_angle = np.digitize([state[2]], self.sin_vals) / self.sin_vals.size

            checkpoint_x_dist = np.digitize([self.x_diff], self.distance_vals) / self.checkpoint_distance_vals.size
            checkpoint_y_dist = np.digitize([self.y_diff], self.distance_vals) / self.checkpoint_distance_vals.size
        else:
            distances = np.array(state[0]) / self.distance_vals[-1]
            car_distances = np.array(state[1]) / self.distance_vals[-1]
            sin_angle = np.array(state[2]) / self.sin_vals[-1]
            checkpoint_x_dist = np.array([self.x_diff]) / self.checkpoint_distance_vals[-1]
            checkpoint_y_dist = np.array([self.y_diff]) / self.checkpoint_distance_vals[-1]

        flat_state = np.concatenate([
            distances,
            car_distances,
            sin_angle,
            checkpoint_x_dist,
            checkpoint_y_dist
        ])

        if np.any(checkpoint_x_dist > 0.5) and np.any(checkpoint_y_dist > 0.5):
            print(f'{checkpoint_y_dist=}, {checkpoint_x_dist=}')

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
        if self.epsilon > 0.05:
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

        loss = self.loss(expected_reward, previous_reward) / self.update_frequency
        loss.backward() # UWAGA BO BYLO 1.0
        
        self.update_counter += 1

        if self.update_counter >= self.update_frequency:
            self.update_counter = 0
            
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optim.step()
            self.loss_record.append( loss.item() )
    
    def load_weights(self, i: int):
        self.network.load_state_dict(torch.load(f"model_{i}.pth"))

    def save_model(self, i: int):
        torch.save(self.network.state_dict(), f"model_{i}.pth")

    def save(self, checkpoints_reward: int, sin_reward, collision_punishment):
        self.checkpoints_record.append(checkpoints_reward)
        self.sin_record.append(sin_reward)
        self.pos_record.append(np.array([self.x, self.y]))
        self.collision_record.append(collision_punishment)

    def record(self, i: int, j: int):
        alpha = self.alpha
        gamma = self.gamma

        plt.title(rf'Reward stats for {i=} alpha={alpha}, $\epsilon_0$={self.epsilon}' + f'\ngamma={gamma}, checkpoints covered = {self.checkpoint_index}')
        plt.plot(self.checkpoints_record,   'o', markersize=0.25, label='Checkpoints covered')
        plt.plot(self.sin_record,           'o', markersize=0.25, label='Sin')
        plt.plot(self.collision_record,     'o', markersize=0.25, label='Collision')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Reward value')
        plt.savefig(f'./linear_{self.checkpoint_index}_{i}_epoch{j}.png')
        plt.close('all')
        plt.clf()

        checkpoints_record = np.array(self.checkpoint_distance)
        sin_record = np.array(self.sin_record)
        collision_record = np.array(self.collision_record)

        plt.title(rf'Reward stats for {i=} alpha={alpha}, $\epsilon_0$={self.epsilon}' + f'\ngamma={gamma}, checkpoints covered = {self.checkpoint_index}')
        plt.plot(np.log10(np.abs(checkpoints_record)+1e-8),    'o', markersize=0.25, label='Checkpoints covered')
        plt.plot(np.log10(np.abs(sin_record)+1e-8),            'o', markersize=0.25, label='Sin')
        plt.plot(np.log10(np.abs(collision_record)+1e-8),      'o', markersize=0.25, label='Collision')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Reward value')
        plt.savefig(f'./log_{self.checkpoint_index}_{i}_epoch{j}.png')
        plt.close('all')
        plt.clf()

        pos_record = np.array(self.pos_record)

        plt.title(f'Position history for {i=} car')
        plt.plot(pos_record[:, 0], -pos_record[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('off')
        plt.savefig(f'./position_reocord{i}_epoch{j}.png')
        plt.close('all')
        plt.clf()

        self.checkpoints_record = []
        self.sin_record = []
        self.distance_record = []
        self.collision_record = []
        self.pos_record = []