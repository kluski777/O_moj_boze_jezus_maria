import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from abstract_car import AbstractCar
from constants import *


#! Przetestowac czy to w ogole dziala i ma sens
def post_trening(old_state: np.ndarray, new_state: np.ndarray, cos_aft: float) -> float:
    old_car_dists = np.array(old_state[1])
    new_car_dists = np.array(new_state[1])
    old_sin = old_state[-2]
    new_sin = new_state[-2]

    # Dodatnie = oddalają się. Ujemne = zbliżają się.
    diff_dists = new_car_dists - old_car_dists
    
    # Znajdź najbliższego sąsiada teraz
    min_dist = np.min(new_car_dists)
    closest_idx = np.argmin(new_car_dists)
    dist_change = diff_dists[closest_idx]

    reward = 0.0
    
    #! Tlum - jak nie w tlumie to niech jedzie jak chce - byle do mety
    if min_dist < 0.4:
        if dist_change < -0.005: # zblizyl sie do innego auta -> UWAGA: musi to byc gorsze niz oddalanie sie 
            reward -= 20.0 
        elif dist_change > 0.005: # oddalil sie -> superancko
            reward += 10.0
        else:
            reward += 0.0 
    
    if min_dist < 0.05:     # zderzka - musi dostac po twarzy za to co uczynil
        reward -= 50.0

    old_dists = np.array(old_state[0])
    new_dists = np.array(old_state[0])
    arg_min = np.argmin(new_dists)
    if np.any(new_dists[arg_min] < 0.2) and old_dists[arg_min] > new_dists[arg_min]: # prawie zderzka i w dodatku jeszcze sie przybliza
        reward -= 15.0

    v = new_state[-1]
    # chcemy zeby jechal do przodu
    if cos_aft * v > 0.2:
        reward += - (v - 3.0) * (v - 0.2) # niech max bedzie dla 3-4
    else:
        reward -= 20.0

    if np.sign(old_sin - new_sin) == np.sign(old_sin):
        reward += abs(old_sin - new_sin) * 4.0 # w dobra strone skreca
    elif abs(old_sin - new_sin) > 2e-2:
        reward -= abs(old_sin - new_sin) * 4.0 # w zla strone skreca

    if np.all( np.abs(old_dists - new_dists) < 7e-3 ):
        reward -= 10.0 # nic nie robi.

    return reward


def action_rewards(state: list, action: str, cos: float, car, show: bool) -> float:
    reward = 0.0
    vel = 2 * car.vel * (cos - 0.5)

    distances, car_distances, sin, _ = state
    right, right_front, front, left_front, left = distances

    # trzeba dodac not action == KONIECZNIE

    # nie naprawilem tego bledu krecenia sie
    if action == 'left': #! szczegolnie w lewo
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
        
        if sin > 0.4:
            reward -= 50.0
        elif sin < -0.4:
            reward += GOOD_SHARP_TURN * 5
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

        if sin < -0.4:
            reward -= 50.0
        elif sin > 0.4:
            reward += GOOD_SHARP_TURN * 5
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
        if front < 0.2:
            reward -= car.vel * 2
        reward += 3.75 / (car.vel + 1e-2)

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

    # print(f'{left=:10.2f}, {left_front=:10.2f}, {right=:10.2f}, {right_front=:10.2f}')

    if show:
        print(f'{action:10}, {reward:10.2f}, {car.vel:10.2f}')

    return reward


class FunctionApproximation(AbstractCar):
    def __init__(self, name, alpha: float, gamma: float, epsilon: float, epsilon_decay: float, min_epsilon: float):
        AbstractCar.__init__(self, name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.to_plot_dict = defaultdict(list)
        self.possible_actions = ['forward', 'backward', 'stop', 'left', 'right']
        self.weights_per_action = {action: np.random.random(size=20) for action in self.possible_actions}
        
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

    def features(self, state: list) -> np.ndarray:
        distances, sin, velocity = state
        right, front_right, front, front_left, left = distances

        base = np.array([ # to sa tak naprawde moje funkcje
            right - left,
            front_right - front_left,
            front,
            front_left - right, 
            front_right - left,
            front - left,
            front - right,
            front - front_left,
            front - front_right,
            sin,
        ])
        base_and_velocity = base * velocity / 8 # tutaj normalizacja predkosci
        
        return np.concatenate([base, base_and_velocity]) # zobaczyc wymiarowosc

    def estimate_q(self, state: list, action: str) -> float:
        return np.dot(self.features(state), self.weights_per_action[action])
    
    def update_weights(self, state: list, action: str, reward: float, next_state: list) -> None:
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        max_q_next = max(self.estimate_q(next_state, a) for a in self.possible_actions)
        delta = reward + self.gamma * max_q_next - self.estimate_q(state, action)
        self.weights_per_action[action] += self.alpha * delta * self.features(state)

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

    def save_weights(self, player: int):
        from datetime import datetime
        filename = f'weights_{player}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'

        with open(f'./{filename}', 'wb') as f:
            pickle.dump(self.weights_per_action, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            self.weights_per_action = pickle.load(f)