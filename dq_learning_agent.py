from abstract_car import AbstractCar
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle

class DQLearningAgent(AbstractCar):
    def __init__(self, name, alpha, epsilon, discount, weights_path = None):
        """
        Double Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
            - self.epsilon (exploration prob)
            - self.alpha (learning rate)
            - self.discount (discount rate aka gamma)
        """
        AbstractCar.__init__(self, name)
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.get_legal_actions = ['forward', 'backward', 'stop', 'left', 'right']
        self.to_plot_dict = defaultdict(list)
        self.distance_probing = np.linspace(0, 1, num=5)
        self.sin_probing = np.linspace(-1, 1, num=5)

        if weights_path is not None:
            self.load_weights(weights_path)

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        # dzielić to przez 2 czy nie ??? 
        return (self._qvaluesA[state][action] + self._qvaluesB[state][action]) / 2

    def get_qvalueA(self, state, action):
        """ Returns Qa(state, action) """
        return self._qvaluesA[state][action]

    def get_qvalueB(self, state, action):
        """ Returns Qa(state, action) """
        return self._qvaluesB[state][action]

    def get_best_action(self, state, q_value_function):
        """ Returns action that maxes out q_value_function """
        possible_actions = self.get_legal_actions.copy()
        q_values = [q_value_function(state, action) for action in possible_actions]

        max_q = np.max(q_values)
        best_indices = np.where(np.isclose(max_q, q_values))[0]

        return possible_actions[np.random.choice(best_indices)]

    #---------------------START OF YOUR CODE---------------------#

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        if np.random.random() < 0.5:
            best_qa_action = self.get_best_action(next_state, self.get_qvalueA)
            value = self._qvaluesA[state][action] + learning_rate * (reward + gamma * self._qvaluesB[next_state][best_qa_action] - self._qvaluesA[state][action])
            self._qvaluesA[state][action] = value
        else:
            best_qb_action = self.get_best_action(next_state, self.get_qvalueB)
            value = self._qvaluesB[state][action] + learning_rate * (reward + gamma * self._qvaluesA[next_state][best_qb_action] - self._qvaluesB[state][action])
            self._qvaluesB[state][action] = value

    def prepare_state(self, state):
        distances, car_distances, sin_diff = state
        min_distances = np.minimum(distances, car_distances)
        important_distances = min_distances[[0, -1, -2, -3, -4]]
        digitized_distances = np.digitize(important_distances, self.distance_probing)
        digitized_sin = np.digitize(sin_diff, self.sin_probing)
        return np.concatenate([[digitized_sin], digitized_distances], dtype=int)

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
            To pick True or False with a given probablity, generate uniform number in [0, 1]
            and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions.copy()

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if np.random.random() < epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state, self.get_qvalue)

        return chosen_action

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0

    def save_model(self, i: int):
        state = {
            'qvaluesA': dict(self._qvaluesA),
            'qvaluesB': dict(self._qvaluesB),
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'discount': self.discount
        }
        with open(f"model_{i}.pkl", 'wb') as f:
            pickle.dump(state, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0), state['qvaluesA'])
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0), state['qvaluesB'])
        self.alpha = state['alpha']
        self.epsilon = state['epsilon']
        self.discount = state['discount']

    def record(self, car: int, game: int):
        # Plot all metrics except positions
        for key, vals in self.to_plot_dict.items():
            if key in ('x', 'y'):
                continue
            plt.figure()
            plt.title(f'{key} for {car=}, {game=}')
            plt.plot(vals, 'o', markersize=1)
            plt.savefig(f'./last_{key}_{game}_{car}.png')
            plt.close()
            self.to_plot_dict[key] = []
        
        # Plot trajectory if position data exists
        if 'x' in self.to_plot_dict and 'y' in self.to_plot_dict:
            plt.figure()
            plt.scatter(self.to_plot_dict['x'], -np.array(self.to_plot_dict['y']), s=0.2)
            plt.axis('off')
            plt.savefig(f'./last_positions_{game}_{car}.png')
            plt.close()
            self.to_plot_dict['x'] = []
            self.to_plot_dict['y'] = []