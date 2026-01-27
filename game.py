import matplotlib.pyplot as plt
from constants import *
from time import time
from importlib import reload
import pygame
import numpy as np
from abstract_car import AbstractCar
from utils import scale_image
import random
import approximation
import bots

reload(bots)
reload(approximation)
#Based on https://github.com/techwithtim/Pygame-Car-Racer

pygame.init()

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.35)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.35)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.35)
GRAY_CAR = scale_image(pygame.image.load("imgs/grey-car.png"), 0.35)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24

FPS = 150

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=100):
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints

CHECKPOINTS = generate_checkpoints(track_path, num_checkpoints=250)
CHECKPOINT_DISTANCES = [np.hypot(CHECKPOINTS[0][1] - 180, CHECKPOINTS[0][0] - 200)] + [
    np.hypot(CHECKPOINTS[i][1] - CHECKPOINTS[i+1][1], CHECKPOINTS[i][0] - CHECKPOINTS[i+1][0]) 
    for i in range(len(CHECKPOINTS) - 1)
]

def draw_checkpoints(win, checkpoints, color=(0, 255, 0)):
    for x, y in checkpoints:
        pygame.draw.circle(win, color, (x, y), 5)


class Game:
    def __init__(self, width, height, fps=60, show=False):
        flag = pygame.SHOWN if show else pygame.HIDDEN
        self.win = pygame.display.set_mode((width, height), flag)
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.cars = []  # List to hold car objects
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
            (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        self.running = True

    def add_car(self, car):
        """Add a car to the game."""
        if not isinstance(car, AbstractCar):
            raise ValueError("Only instances of AbstractCar or its subclasses can be added.")

        if len(self.cars) == 0:
            car.set_image(RED_CAR)
            car.set_position((180, 200))
        elif len(self.cars) == 1:
            car.set_image(GREEN_CAR)
            car.set_position((150, 200))
        if len(self.cars) == 2:
            car.set_image(GRAY_CAR)
            car.set_position((180, 160))
        elif len(self.cars) == 3:
            car.set_image(PURPLE_CAR)
            car.set_position((150, 160))

        car.reset()
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""
        for img, pos in self.images:
            self.win.blit(img, pos)

        for car in self.cars:
            car.draw(self.win)
            car.draw_rays(self.win, TRACK_BORDER_MASK)

        pygame.display.update()

    def check_collisions(self):
        for car in self.cars:
            if car.collide(TRACK_BORDER_MASK):
                car.bounce()

        """Check for collisions between cars."""
        for i, car1 in enumerate(self.cars):
            for j, car2 in enumerate(self.cars):
                if i != j and car1.collide_car(car2):
                    car1.bounce()
                    car2.bounce()
                    # print(f"Collision between Car {i+1} and Car {j+1}!")

    def check_finish_line(self):
        finished = []

        for car in self.cars:
            finish_poi_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_poi_collide is not None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    self.cars.remove(car)

        return finished

    def get_state(self, car) -> list:
        _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
        car_distances = car.get_distances_to_cars(self.cars)
        # sin generalnie musi byc w range'u od -1 do 1 i dla tego kodziku jest
        angle_diff = (car.angle - car.angle_to_checkpoint + 180) % 360 - 180
        sin_diff = np.sin(np.radians(angle_diff / 2))

        # rzadko wychodzi poza 200
        distances = np.array(distances) / 200
        car_distances = np.array(car_distances) / 200

        car.to_plot_dict['x'].append(car.x)
        car.to_plot_dict['y'].append(car.y)

        front_indices = [0, -1, -2, -3, -4]

        # MINIMALIZM TUTAJ JAK NAJMNIEJ TEGO DAWAJ
        return [
            distances[front_indices],
            # car_distances[front_indices],
            sin_diff,
            car.vel / car.max_vel
        ]

    def move_cars(self, show):
        """Handle car movements."""
        # wszystko chce zrobic dyskretnym na DQLearning

        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        for i, car in enumerate(self.cars):
            state = self.get_state(car)
            
            action = car.choose_action(state)
            car.perform_action(action)
            car.update_progress(CHECKPOINTS)
            
            cos_angle = np.cos(np.radians(car.angle - car.angle_to_checkpoint))
            
            distances, sin, _ = state
            left, right = distances[0], distances[-1]

            #! idealna nagroda to nagroda ktora jest ciagla.
            # nagroda za jazde naprzod w kierunku checkpointa - powinno niezle dzialac
            velocity_reward = cos_angle * car.vel

            diff_reward = abs(left - right)

            sin_punish = 0.0
            if abs(state[1]) > 0.5: # za odwrocenie wzledem toru jazdy
                sin_punish = abs(state[1]) - 0.5

            action_reward = velocity_reward + diff_reward + sin_punish + bots.action_rewards(state, action, cos_angle, car, show)

            reward = action_reward
            # print(f'\r{velocity_reward=:20.2f}, {sin_punish=:15.2f}, {action_reward=}. {car.epsilon=:20.5f}', flush=True, end='')

            next_state = self.get_state(car)
            car.update_weights(state, action, reward, next_state)

    def run(self, show=False) -> list:
        """Main game loop."""
        who_finished_first = []
        start_time = time()
        time_passed = 0

        loop = 0

        while self.running and len(self.cars) != 0 and time_passed < SINGLE_GAME_STOP_TIME:
            # print(f'\r{time_passed / SINGLE_GAME_STOP_TIME * 100 :.2f}%, Epsilon={self.cars[0].epsilon}', flush=True, end='')
            # self.clock.tick(self.fps)
            loop += 1

            if show:
                for i, car in enumerate(self.cars):
                    draw_checkpoints(
                        self.win, 
                        CHECKPOINTS[car.checkpoint_index:car.checkpoint_index+1],
                        color=(255, 0, 0) if i == 0 else (0, 255, 0)
                    )
                    pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.move_cars(show)
            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            time_passed = time() - start_time
            if show:
                self.draw()

        pygame.quit()
        print("Game over!")
        print(who_finished_first)
        return who_finished_first
    

def main():
    final_results = dict()

    #initializing players - it is possible to play up to 4 players together
    players = [
        bots.FunctionApproximationCar(
            name="P2", 
            alpha=alpha_1,
            gamma=gamma_1,
            epsilon=epsilon_1,
            epsilon_decay=epsilon_decay_1,
            min_epsilon=min_epsilon_1,
        ),
        # bots.FunctionApproximationCar(
        #     "P1", 
        #     epsilon2,
        #     gamma2,  
        #     alpha2,  
        #     epsilon_decay_2,  
        #     min_epsilon_2,  
        # ),
        # bots.FunctionApproximationCar(
        #     "P3", 
        #     epsilon_1,
        #     gamma_1,
        #     alpha_1,
        #     epsilon_decay_1,
        #     min_epsilon_1,
        # ),
        # bots.FunctionApproximationCar(
        #     "P4", 
        #     epsilon_1,
        #     gamma_1,
        #     alpha_1,
        #     epsilon_decay_1,
        #     min_epsilon_1,
        # ),
    ]

    for i, p in enumerate(players):
        # p.load_weights(i % 2)
        final_results[p.get_name()] = 0

    # Wczytywanie wag
    # for i, p in enumerate(players):
    #     p.load_weights(i)
    start_before = time()
    last_loop = False
    game_counter = 0


    while time() - start_before < TRENING_TIME: 
        # zmienilbym trening calkiem zeby czasowo sie trenowaly a nie na iteracje
        p = list(players) # Create a copy if 'players' must remain in its original order
        random.shuffle(p)
        start_time = time()
        while (not last_loop and time() - start_time < SINGLE_GAME_STOP_TIME) or \
            (last_loop and time() - start_time < SHOWING_GAME_STOP_TIME):
            game = Game(WIDTH, HEIGHT, FPS, last_loop)

            # Add cars
            for player in p:
                game.add_car(player)

            # Run the game - show only the last game
            print(f'Przed runem {last_loop}')
            temp_rank = game.run(last_loop) 

            points = len(players)

            for tr in temp_rank:
                for t in tr:
                    final_results[t] += points
                points -= 1

            for j, car in enumerate(players):
                car.record(j, game_counter, TRACK)

        game_counter += 1

        if last_loop:
            break
        if np.max([car.epsilon for car in players]) < EPSILON_STOP \
            or time() - start_before > TRENING_TIME:
            last_loop = True

    for i, player in enumerate(players):
        player.save_model(i)
    print(final_results)

if __name__ == "__main__":
    main()