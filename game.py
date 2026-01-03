import matplotlib.pyplot as plt
from time import time
from importlib import reload
import pygame
import bots
from abstract_car import AbstractCar
from utils import scale_image
from itertools import permutations
import numpy as np

reload(bots)
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

show_every_i = 1
lasting = 15 * 60 # czas trenowania
velocity_scaler = 0.0

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
    def __init__(self, width, height, fps=60):
        self.win = pygame.display.set_mode((width, height))
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
            if finish_poi_collide != None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    self.cars.remove(car)

        return finished

    def get_state(self, car) -> list:
        _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
        car_distances = car.get_distances_to_cars(self.cars)
        sin_diff = np.sin(np.radians(car.angle - car.angle_to_checkpoint))

        # niby max length raya to 1000 ale dam 500
        distances = np.array(distances) / 500
        car_distances = np.array(car_distances) / 500

        return [
            distances, 
            car_distances,
            sin_diff
        ]

    def move_cars(self, show_stats: bool):
        """Handle car movements."""

        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        for i, car in enumerate(self.cars):
            state = self.get_state(car)
            
            prev_distance = car.progress_distance
            action = car.choose_action(state)
            car.update_progress(CHECKPOINTS)
            prev_index, _ = car.get_progress()
            
            curr_distance = car.progress_distance
            car.perform_action(action)
            car.update_progress(CHECKPOINTS)
            current_index, _ = car.get_progress()

            # kara za zle ustawienie sie do kierunku jazdy
            cos_angle = np.cos(np.radians(car.angle - car.angle_to_checkpoint))
            velocity = abs(prev_distance - curr_distance)
            proximity_reward = velocity * cos_angle

            #TODO ktory to sensor w lewo a ktory to sensor w prawo
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            left_distance = distances[0]
            right_distance = distances[4]
            distance_from_the_wall = abs(left_distance - right_distance)

            # nagroda za to jak duza czesc toru pokryl
            checkpoint_reward = 0.0
            if prev_index != current_index:
                checkpoint_reward = 15.0

            # kara za kolizje
            collision = car.collide(TRACK_BORDER_MASK)
            collision_punishment = 0.0 if collision is None else -2.0

            # if i == 1 and show_stats:
            #     print(f'\r{checkpoint_reward=}, {proximity_reward=}, {collision_punishment=}, {car.epsilon=}', flush=True, end='')

            reward = checkpoint_reward + \
                proximity_reward + \
                collision_punishment - \
                (distance_from_the_wall / 20) ** 2

            # print(f'\r{(distance_from_the_wall / 100) ** 2}, {collision_punishment=}, {proximity_reward=}, {checkpoint_reward=}', flush=True, end='')

            next_state = self.get_state(car)
            car.update_weights(state, action, reward, next_state)
            car.save(checkpoint_reward, cos_angle, collision_punishment)

    def run(self, show=False) -> list:
        """Main game loop."""
        who_finished_first = []
        start_time = time()
        time_passed = 0

        loop = 0

        while self.running and len(self.cars) != 0 and time_passed < lasting:
            print(f'\r{time_passed / lasting * 100 :.2f}%', flush=True, end='')
            # self.clock.tick(self.fps)
            loop += 1

            if loop % show_every_i == 0 and show:
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

def save_ways(players, num_circuit: int):
    plt.title(f'loss record for all cars, circuit={num_circuit}')
    for i, car in enumerate(players):
        plt.plot(car.loss_record, 'o', markersize=1, label=f'{i+1}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'./loss_{num_circuit}.png')
    plt.clf()
    plt.close('all')

    plt.title(f'log10 loss record for all cars, circuit={num_circuit}')
    for i, car in enumerate(players):
        plt.plot(np.log10(np.array(car.loss_record)+1e-8), 'o', markersize=1, label=f'{i+1}')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'./log_loss_{num_circuit}.png')
    plt.clf()
    plt.close('all')

def main():
    final_results = dict()

    epsilon = 1.0
    gamma = 0.99
    alpha = 1.5e-2
    epsilon_decay_1 = 1e-5
    min_epsilon_1 = 0.3
    update_freq_1 = 1
    digitize_1 = True

    epsilon2 = 1.0
    gamma2 = 0.99
    alpha2 = 2.5e-2
    epsilon_decay_2 = 1e-5
    min_epsilon_2 = 0.3
    update_freq_2 = 1
    digitize_2 = False

    epsilon3 = 1.0
    gamma3 = 0.99
    alpha3 = 1.5e-2
    epsilon_decay_3 = 1e-5
    min_epsilon_3 = 0.3
    update_freq_3 = 1
    digitize_3 = False

    #initializing players - it is possible to play up to 4 players together
    players = [
        bots.FunctionApproximationCar(
            "P1", 
            epsilon, 
            gamma, 
            alpha, 
            epsilon_decay_1, 
            min_epsilon_1, 
            update_freq_1
        ),
        bots.FunctionApproximationCar(
            "P2", 
            epsilon2, 
            gamma2, 
            alpha2,  
            epsilon_decay_2, 
            min_epsilon_2, 
            update_freq_2
        ),
        bots.FunctionApproximationCar(
            "P3", 
            epsilon3, 
            gamma3, 
            alpha3, 
            epsilon_decay_3, 
            min_epsilon_3, 
            update_freq_3
        ),
        # bots.FunctionApproximationCar("P4", epsilon, gamma, alpha, TRACK.get_width(), TRACK.get_height()),
    ]

    for p in players:
        final_results[p.get_name()] = 0

    perm = list(permutations(players))

    # Wczytywanie wag
    # for i, p in enumerate(players):
    #     p.load_weights(i)

    for k in range(2):
        print('Training' if k == 0 else 'Testing')
        for i, p in enumerate(perm):
            print(f'Loop {i}')

            start_time = time()
            while time() - start_time < lasting:
                game = Game(WIDTH, HEIGHT, FPS)

                # Add cars
                for player in p:
                    game.add_car(player)

                # Run the game
                temp_rank = game.run(k==1)

                points = len(players)

                for tr in temp_rank:
                    for t in tr:
                        final_results[t] += points
                    points -= 1

                if k == 0: # jak jest k == 0 to pokazuje na screenie nie ma potrzeby na wykresy
                    for j, car in enumerate(players):
                        car.record(j, i)

    for i, p in enumerate(players):
        p.save_model(i)

    print(final_results)

if __name__ == "__main__":
    main()