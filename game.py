import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no tkinter, safe to use from any thread

import math
import threading
import cv2
import pygame
import torch
from abstract_car import AbstractCar
from utils import scale_image
from collections import deque
import numpy as np

#Based on https://github.com/techwithtim/Pygame-Car-Racer

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


FPS = 10000  # was 60; clock.tick() only caps — raising it removes the artificial sleep

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=250): # 1000 powinno wystarczyc
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints


CHECKPOINTS = generate_checkpoints(track_path)


def _reset_car_random(car):
    """Reset car to the actual starting line position and angle."""
    car.reset()  # clears vel, angle, checkpoint_index
    car.set_position((180, 200))
    car.angle = 0
    car.checkpoint_index = 0


def save_plots(actor_losses, critic_losses, rewards_history, entropy_losses, ratios,
               layer_norms_history, kl_history, mean_max_prob_history, action_freq_history,
               checkpoints_history, vel_history, wall_hits_history, latest_values, latest_returns):
    import matplotlib.pyplot as plt
    import matplotlib.cm as mplcm
    from model import ACTIONS

    single = [
        (actor_losses,          'Actor loss'),
        (critic_losses,         'Critic loss'),
        (rewards_history,       'Rollout reward'),
        (entropy_losses,        'Entropy loss'),
        (ratios,                'Mean PPO ratio'),
        (kl_history,            'KL divergence (old || new)'),
        (mean_max_prob_history, 'Mean max action probability'),
        (checkpoints_history,   'Mean checkpoints / episode'),
        (vel_history,           'Mean velocity (px/step)'),
        (wall_hits_history,     'Wall hits / episode'),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    fig.suptitle('Training diagnostics', fontsize=14)
    axs = axes.flatten()

    for ax, (data, title) in zip(axs, single):
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel('rollout')

    N = len(single)

    # action frequency: one line per action
    ax = axs[N]
    if action_freq_history:
        cmap = mplcm.get_cmap('tab10', len(ACTIONS))
        for i, name in enumerate(ACTIONS):
            ax.plot([f[i] for f in action_freq_history], label=name, color=cmap(i))
        ax.legend(fontsize=7)
    ax.set_title('Action frequency')
    ax.set_xlabel('rollout')
    ax.set_ylim(0, 1)

    # Karpathy update/weight ratio per module: |Δw|/|w|, healthy ≈ 1e-3
    ax = axs[N + 1]
    if layer_norms_history:
        keys = list(layer_norms_history[0].keys())
        cmap = mplcm.get_cmap('tab20', len(keys))
        for i, key in enumerate(keys):
            ax.plot([d[key] for d in layer_norms_history], label=key, color=cmap(i))
    ax.axhline(1e-3, color='k', linestyle='--', linewidth=1, label='1e-3 target')
    ax.set_yscale('log')
    ax.set_title('Update/weight ratio per layer')
    ax.set_xlabel('rollout')
    ax.legend(fontsize=6, ncol=3)

    # scatter: critic V(s) vs GAE returns
    ax = axs[N + 2]
    ax.scatter(latest_returns, latest_values, s=2, alpha=0.3)
    lo = min(latest_returns.min(), latest_values.min())
    hi = max(latest_returns.max(), latest_values.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    ax.set_title('V(s) vs returns')
    ax.set_xlabel('returns')
    ax.set_ylabel('V(s)')

    for ax in axs[N + 3:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=120)
    plt.close(fig)


def capture_frame(surface):
    """Capture surface → 256x256 grayscale float32 in [0,1]."""
    rgb  = pygame.surfarray.array3d(surface).transpose(1, 0, 2).astype(np.uint8)  # (H,W,3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
    return gray.astype(np.float32) / 255.0

def build_state(frames):
    grays = list(frames)  # oldest -> newest
    while len(grays) < 4:
        grays.insert(0, grays[0])
    # [newest, newest - prev, prev - prev2, prev2 - prev3]
    channels = [
        grays[3],
        grays[3] - grays[2],
        grays[2] - grays[1],
        grays[1] - grays[0],
    ]
    return torch.FloatTensor(np.stack(channels))


def show_frames(state):
    """Plot all frames of a stacked state (C, H, W) in one grid figure."""
    import matplotlib.pyplot as plt
    frames = state.detach().cpu().numpy()
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'frame {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# Actor - Critic gotta be
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
        self.frames = deque(maxlen=4)

        # pre-bake static background: blit all 4 images once into a single surface
        # so draw() does 1 blit instead of 4 (saves ~9ms per step)
        self._bg = pygame.Surface((width, height))
        for img, pos in self.images:
            self._bg.blit(img, pos)


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

        self.win.blit(self._bg, (0, 0))  # single blit of pre-baked static background

        for car in self.cars:
            car.draw(self.win)
            # car.draw_rays(self.win, TRACK_BORDER_MASK)

        pygame.display.update()
        self.frames.append(capture_frame(self.win))

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

    def move_cars(self):
        state = build_state(self.frames)

        reward = 0.0
        wall_hits = 0
        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        # "right", "back-right", "back", "back-left", "left", "front-left", "front", "front-right"
        for car in self.cars:
            action = car.choose_action(state.unsqueeze(0))
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            right, back_right, back, back_left, left, front_left, front, front_right = distances

            # kary za bycie zbyt blisko do sciany
            if front < 15:
                reward += (front - 15) / 25
                wall_hits += 1
            if front_left < 15:
                reward += (front_left - 15) / 25
                wall_hits += 1
            if front_right < 15:
                reward += (front_right - 15) / 25
                wall_hits += 1

            car.perform_action(action)

        return state, reward, wall_hits

    def run(self):
        """Main game loop."""
        car                 = self.cars[0]
        steps_since_checkpoint = 0
        rollout_reward      = 0.0
        rollout_vel         = 0.0
        rollout_wall_hits   = 0
        cp_crossings        = 0
        episode_count       = 0
        actor_losses        = []
        critic_losses       = []
        entropy_losses      = []
        ratios              = []
        layer_norms_history = []
        kl_history          = []
        mean_max_prob_history = []
        action_freq_history = []
        rewards_history     = []
        checkpoints_history = []
        vel_history         = []
        wall_hits_history   = []

        self.draw()  # seed self.frames before first move_cars

        while True:
            while self.running:
                checkpoint_idx_bef = car.get_progress()[0]
                self.clock.tick(self.fps)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                state, reward, wall_hits = self.move_cars()
                self.check_collisions()

                rollout_vel       += car.vel
                rollout_wall_hits += wall_hits

                # finish line: any contact = episode done + random reset (no bounce/remove)
                done = not self.running
                if car.collide(FINISH_MASK, *FINISH_POSITION):
                    done = True
                    steps_since_checkpoint = 0
                    _reset_car_random(car)
                    self.frames.clear()
                    self.draw()

                self.draw()
                next_state = build_state(self.frames)

                checkpoint_idx_now = car.get_progress()[0]
                if checkpoint_idx_now != checkpoint_idx_bef:
                    cp_crossings += 1
                    steps_since_checkpoint = 0
                else:
                    steps_since_checkpoint += 1

                cx, cy    = car.get_center()
                cp        = CHECKPOINTS[car.checkpoint_index % len(CHECKPOINTS)]
                to_cp     = (cp[0] - cx, cp[1] - cy)
                cp_dist   = math.sqrt(to_cp[0]**2 + to_cp[1]**2) + 1e-8
                to_cp     = (to_cp[0] / cp_dist, to_cp[1] / cp_dist)
                angle_rad = math.radians(car.angle)
                heading   = (-math.sin(angle_rad), -math.cos(angle_rad))
                cos_a     = heading[0] * to_cp[0] + heading[1] * to_cp[1]
                # cofanie dla cos < 0 jest wg gry ok. nie powinno tak byc imo. Poza tym trzeba jakos przeskalowac vel zeby nie wariowal.
                reward   += cos_a * car.vel / 8
                if abs(car.vel) < 0.5:
                    reward -= 0.01

                if steps_since_checkpoint >= 1024:
                    done = True
                    steps_since_checkpoint = 0
                    _reset_car_random(car)
                    self.frames.clear()
                    self.draw()

                if done:
                    episode_count += 1

                rollout_reward += reward
                car.store(state, car._last_action, car._last_log_prob, reward, done)

                # run the PPO update on a background thread so the main thread keeps
                # pumping events — prevents the OS "not responding" freeze
                _update_result = [(None,) * 10]
                def _do_update():
                    _update_result[0] = car.update(next_state.unsqueeze(0))
                _t = threading.Thread(target=_do_update, daemon=True)
                _t.start()
                while _t.is_alive():
                    for _e in pygame.event.get():
                        if _e.type == pygame.QUIT:
                            self.running = False
                    _t.join(timeout=0.05)
                a_loss, c_loss, e_loss, ratio, lnorms, kl, mean_max_prob, action_freq, v_arr, r_arr = _update_result[0]
                if a_loss is not None:
                    n = car.N
                    actor_losses.append(a_loss)
                    critic_losses.append(c_loss)
                    entropy_losses.append(e_loss)
                    ratios.append(ratio)
                    layer_norms_history.append(lnorms)
                    kl_history.append(kl)
                    mean_max_prob_history.append(mean_max_prob)
                    action_freq_history.append(action_freq)
                    rewards_history.append(rollout_reward)
                    checkpoints_history.append(cp_crossings / max(1, episode_count))
                    vel_history.append(rollout_vel / n)
                    wall_hits_history.append(rollout_wall_hits / max(1, episode_count))
                    rollout_reward    = 0.0
                    rollout_vel       = 0.0
                    rollout_wall_hits = 0
                    cp_crossings      = 0
                    episode_count     = 0
                    save_plots(actor_losses, critic_losses, rewards_history, entropy_losses, ratios,
                               layer_norms_history, kl_history, mean_max_prob_history, action_freq_history,
                               checkpoints_history, vel_history, wall_hits_history, v_arr, r_arr)
                    print(f"Update #{len(actor_losses):4d} | actor={a_loss:.4f}  critic={c_loss:.4f}  entropy={e_loss:.4f}  ratio={ratio:.4f}  kl={kl:.4f}  maxp={mean_max_prob:.3f}  reward={rewards_history[-1]:.3f}  cp/ep={checkpoints_history[-1]:.2f}  vel={vel_history[-1]:.2f}  hits/ep={wall_hits_history[-1]:.1f}")

            if not self.running:
                break  # window closed — exit the outer loop

        pygame.quit()


class PlayerCar2(AbstractCar):

    def __init__(self, name):
        # Call the AbstractCar __init__ method
        super().__init__(name)

    def choose_action(self, state):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            return "forward"
        elif keys[pygame.K_s]:
            return "backward"
        elif keys[pygame.K_a]:
            return "left"
        elif keys[pygame.K_d]:
            return "right"
        else:
            return "stop"

def main():
    import os
    from model import ActorCritic

    car = ActorCritic("P1", feature_dim=256)

    if os.path.exists('weights.pth'):
        car.load_state_dict(torch.load('weights.pth', map_location=car.device), strict=False)
        print("Loaded weights from weights.pth")

    # car = PlayerCar2("P1")

    game = Game(WIDTH, HEIGHT, FPS)
    game.add_car(car)

    try:
        game.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        torch.save(car.state_dict(), 'weights.pth')
        print("Saved weights to weights.pth")


if __name__ == "__main__":
    main()