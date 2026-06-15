import math
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


WALL_VAL     = 0.0
DRIVABLE_VAL = 0.5
CAR_VAL      = 1.0
CROP = 256

_border_alpha = pygame.surfarray.array_alpha(TRACK_BORDER).T  # (H, W)
TRACK_MAP = np.where(_border_alpha > 0, WALL_VAL, DRIVABLE_VAL).astype(np.float32)


FPS = 60

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]

def draw_checkpoints(win, checkpoints):
    for x, y in checkpoints:
        pygame.draw.circle(win, (0, 255, 0), (x, y), 5)


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=250):
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


def save_plots(actor_losses, critic_losses, rewards_history, entropy_losses, ratios,
            layer_norms_history, critic_values, states_stds, checkpoints_history,
            action_freq_history, wall_hits_history, episode_results,
            latest_values, latest_returns):
    import matplotlib.pyplot as plt
    import numpy as np

    ROWS, COLS = 5, 3
    fig, axes = plt.subplots(ROWS, COLS, figsize=(18, 25))
    fig.suptitle('Training diagnostics', fontsize=14)
    axs = axes.flatten()

    single = [
        (actor_losses,        'Actor loss'),
        (critic_losses,       'Critic loss'),
        (rewards_history,     'Rollout reward'),
        (entropy_losses,      'Entropy loss'),
        (ratios,              'Mean PPO ratio'),
        (critic_values,       'Mean critic value'),
        (states_stds,         'States std (batch)'),
        (wall_hits_history,   'Wall hits per rollout'),
    ]

    for ax, (data, title) in zip(axs, single):
        ax.plot(data, alpha=0.5)
        if len(data) > 1:
            running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
            ax.plot(running_avg, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('rollout')

    ax = axs[8]
    ax.plot(checkpoints_history, alpha=0.5, label='per rollout')
    if len(checkpoints_history) > 1:
        running_avg = np.cumsum(checkpoints_history) / np.arange(1, len(checkpoints_history) + 1)
        ax.plot(running_avg, linewidth=1.5, label='running avg')
    ax.set_title('Checkpoints per rollout')
    ax.set_xlabel('rollout')
    ax.legend(fontsize=7)

    ax = axs[9]
    if layer_norms_history:
        keys = list(layer_norms_history[0].keys())
        import matplotlib.cm as mplcm
        cmap = mplcm.get_cmap('tab20', len(keys))
        for i, key in enumerate(keys):
            vals = [d[key] for d in layer_norms_history]
            ax.plot(vals, label=key, color=cmap(i))
    ax.axhline(1e-3, color='k', linestyle='--', linewidth=1, label='1e-3 target')
    ax.set_yscale('log')
    ax.set_title('Update/weight ratio per layer')
    ax.set_xlabel('rollout')
    ax.legend(fontsize=6, ncol=3)

    ax = axs[10]
    ax.scatter(latest_returns, latest_values, s=2, alpha=0.3)
    lo = min(latest_returns.min(), latest_values.min())
    hi = max(latest_returns.max(), latest_values.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    ax.set_title('V(s) vs returns')
    ax.set_xlabel('returns')
    ax.set_ylabel('V(s)')

    ax = axs[11]
    from model import ACTIONS
    if action_freq_history:
        for i, name in enumerate(ACTIONS):
            ax.plot([f[i] for f in action_freq_history], label=name)
    ax.set_title('Action frequency per rollout')
    ax.set_xlabel('rollout')
    ax.set_ylabel('fraction')
    ax.legend(fontsize=8)

    ax = axs[12]
    for ep_idx, (steps, finished) in enumerate(episode_results):
        ax.scatter(ep_idx, steps, color='green' if finished else 'red', s=10, alpha=0.7)
    ax.set_xlabel('episode')
    ax.set_ylabel('steps')
    if episode_results:
        success_rate = sum(1 for _, f in episode_results if f) / len(episode_results)
        running_sr = [sum(1 for _, f in episode_results[:i+1] if f) / (i+1) for i in range(len(episode_results))]
        ax2 = ax.twinx()
        ax2.plot(running_sr, color='blue', linewidth=1.5, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('success rate', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax.set_title(f'Steps per episode — success rate: {success_rate:.1%}')
    else:
        ax.set_title('Steps per episode (green=finish, red=timeout)')

    for ax in axs[13:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=120)
    plt.close(fig)


def capture_frame(car):
    """Heading-aligned 256x256 semantic crop: the car's forward direction always
    points up, and the car sits near the bottom so the view shows the track ahead.
    Three classes only: wall=0.0, drivable=0.5, car=1.0."""
    cx, cy = car.get_center()
    a = math.radians(car.angle)
    ca, sa = math.cos(a), math.sin(a)

    # car anchored near the bottom-center of the output
    ch = car.img.get_height()
    car_row = CROP - ch // 2 - 2

    # inverse affine: output (col, row) -> world (x, y).
    # output "up" maps to the car's forward heading, output (128, car_row) -> car center.
    tx = cx - (ca * 128 + sa * car_row)
    ty = cy - (-sa * 128 + ca * car_row)
    M = np.array([[ca, sa, tx],
                  [-sa, ca, ty]], dtype=np.float32)
    out = cv2.warpAffine(
        TRACK_MAP, M, (CROP, CROP),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderValue=WALL_VAL,   # off-track is wall
    )

    # stamp the car (always pointing up in the aligned view) at the bottom anchor
    car_mask = pygame.surfarray.array_alpha(car.img).T > 0   # (mh, mw), base (angle 0)
    mh, mw = car_mask.shape
    oy, ox = car_row - mh // 2, 128 - mw // 2
    ys0, xs0 = max(0, oy), max(0, ox)
    ys1, xs1 = min(CROP, oy + mh), min(CROP, ox + mw)
    if ys1 > ys0 and xs1 > xs0:
        sub = car_mask[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        out[ys0:ys1, xs0:xs1][sub] = CAR_VAL

    # mark next 5 checkpoints in ego-centric space at 0.75
    # world → output: invert M (orthogonal rotation, so transpose suffices)
    for i in range(30):
        cp = CHECKPOINTS[(car.checkpoint_index + i) % len(CHECKPOINTS)]
        dx, dy = cp[0] - tx, cp[1] - ty
        cp_col = int(ca * dx - sa * dy)
        cp_row = int(sa * dx + ca * dy)
        cv2.circle(out, (cp_col, cp_row), 5, 0.75, -1)

    return out

def build_state(frames):
    grays = list(frames) 
    while len(grays) < 4:
        grays.insert(0, grays[0])
    return torch.FloatTensor(np.stack(grays[-4:]))


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

        if self.cars:
            frame = capture_frame(self.cars[0])
            self.frames.append(frame)
            # overlay the newest net input (256x256 semantic crop) in the top-right corner
            g = (frame * 255).astype(np.uint8).T          # (W, H)
            surf = pygame.surfarray.make_surface(np.repeat(g[:, :, None], 3, axis=2))
            self.win.blit(surf, (WIDTH - CROP, 0))

        pygame.display.update()

    def check_collisions(self):
        for car in self.cars:
            if car.collide(TRACK_BORDER_MASK):
                car.bounce()

        for i, car1 in enumerate(self.cars):
            for j, car2 in enumerate(self.cars):
                if i != j and car1.collide_car(car2):
                    car1.bounce()
                    car2.bounce()

    def check_finish_line(self):

        finished = []

        for car in self.cars:
            finish_poi_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_poi_collide is not None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    car.reset()
                    car.set_position((180, 200))

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
            wall_reward = 0.0
            if front < 15:
                wall_reward += (front - 15) / 25
            if front_left < 15:
                wall_reward += (front_left - 15) / 25
            if front_right < 15:
                wall_reward += (front_right - 15) / 25
            reward += wall_reward
            if wall_reward < 0:
                wall_hits += 1

            car.perform_action(action)

        return state, reward, wall_hits

    def run(self):
        """Main game loop."""
        who_finished_first      = []
        car                     = self.cars[0]
        steps_since_checkpoint   = 0
        checkpoints_this_rollout = 0
        wall_hits_this_rollout   = 0
        episode_steps            = 0
        rollout_reward           = 0.0
        actor_losses             = []
        critic_losses            = []
        rewards_history          = []
        entropy_losses           = []
        ratios                   = []
        layer_norms_history      = []
        critic_values            = []
        states_stds              = []
        checkpoints_history      = []
        action_freq_history      = []
        wall_hits_history        = []
        episode_results          = []

        self.draw()  # seed self.frames before first move_cars

        while self.running:
            checkpoint_idx_bef = car.get_progress()[0]
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            state, reward, wall_hits = self.move_cars()
            wall_hits_this_rollout += wall_hits
            self.check_collisions()
            finish_lines = self.check_finish_line()
            done = len(finish_lines) > 0 or not self.running
            if finish_lines:
                who_finished_first.append(finish_lines)
                episode_results.append((episode_steps, True))
                episode_steps = 0
                steps_since_checkpoint = 0
                self.frames.clear()
                self.draw()
                print(f'lap finished! total laps: {len(who_finished_first)}')

            self.draw()
            next_state = build_state(self.frames)

            if self.cars:
                checkpoint_idx_now = car.get_progress()[0]
                if checkpoint_idx_now != checkpoint_idx_bef:
                    steps_since_checkpoint = 0
                    checkpoints_this_rollout += 1
                else:
                    steps_since_checkpoint += 1

            cx, cy    = car.get_center()
            cp        = CHECKPOINTS[car.checkpoint_index % len(CHECKPOINTS)]
            to_cp     = (cp[0] - cx, cp[1] - cy)
            cp_dist   = math.sqrt(to_cp[0]**2 + to_cp[1]**2)
            to_cp     = (to_cp[0] / cp_dist, to_cp[1] / cp_dist)
            angle_rad = math.radians(car.angle)
            heading   = (-math.sin(angle_rad), -math.cos(angle_rad))
            cos_a     = heading[0] * to_cp[0] + heading[1] * to_cp[1]
            reward   += cos_a * car.vel / 8
            reward   -= 2.5e-2 # to musi zostac. I'd add checkpoint reweard.

            episode_steps += 1

            if steps_since_checkpoint >= 2048:
                done = True
                steps_since_checkpoint = 0
                episode_results.append((episode_steps, False))
                episode_steps = 0
                car.reset()
                car.set_position((180, 200))
                self.draw()

            rollout_reward += reward
            car.store(state, car._last_action, car._last_log_prob, reward, done)

            a_loss, c_loss, e_loss, ratio, lnorms, cval, sstd, action_freq, v_arr, r_arr = car.update(next_state.unsqueeze(0))
            if a_loss is not None:
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropy_losses.append(e_loss)
                ratios.append(ratio)
                layer_norms_history.append(lnorms)
                critic_values.append(cval)
                states_stds.append(sstd)
                checkpoints_history.append(checkpoints_this_rollout)
                action_freq_history.append(action_freq)
                wall_hits_history.append(wall_hits_this_rollout)
                checkpoints_this_rollout = 0
                wall_hits_this_rollout = 0
                rewards_history.append(rollout_reward)
                rollout_reward = 0.0
                save_plots(actor_losses, critic_losses, rewards_history, entropy_losses, ratios,
                           layer_norms_history, critic_values, states_stds, checkpoints_history,
                           action_freq_history, wall_hits_history, episode_results, v_arr, r_arr)
                print(f"Update #{len(actor_losses):4d} | actor={a_loss:.4f}  critic={c_loss:.4f}  entropy={e_loss:.4f}  ratio={ratio:.4f}  value={cval:.4f}  sstd={sstd:.4f}  checkpoints={checkpoints_history[-1]}  wall_hits={wall_hits_history[-1]}  reward={rewards_history[-1]:.3f}")

        pygame.quit()
        print("Game over!")
        print(who_finished_first)
        return who_finished_first, actor_losses, critic_losses, rewards_history


def main():
    import os
    from model import ActorCritic

    car = ActorCritic("P1", feature_dim=256, lr=2e-4)

    if os.path.exists('weights.pth'):
        car.load_state_dict(torch.load('weights.pth', map_location=car.device))
        print("Loaded weights from weights.pth")

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