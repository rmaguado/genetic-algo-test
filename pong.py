import numpy as np
from math import sqrt


class SettingsPong:
    def __init__(self, settings):
        self.screen_width = settings["screen_width"]
        self.screen_height = settings["screen_height"]
        self.paddle_size = settings["paddle_size"]
        self.paddle_speed = settings["paddle_speed"]
        self.ball_speed = settings["ball_speed"]


class Paddle:
    def __init__(self, x, y, env):
        self.env = env
        self.x = x
        self.y = y
        self.velocity_y = env.paddle_speed
        self.reset = lambda: self._reset(x, y)

    def _reset(self, x, y):
        self.x = x
        self.y = y
        self.velocity_y = self.env.paddle_speed

    def move(self):
        self.y += self.velocity_y
        if self.y <= 0:
            self.y = 0
        elif self.y >= self.env.screen_height - self.env.paddle_size:
            self.y = self.env.screen_height - self.env.paddle_size

    def action(self, action):
        if action == 0:
            self.velocity_y = -self.env.paddle_speed
        elif action == 1:
            self.velocity_y = 0
        elif action == 2:
            self.velocity_y = self.env.paddle_speed


class Ball:
    def __init__(self, x, y, env):
        self.env = env
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.reset = lambda: self._reset(x, y)
        self.init_velocity()

    def _reset(self, x, y):
        self.x = x
        self.y = y
        self.init_velocity()

    def init_velocity(self):
        self.velocity_y = self.env.ball_speed * \
            np.random.choice([-1, 1]) * np.random.rand() / 1.125
        self.velocity_x = sqrt(self.env.ball_speed ** 2 -
                               self.velocity_y ** 2) * np.random.choice([-1, 1])

    def move(self, paddle):
        self.x += self.velocity_x
        self.y += self.velocity_y

        if self.y < 0:
            self.y = 0
            self.velocity_y = -self.velocity_y
        elif self.y >= self.env.screen_height:
            self.y = self.env.screen_height - 1
            self.velocity_y = -self.velocity_y

        if self.x >= self.env.screen_width:
            self.x = self.env.screen_width - 1
            self.velocity_x = -self.velocity_x
        elif self.x < 0:
            if self.y >= paddle.y and self.y <= paddle.y + self.env.paddle_size:
                self.x = 0
                self.velocity_x = -self.velocity_x
                return 1
            else:
                return -1
        return 0


class Pong:
    def __init__(self, settings):
        self.settings = settings
        self.paddle = Paddle(0, settings.screen_height // 2, settings)
        self.ball = Ball(settings.screen_width // 2,
                         settings.screen_height // 2, settings)

    def reset(self):
        self.paddle.reset()
        self.ball.reset()

    def step(self, action):
        self.paddle.action(action)
        self.paddle.move()
        score = self.ball.move(self.paddle)
        if score < 0:
            self.reset()
        return score

    def get_state(self):
        return np.array([
            self.ball.x / self.settings.screen_width,
            self.ball.y / self.settings.screen_height,
            self.ball.velocity_x / self.settings.ball_speed,
            self.ball.velocity_y / self.settings.ball_speed,
            self.paddle.y / self.settings.screen_height
        ])

    def render(self):

        for i in range(self.settings.screen_height):
            for j in range(self.settings.screen_width):
                if i == int(self.ball.y) and j == int(self.ball.x):
                    print("O", end="")
                elif i == 0 or i == self.settings.screen_height - 1:
                    print("-", end="")
                elif j == 0 or j == self.settings.screen_width - 1:
                    print("|", end="")
                elif i >= int(self.paddle.y) and i < int(self.paddle.y) + self.settings.paddle_size and j == 1:
                    print("|", end="")
                else:
                    print(" ", end="")
            print()

        print(
            f"Paddle: {self.paddle.y} Ball: {int(self.ball.x)}, {int(self.ball.y)}")
