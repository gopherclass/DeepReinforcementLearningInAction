"""
CartPole implementation for PyTorch
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from torch import nn
from torch import optim
import math
import numpy as np
import time
import torch

class CartPoleTorch(object):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def step(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        new_state = torch.stack((x, x_dot, theta, theta_dot))

        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        ).bool().item()

        return new_state, done

    def reset(self, np_random):
        return torch.tensor(np_random.uniform(low=-0.05, high=0.05, size=(4,)))

def strategy_choice(acts, act_prob):
    return np.random.choice(acts, p=act_prob)

def strategy_max(acts, act_prob):
    return np.argmax(act_prob)

def sim(policy, env, limit=500, skip=None, seed=None, reset=None, initial_state=None, animation=False, sleep=None, strategy=strategy_choice, live_state=False, terminal=False):
    acts = np.array([0, 1])
    tries = 0
    try:
        while True:
            if seed is not None:
                env.seed(seed)
            state = env.reset()
            if reset is not None:
                initial_state = reset(env.np_random)
            if initial_state is not None:
                env.state = initial_state
                state = initial_state
            done = False
            steps = 0
            if animation:
                env.render()
            while not done and (limit is None or steps < limit):
                with torch.no_grad():
                    act_prob = policy(torch.as_tensor(state, dtype=torch.float64).unsqueeze(0)).squeeze(0).numpy()
                act = strategy(acts, act_prob)
                state, _, done, _ = env.step(act)
                steps += 1
                if live_state:
                    print('#{} => #{} {}'.format(tries, steps, format_state(state)))
                if animation:
                    if skip is None or steps % skip == 0:
                        env.render()
                if sleep is not None:
                    time.sleep(sleep)
            if animation:
                if terminal:
                    print('#{} => #{} {}'.format(tries, steps, format_state(state)))
                tries += 1
            else:
                return steps
    except KeyboardInterrupt:
        if not animation:
            raise
        env.close()

def format_state(state):
    if isinstance(state, torch.Tensor):
        state = state.data.numpy()
    x, x_dot, theta, theta_dot = state
    deg = math.degrees(theta)
    return "(x = {:+.4f}, dx = {:+.4f}, θ = {:+8.4f}, dθ = {:+.4f})".format(
        x, x_dot, deg, theta_dot)

def ConvNet0():
    def Lambda(fn):
        class LambdaModule(nn.Module):
            def forward(self, x):
                return fn(x)
        return LambdaModule()

    return nn.Sequential(
        Lambda(lambda x: x.unsqueeze(1)),
        nn.Conv1d(1, 32, 4),         # FC
        Lambda(lambda x: x.swapdims(2, 1)),
        nn.Conv1d(1, 32, 3),         # 32 - 3 + 1 = 30
        nn.Conv1d(32, 32, 3),        # 30 - 3 + 1 = 28
        nn.MaxPool1d(4, stride=2),   # (28 - 4) / 2 + 1 = 13
        nn.Conv1d(32, 16, 3),        # 15 - 3 + 1 = 11
        nn.Conv1d(16, 16, 3),        # 13 - 3 + 1 = 9
        nn.MaxPool1d(3, stride=2),   # (11 - 3) / 2 + 1 = 4
        nn.Conv1d(16, 8, 2),         # 5 - 3 + 1 = 3
        nn.Conv1d(8, 1, 2),          # 3 - 2 + 1 = 2
        Lambda(lambda x: x.squeeze(1)),
        nn.Softmax(dim=1),
    ).double()

class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv1d(1, 32, 4)    # FC
        self.l2 = nn.Sequential(
            nn.Conv1d(1, 32, 3),         # 32 - 3 + 1 = 30
            nn.Conv1d(32, 32, 3),        # 30 - 3 + 1 = 28
            nn.MaxPool1d(4, stride=2),   # (28 - 4) / 2 + 1 = 13
            nn.Conv1d(32, 16, 3),        # 15 - 3 + 1 = 11
            nn.Conv1d(16, 16, 3),        # 13 - 3 + 1 = 9
            nn.MaxPool1d(3, stride=2),   # (11 - 3) / 2 + 1 = 4
            nn.Conv1d(16, 8, 2),         # 5 - 3 + 1 = 3
            nn.Conv1d(8, 1, 2),          # 3 - 2 + 1 = 2
        )
        self.l3 = nn.Softmax(dim=1)
        self.double()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.l1(x)
        x = x.swapdims(2, 1)
        x = self.l2(x)
        x = x.squeeze(1)
        x = self.l3(x)
        return x

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv1d(1, 32, 4)    # FC
        self.l2 = nn.Sequential(
            nn.Conv1d(1, 32, 3),         # 32 - 3 + 1 = 30
            nn.GELU(),
            nn.Conv1d(32, 32, 3),        # 30 - 3 + 1 = 28
            nn.GELU(),
            nn.MaxPool1d(4, stride=2),   # (28 - 4) / 2 + 1 = 13
            nn.Conv1d(32, 16, 3),        # 15 - 3 + 1 = 11
            nn.GELU(),
            nn.Conv1d(16, 16, 3),        # 13 - 3 + 1 = 9
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),   # (11 - 3) / 2 + 1 = 4
            nn.Conv1d(16, 8, 2),         # 5 - 3 + 1 = 3
            nn.GELU(),
            nn.Conv1d(8, 1, 2),          # 3 - 2 + 1 = 2
            nn.GELU(),
        )
        self.l3 = nn.Softmax(dim=1)
        self.double()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.l1(x)
        x = x.swapdims(2, 1)
        x = self.l2(x)
        x = x.squeeze(1)
        x = self.l3(x)
        return x

class ConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv1d(1, 32, 4)    # FC
        self.l2 = nn.Sequential(
            nn.Conv1d(1, 16, 3),         # 32 - 3 + 1 = 30
            nn.GELU(),
            nn.Conv1d(16, 16, 3),        # 30 - 3 + 1 = 28
            nn.GELU(),
            nn.MaxPool1d(4, 2),   # (28 - 4) / 2 + 1 = 13
            nn.Conv1d(16, 8, 3),         # 15 - 3 + 1 = 11
            nn.GELU(),
            nn.Conv1d(8, 8, 3),          # 13 - 3 + 1 = 9
            nn.GELU(),
            nn.MaxPool1d(3, 2),   # (11 - 3) / 2 + 1 = 4
            nn.Conv1d(8, 4, 2),          # 5 - 3 + 1 = 3
            nn.GELU(),
            nn.Conv1d(4, 1, 2),          # 3 - 2 + 1 = 2
            nn.GELU(),
        )
        self.l3 = nn.Softmax(dim=1)
        self.double()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.l1(x)
        x = x.swapdims(2, 1)
        x = self.l2(x)
        x = x.squeeze(1)
        x = self.l3(x)
        return x

class ConvNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1, 1), # 16
            nn.GELU(),
            nn.Conv1d(4, 4, 3, 1, 1), # 16
            nn.GELU(),
            nn.MaxPool1d(2, 2), # 8
            nn.Conv1d(4, 2, 3, 1, 1), # 8
            nn.GELU(),
            nn.Conv1d(2, 2, 3, 1, 1), # 8
            nn.GELU(),
            nn.MaxPool1d(2, 2), # 4
            nn.Conv1d(2, 1, 3, 1, 1), # 4
            nn.GELU(),
            nn.Conv1d(1, 1, 3, 1, 1), # 4
            nn.GELU(),
            nn.MaxPool1d(2, 2), # 2
        )
        self.softmax = nn.Softmax(dim=1)
        self.double()
        
    def forward(self, x):
        x = self.fc(x).unsqueeze(1)
        x = self.conv(x).squeeze(1)
        x = self.softmax(x)
        return x

def FCNet0():
    return nn.Sequential(
        nn.Linear(4, 150),
        nn.GELU(),
        nn.Linear(150, 2),
        nn.Softmax(dim=1),
    ).double()

def FCNet1():
    return nn.Sequential(
        nn.Linear(4, 150),
        nn.GELU(),
        nn.Linear(150, 150),
        nn.GELU(),
        nn.Linear(150, 2),
        nn.Softmax(dim=1),
    ).double()

def ask(policy, state):
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.float64)
    single = state.dim() == 1
    if single:
        state = state.unsqueeze(0)
    act_prob = policy(state)
    if single:
        act_prob = act_prob.squeeze(0)
    return act_prob

def choice_act(act_prob, acts = np.array([0, 1])):
    if isinstance(act_prob, torch.Tensor):
        act_prob = act_prob.data.numpy()
    return np.random.choice(acts, p=act_prob)

