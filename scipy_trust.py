import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import gymnasium as gym
import imageio
from pendulum_model import PendulumModel

# Constants
g = 10.0  # gravitational acceleration
l = 1.0  # length of the pendulum
m = 1.0  # mass of the pendulum
dt = 0.05  # time step
n = 100  # number of time steps
x0 = np.zeros(3 * n)


class ScipySolver:
    def __init__(self):
        self.theta0 = np.pi
        self.theta_dot0 = 0.0

    @staticmethod
    def objective(x):
        theta = x[:n]
        theta_dot = x[n:2 * n]
        u = x[2 * n:]

        cost = np.sum(theta ** 2) + 0.1 * np.sum(theta_dot ** 2) + 0.001 * np.sum(u ** 2)
        return cost

    def dynamics(self, x):
        theta = x[:n]
        theta_dot = x[n:2 * n]
        u = x[2 * n:]

        constraints = []

        constraints.append(theta[0] - self.theta0)
        constraints.append(theta_dot[0] - self.theta_dot0)

        for t in range(n - 1):
            constraints.append(theta_dot[t + 1] - (
                    theta_dot[t] + (3 * g * dt) / (2 * l) * np.sin(theta[t]) + (3 * dt) / (m * l ** 2) * u[t]))
            constraints.append(theta[t + 1] - (theta[t] + theta_dot[t+1] * dt))

        return np.array(constraints)

    @staticmethod
    def plot_results(theta, theta_dot, u):
        time = np.linspace(0, dt * n, n)

        plt.figure(figsize=(10, 8))

        # Plot theta
        plt.subplot(3, 1, 1)
        plt.plot(time, theta)
        plt.ylabel('Theta (rad)')
        plt.title('Optimal Control Results')

        # Plot theta_dot
        plt.subplot(3, 1, 2)
        plt.plot(time, theta_dot)
        plt.ylabel('Theta_dot (rad/s)')

        # Plot u
        plt.subplot(3, 1, 3)
        plt.plot(time, u)
        plt.ylabel('Control Input (u)')
        plt.xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def solve(self):
        env = make_env("Pendulum-v1")
        observation, info = env.reset(seed=4)
        model = PendulumModel()
        model.reset(observation)
        model_log = []

        self.theta0 = model.state[0]
        self.theta_dot0 = model.state[1]
        x0[:n] = np.linspace(self.theta0, 0, n)
        # x0[:n] = self.theta0
        x0[n:2 * n] = np.linspace(self.theta_dot0, 0, n)
        # x0[n:2] = self.theta_dot0

        for i in range(100):
            model_log.append(model.state)
            action = np.array([0.0])
            model.step(action)
            model_log.append(action)
        model_log = np.hstack(model_log)

        # Initial guess

        # Bounds
        theta_dot_bounds = (-8, 8)
        u_bounds = (-2, 2)
        bounds = [(None, None)] * n + [theta_dot_bounds] * n + [u_bounds] * n

        # Constraints
        constraints = {'type': 'eq', 'fun': self.dynamics}

        # Optimize
        result = minimize(self.objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                          options={'gtol': 1e-5})

        print(result)

        if result.success:
            theta_opt = result.x[:n]
            theta_dot_opt = result.x[n:2 * n]
            u_opt = result.x[2 * n:]
            print(theta_opt)
            print(theta_dot_opt)
            print(u_opt)
        else:
            print("Optimization failed.")
            theta_opt = result.x[:n]
            theta_dot_opt = result.x[n:2 * n]
            u_opt = result.x[2 * n:]
            print(theta_opt)
            print(theta_dot_opt)
            print(u_opt)

        frames = []
        for i in range(100):
            observation, reward, terminated, truncated, info = env.step(u_opt[i].reshape(1, ))
            print(observation, reward, u_opt[i])
            frame = env.render()
            frames.append(frame)

        imageio.mimsave('pendulum_run.gif', frames, duration=1.0 / 20)
        self.plot_results(theta_opt, theta_dot_opt, u_opt)


def make_env(name):
    gym_env = gym.make(name, render_mode='rgb_array')
    return gym_env


if __name__ == "__main__":
    scipy_solve = ScipySolver()
    scipy_solve.solve()
