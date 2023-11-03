import gymnasium as gym
import numpy as np
from scipy import sparse
import osqp
from pendulum_model import PendulumModel

dt3g2l = 3 * 10 / (2 * 1) * 0.05
dt = 0.05
dt3ml2 = 3 * 0.05 / (1 * 1 * 1)


class SQP:
    def __init__(self):
        self.x0 = np.zeros(3)
        self.N = 50
        self.constraints_num = (self.N - 1) * 2 + 7 * self.N + 2
        self.Q = sparse.kron(sparse.eye(self.N), np.diag([1, 0.1, 0.001]), format="csc")

        self.X = np.zeros((self.N * 3))
        self.P_mat = self.Q
        self.q_vec = np.zeros(self.N * 3)
        self.A_mat = np.zeros((self.constraints_num, self.N * 3))
        self.l_bound = np.zeros(self.constraints_num)
        self.u_bound = np.zeros(self.constraints_num)
        self.lambda_vec = np.zeros(self.constraints_num)

    def set_init_traj(self, x0):
        self.x0 = x0
        self.X = self.x0

    def set_init_point(self, x0):
        self.x0 = x0
        self.X[0:2] = self.x0

    def cost_value(self, x):
        return x @ self.Q @ x

    def solve_once(self):
        controller = osqp.OSQP()
        self.P_mat = 2 * self.Q
        for i in range(self.N):
            self.P_mat[i * 3, i * 3] += -dt3g2l * self.lambda_vec[i] * np.sin(self.X[i * 3])
        self.q_vec = 2 * self.Q @ self.X

        dynamic_constraint = np.zeros((self.N - 1, self.N * 3))
        dynamic_raw = np.zeros(self.N - 1)
        kinetic_constraint = np.zeros((self.N - 1, self.N * 3))
        kinetic_raw = np.zeros(self.N - 1)
        for i in range(self.N - 1):
            dynamic_constraint[i, i * 3:i * 3 + 5] = np.array([dt3g2l * np.cos(self.X[i * 3]), 1.0, dt3ml2, 0, -1])
            dynamic_raw[i] -= (-self.X[(i + 1) * 3 + 1] + self.X[i * 3 + 1] +
                               dt3g2l * np.sin(self.X[i * 3]) + dt3ml2 * self.X[i * 3 + 2])
            kinetic_constraint[i, i * 3:i * 3 + 4] = np.array([1, dt, 0, -1])
            kinetic_raw[i] -= -self.X[(i + 1) * 3] + self.X[i * 3] + dt * self.X[i * 3 + 1]

        theta_up_constraint = np.zeros((self.N, self.N * 3))
        theta_up_raw = np.zeros(self.N)
        theta_down_constraint = np.zeros((self.N, self.N * 3))
        theta_down_raw = np.zeros(self.N)
        theta_dot_up_constraint = np.zeros((self.N, self.N * 3))
        theta_dot_up_raw = np.zeros(self.N)
        theta_dot_down_constraint = np.zeros((self.N, self.N * 3))
        theta_dot_down_raw = np.zeros(self.N)
        control_up_constraint = np.zeros((self.N, self.N * 3))
        control_up_raw = np.zeros(self.N)
        control_down_constraint = np.zeros((self.N, self.N * 3))
        control_down_raw = np.zeros(self.N)
        control_trust_region_constraint = np.zeros((self.N, self.N * 3))
        control_trust_region_raw = np.ones(self.N) * 0.5
        for i in range(self.N):
            theta_up_constraint[i, i * 3] = 1
            theta_down_constraint[i, i * 3] = -1
            theta_dot_up_constraint[i, i * 3 + 1] = 1
            theta_dot_down_constraint[i, i * 3 + 1] = -1
            control_up_constraint[i, i * 3 + 2] = 1
            control_down_constraint[i, i * 3 + 2] = -1
            control_trust_region_constraint[i, i * 3 + 2] = 1

            theta_up_raw[i] -= self.X[i * 3] - 10
            theta_down_raw[i] -= -self.X[i * 3] - 10
            theta_dot_up_raw[i] -= self.X[i * 3 + 1] - 8
            theta_dot_down_raw[i] -= -self.X[i * 3 + 1] - 8
            control_up_raw[i] -= self.X[i * 3 + 2] - 2
            control_down_raw[i] -= -self.X[i * 3 + 2] - 2

        initial_constraint = np.zeros((2, self.N * 3))
        initial_constraint[0, 0] = 1
        initial_constraint[1, 1] = 1
        self.A_mat = np.vstack((dynamic_constraint, kinetic_constraint,
                                theta_up_constraint, theta_down_constraint,
                                theta_dot_up_constraint, theta_dot_down_constraint,
                                control_up_constraint, control_down_constraint,
                                control_trust_region_constraint, initial_constraint))
        self.A_mat = sparse.csc_matrix(self.A_mat)

        n_inf = np.inf * np.ones(self.N)
        self.l_bound = np.hstack((dynamic_raw, kinetic_raw,
                                  -n_inf, -n_inf,
                                  -n_inf, -n_inf,
                                  -n_inf, -n_inf,
                                  -control_trust_region_raw, np.zeros(2)))
        self.u_bound = np.hstack((dynamic_raw, kinetic_raw,
                                  theta_up_raw, theta_down_raw,
                                  theta_dot_up_raw, theta_dot_down_raw,
                                  control_up_raw, control_down_raw,
                                  control_trust_region_raw, np.zeros(2)))

        controller.setup(self.P_mat, self.q_vec, self.A_mat, self.l_bound, self.u_bound,
                         warm_start=True, verbose=True, max_iter=1,rho=1e-3)
        controller.warm_start(x=np.zeros_like(self.X))
        result = controller.solve()
        print(result.x)
        print(0.5*result.x@self.P_mat@result.x + self.q_vec@result.x)

    def solve(self):
        pass


def make_env(name):
    gym_env = gym.make(name, render_mode="human")
    return gym_env


def main():
    env = make_env("Pendulum-v1")
    observation, info = env.reset(seed=1)
    model = PendulumModel()
    model.reset(observation)
    model_log = []
    for i in range(50):
        model_log.append(model.state)
        action = np.array([0.0])
        model.step(action)
        model_log.append(action)
    model_log = np.hstack(model_log)

    sqp = SQP()
    sqp.set_init_traj(model_log)
    sqp.solve_once()

    # terminated = False
    # truncated = False
    # tick = 0
    # while tick < 300:
    #     action = np.random.rand(1) * 4 - 2
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     obs_, reward_, _, _, _ = model.step(action)
    #     tick += 1


if __name__ == "__main__":
    main()
