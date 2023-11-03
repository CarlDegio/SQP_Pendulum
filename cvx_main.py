import numpy as np
import gymnasium as gym
from pendulum_model import PendulumModel
import cvxpy as cp

dt3g2l = 3 * 10 / (2 * 1) * 0.05
dt = 0.05
dt3ml2 = 3 * 0.05 / (1 * 1 * 1)


class CVX_SQP:
    def __init__(self):
        self.N = 30
        self.theta_cost_weight = 1
        self.theta_dot_cost_weight = 0.1
        self.u_cost_weight = 0.001
        self.theta = np.zeros(self.N)
        self.theta_dot = np.zeros(self.N)
        self.u = np.zeros(self.N)
        self.lambda_vec = np.zeros(self.N*8+2)
        self.delta_theta = cp.Variable(self.N)
        self.delta_theta_dot = cp.Variable(self.N)
        self.delta_u = cp.Variable(self.N)
        self.slack_var_1 = cp.Variable(self.N)
        self.slack_var_2 = cp.Variable(self.N)

        self.target_value=[]

    def set_init_traj(self, model_log):
        self.theta = model_log['theta']
        self.theta_dot = model_log['theta_dot']
        self.u = model_log['u']

    def solve_once(self):
        cost = 0
        constr = []
        constr += [self.delta_theta[0] == 0]
        constr += [self.delta_theta_dot[0] == 0]
        for i in range(0, self.N - 1):

            cost += self.theta_cost_weight * cp.square(self.delta_theta[i]) + \
                    self.theta_dot_cost_weight * cp.square(self.delta_theta_dot[i]) + \
                    self.u_cost_weight * cp.square(self.delta_u[i]) + \
                    0.5 * self.lambda_vec[2+8*i] * cp.square(self.delta_theta[i]) * (-dt3g2l * np.sin(self.theta[i])) + \
                    self.theta_cost_weight * self.theta[i] * self.delta_theta[i] + \
                    self.theta_dot_cost_weight * self.theta_dot[i] * self.delta_theta_dot[i] + \
                    self.u_cost_weight * self.u[i] * self.delta_u[i]
                    # 0.1*cp.square(self.slack_var_1[i])+0.1*cp.square(self.slack_var_2[i])
            constr += [dt3g2l * np.cos(self.theta[i]) * self.delta_theta[i] +
                       self.delta_theta_dot[i] + dt3ml2 * self.delta_u[i] - self.delta_theta_dot[i + 1]
                       == -(
                    -self.theta_dot[i + 1] + self.theta_dot[i] + dt3g2l * np.sin(self.theta[i]) +
                    dt3ml2 * self.u[i]
            ),
                       self.theta[i + 1] + self.delta_theta[i + 1] == self.theta[i] + self.delta_theta[i] + dt * (
                               self.theta_dot[i] + self.delta_theta_dot[i]),
                       self.theta_dot[i] + self.delta_theta_dot[i] <= 8,
                       self.theta_dot[i] + self.delta_theta_dot[i] >= -8,
                       self.u[i] + self.delta_u[i] <= 2,
                       self.u[i] + self.delta_u[i] >= -2,
                       self.delta_u[i] <= 0.1,
                       self.delta_u[i] >= -0.1,
                       ]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        print("status:", problem.status)
        print("optimal value", problem.value)
        print("optimal var: delta_theta", self.delta_theta.value)
        print("optimal var: delta_theta_dot", self.delta_theta_dot.value)
        print("optimal var: delta_u", self.delta_u.value)
        self.target_value.append(problem.value)
        for i in range(len(problem.constraints)):
            self.lambda_vec[i] = problem.constraints[i].dual_value

    def solve(self):
        for i in range(30):
            self.solve_once()
            self.theta += self.delta_theta.value
            self.theta_dot += self.delta_theta_dot.value
            self.u += self.delta_u.value
        print(self.target_value)


def make_env(name):
    gym_env = gym.make(name, render_mode="human")
    return gym_env


def main():
    env = make_env("Pendulum-v1")
    observation, info = env.reset(seed=1)
    print(observation)
    model = PendulumModel()
    model.reset(observation)
    print(model.state)
    model_log = {'theta': [], 'theta_dot': [], 'u': []}
    for i in range(30):
        model_log['theta'].append(model.state[0])
        model_log['theta_dot'].append(model.state[1])
        action = np.random.uniform(-2, 2, 1)
        # action=np.array([0])
        model.step(action)
        model_log['u'].append(action)

    model_log['theta'] = np.hstack(model_log['theta'])
    model_log['theta_dot'] = np.hstack(model_log['theta_dot'])
    model_log['u'] = np.hstack(model_log['u'])
    cvx_sqp = CVX_SQP()
    cvx_sqp.set_init_traj(model_log)
    cvx_sqp.solve()
    control = cvx_sqp.u
    for i in range(200):
        observation, reward, terminated , truncated , info = env.step(control[i].reshape(1,))
        print(observation, reward, control[i])


if __name__ == "__main__":
    main()
