import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 不使用GPU

# mpl.use('TkAgg')
mpl.rcParams['lines.linewidth'] = 1.5

# 设置新罗马字体
fontsize = 18
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }



def predict_model(x, u0,u):
    # u = u0 + u     # 两种求解方法
    y = 0.5 *x + u**2
    return y,u


def cost_function(u,predict_horizon, trajs,last_u):
    # u  : 1 * dimension
    # trajs : 1 * dimension or constant
    # alpha : 1 * dimension
    # beta  : 1 * dimension
    # predict_horizon: 1 * dimension
    # if len(self.trajs) != predict_horizon:
    # alpha = np.ones_like(predict_horizon)#alpha.reshape(1,-1)
    # alpha[0,0] = 10
    # beta = np.ones_like(predict_horizon)#beta.reshape(1, -1)
    beta = np.array([0.1 * i for i in range(1,len(predict_horizon)+1)]).reshape(1,-1)  # beta.reshape(1, -1)
    alpha = beta
    # beta[0,1:] = 10
    trajs = np.ones_like(predict_horizon) * trajs
    J1 = alpha * (trajs - predict_horizon) @ (trajs - predict_horizon).T
    # print(predict_horizon)
    # J2 = beta[:self.u_horizon] * u[:self.u_horizon] @ u[:self.u_horizon].T

    u_horizon = np.append(last_u,u[:-1])
    delta_u = np.array([u - u_horizon])
    J2 = beta[:,:len(u)] * delta_u @  delta_u.T
    cost_J = J1 + J2
    return cost_J


def objective(u,x,last_u,trajs):
    y = np.zeros((1,len(u)))
    u0 = 0
    for i in range(len(u)):
        x,u0 = predict_model(x, u0,u[i])
        y[:,i] = x
    # print(f'predict_horizon:{y}')
    J = cost_function(u,y, trajs,last_u)  #, alpha, beta
    return J


if __name__ == '__main__':

    import matplotlib as mpl
    import time
    from scipy.optimize import minimize

    u_horizon_num = 10

    lb = [0 for i in range(u_horizon_num)]  # + [0.1 for i in range(u_horizon_num + h_horizon_num)]
    ub = [2 for i in range(u_horizon_num)]  # + [1 for i in range(u_horizon_num + h_horizon_num)]
    x=0
    last_u=0
    trajs=6
    t0 = time.time()
    u_hat = np.ones((u_horizon_num))
    solution = minimize(objective, u_hat, method='SLSQP', args=(x,last_u,trajs), options={'eps': 1e-08})
    t1 = time.time()
    print('Runtime: %.2f s' % (t1 - t0))
    print(solution)


