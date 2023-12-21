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



def predict_model(x, u):
    y = 0.5 *x + u**2
    return y


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


def objective(u):
    y = np.zeros((1,len(u)))
    x = 0
    last_u = 0
    trajs = 6
    for i in range(len(u)):
        x = predict_model(x, u[i])
        y[:,i] = x
    # print(f'predict_horizon:{y}')
    J = cost_function(u,y, trajs,last_u)  #, alpha, beta
    return J


if __name__ == '__main__':

    import matplotlib as mpl
    import time
    from sko.PSO import PSO

    u_horizon_num = 10

    lb = [0 for i in range(u_horizon_num)]  # + [0.1 for i in range(u_horizon_num + h_horizon_num)]
    ub = [2 for i in range(u_horizon_num)]  # + [1 for i in range(u_horizon_num + h_horizon_num)]
    a = time.time()
    # print(f'current time: {a}')
    pso = PSO(func=objective, dim=u_horizon_num, pop=50, max_iter=500, lb=lb, ub=ub,w=0.5, c1=0.5, c2=0.5)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    plt.plot(pso.gbest_y_hist)
    plt.show()
    print(time.time() - a)

