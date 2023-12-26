# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import matplotlib as mpl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 不使用GPU

# mpl.use('TkAgg')
mpl.rcParams['lines.linewidth'] = 1.5

# 设置新罗马字体
fontsize = 18
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize, }



def predict_model(x, u0,u_k):
    # u_k = u0 + u_k     # 两种求解方法
    y = 0.5 * x + u_k**2
    return y,u_k

# @tf.function
def cost_function(u_array,predict_horizon, trajs,last_u,H):
    # u  : 1 * dimension
    # trajs : 1 * dimension or constant
    # alpha : 1 * dimension
    # beta  : 1 * dimension
    # predict_horizon: 1 * dimension
    # if len(self.trajs) != predict_horizon:
    # alpha = np.ones_like(predict_horizon)#alpha.reshape(1,-1)
    # alpha[0,0] = 10
    # beta = np.ones_like(predict_horizon)#beta.reshape(1, -1)
    beta = tf.constant([0.9 * i for i in range(1,H+1)])
    beta = tf.expand_dims(beta,0)# beta.reshape(1, -1)
    alpha = beta * 1
    # beta[0,1:] = 10
    trajs = tf.constant(np.ones_like(predict_horizon) * trajs)
    dy = trajs - predict_horizon
    J1 = tf.transpose(dy) * alpha @ dy
    # print(predict_horizon)
    # J2 = beta[:self.u_horizon] * u[:self.u_horizon] @ u[:self.u_horizon].T
    last_u = tf.expand_dims(last_u, 0)
    u_horizon = tf.concat((tf.expand_dims(last_u,0),tf.expand_dims(u_array[:-1,0],0)),1)
    delta_u = tf.transpose(u_array) - u_horizon#np.array([u - u_horizon])
    J2 = beta[:,:H] * delta_u @  tf.transpose(delta_u)
    cost_J =  J1 + J2
    # J1 + tf.transpose(u_array) @ u_array  #
    return cost_J[0,0]


def multiplypredict(u_array,last_x,last_u,trajs,H):
    y = np.zeros((1,H))
    u0 = last_u
    x = last_x
    temp = tf.ones((1,1))
    for i in range(H):
        x,u0 = predict_model(x, u0,u_array[i,0])
        temp = tf.concat((temp, tf.expand_dims(tf.expand_dims(x,0),0)), 0)
    y = tf.convert_to_tensor(temp[1:],dtype=tf.float32)
    J = cost_function(u_array,y, trajs,last_u,H)  #, alpha, beta
    return J,y


def solve_ocp(u,last_x,last_u,trajs, H,iterations=1000, tol=1e-8):
    J_prev = -1
    optimizer = tf.keras.optimizers.RMSprop(lr=0.05)
    au = tf.Variable(initial_value=u, name='u',trainable=True,dtype=tf.float32)
    for epoch in range(iterations):
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(u)
            J, x_pred = multiplypredict(au, last_x, last_u, trajs,H)
            J = tf.cast(J,dtype=tf.float32)
            x_pred = tf.cast(x_pred,dtype=tf.float32)
        # gradients1 = tape.jacobian(x_pred, au)
        gradients = tape.gradient(J, au)
        optimizer.apply_gradients(zip([gradients], [au]))

        ensure_constraints(au, ub, lb)
        # print(au,J)

        if np.abs(J - J_prev) < tol:
            return J, x_pred
        J_prev = J
    for i in range(H):
        last_x = last_x + au[i,0]
        print(last_x)

    return J, x_pred

def ensure_constraints(u,u_ub,u_lb):
    for i, u_ub_i in enumerate(u_ub):
        if u[i, 0] > u_ub_i:
            # u[0, i]=u_ub_i
            u[i, 0].assign(u_ub_i)

    for i, u_lb_i in enumerate(u_lb):
        if u[i, 0] < u_lb_i:
            u[i, 0].assign(u_lb_i)
            # u[0, i]=u_lb_i


if __name__ == '__main__':

    import matplotlib as mpl
    import time
    from scipy.optimize import minimize



    u_horizon_num = 20

    lb = np.array([0 for i in range(u_horizon_num)])  # + [0.1 for i in range(u_horizon_num + h_horizon_num)]
    ub = np.array([2 for i in range(u_horizon_num)])  # + [1 for i in range(u_horizon_num + h_horizon_num)]
    # last_x = tf.constant(0)
    # last_u= tf.constant(0)
    # trajs = tf.constant(6)
    last_x = tf.constant(0,dtype=tf.float32)
    last_u= tf.constant(0,dtype=tf.float32)
    trajs = tf.constant(6,dtype=tf.float32)
    t0 = time.time()
    u_hat = tf.ones((u_horizon_num,1))
    J, x_pred = solve_ocp(u_hat, last_x, last_u, trajs,u_horizon_num, iterations=1000, tol=1e-8)
    print(J, x_pred)

    t1 = time.time()
    print('Runtime: %.2f s' % (t1 - t0))



