import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools
from sympy import Matrix, tanh, exp, atan, pi
from sympy.abc import x, y, z
from sympy import symbols
import random
import math


h1, h2, h3 = symbols("h1"), symbols("h2"), symbols("h3")

def newtons_method(f1, f2, f3, jacobian, x0, max_iters, epsilon, J, W, alpha, gain):
    eval_x = x0[0]
    eval_y = x0[1]
    eval_z = x0[2]

    for iteration in range(0, max_iters):

        # evaluate jacobian at (x)
        jac_eval = jacobian.subs([(h1, eval_x), (h2, eval_y), (h3, eval_z)])

        # compute f at (x)
        f = -1 * Matrix([f1(eval_x, eval_y, eval_z, J, W, alpha, gain), f2(eval_x, eval_y, eval_z, J, W, alpha, gain),
                         f3(eval_x, eval_y, eval_z, J, W, alpha, gain)])

        dX = jac_eval.inv() * f

        x0 = np.reshape(x0, newshape = (3,1))

        x0 += dX  # step X by dX

        eval_x = x0[0]
        eval_y = x0[1]
        eval_z = x0[2]

        if dX.norm() < epsilon:  # convergence
            print("converged on iteration {}".format(iteration))
            return x0, True

    return x0, False  #no converged point




gain = 1
alpha = 10 ** 6

"""
N = 3

#first, lets establish the Jacobian

W = np.random.normal(0, 1/N, size=(N, N))

#J, connectivity matrix
J = np.random.normal(0, 1/N, size=(N, N))
"""

J = np.array([[-0.14, 0.13, -0.62],
               [-0.45, -0.19, -1.50 ],
              [0.27, -0.28, -1.16]])

W = np.array([[0.57, -0.26, 0.95],
               [0.02, 0.27, -0.46],
              [1.0, -0.04, -0.04]])


rs = Matrix([[(-h1 + J[0,0]*tanh(gain*h1) + J[0,1]*tanh(gain*h2) + J[0,2]*tanh(gain*h3))],
            [(-h2 + J[1,0]*tanh(gain*h1) + J[1,1]*tanh(gain*h2) + J[1,2]*tanh(gain*h3))],
            [(-h3 + J[2,1]*tanh(gain*h1) + J[2,2]*tanh(gain*h2) + J[2,2]*tanh(gain*h3))]])


gd = Matrix([[((4/math.pi)*atan(tanh((W[0,0]*h1 + W[0,1]*h2 + W[0,2]*h3)/2)))/(1 + exp(-alpha*(W[0,0]*h1 + W[0,1]*h2 + W[0,2]*h3)))],
             [((4/math.pi)*atan(tanh((W[1,0]*h1 + W[1,1]*h2 + W[1,2]*h3)/2)))/(1 + exp(-alpha*(W[1,0]*h1 + W[1,1]*h2 + W[1,2]*h3)))],
             [((4/math.pi)*atan(tanh((W[2,0]*h1 + W[2,1]*h2 + W[2,2]*h3)/2)))/(1 + exp(-alpha*(W[2,0]*h1 + W[2,1]*h2 + W[2,2]*h3)))]])



Y = Matrix([h1, h2, h3])

dh_dt = gd.multiply_elementwise(rs)


jacobian = dh_dt.jacobian(Y)


def f1(h1, h2, h3, J, W, alpha, gain):
    ls = ((4 / math.pi) * atan(tanh((W[0, 0] * h1 + W[0, 1] * h2 + W[0, 2] * h3) / 2))) / (
                1 + exp(-alpha * (W[0, 0] * h1 + W[0, 1] * h2 + W[0, 2] * h3)))
    rs = (-h1 + J[0, 0] * tanh(gain * h1) + J[0, 1] * tanh(gain * h2) + J[0, 2] * tanh(gain * h3))

    return ls * rs


def f2(h1, h2, h3, J, W, alpha, gain):
    ls = ((4 / math.pi) * atan(tanh((W[1, 0] * h1 + W[1, 1] * h2 + W[1, 2] * h3) / 2))) / (
                1 + exp(-alpha * (W[1, 0] * h1 + W[1, 1] * h2 + W[1, 2] * h3)))
    rs = (-h2 + J[1, 0] * tanh(gain * h1) + J[1, 1] * tanh(gain * h2) + J[1, 2] * tanh(gain * h3))

    return ls * rs


def f3(h1, h2, h3, J, W, alpha, gain):
    ls = ((4 / math.pi) * atan(tanh((W[2, 0] * h1 + W[2, 1] * h2 + W[2, 2] * h3) / 2))) / (
                1 + exp(-alpha * (W[2, 0] * h1 + W[2, 1] * h2 + W[2, 2] * h3)))
    rs = (-h2 + J[2, 0] * tanh(gain * h1) + J[2, 1] * tanh(gain * h2) + J[2, 2] * tanh(gain * h3))

    return ls * rs


# increment
delta = [0.1, 0.01, 0.05]

fixed_points = []
alpha = 10 ** 6
gain = 2

x0 = np.array([0.01, 0.06, 0.02])

iterations = 100
current_iter= 0

while current_iter != iterations:
    vec, isConvergence = newtons_method(f1, f2, f3, jacobian, x0, max_iters=10, epsilon=1e-2, J=J, W=W, alpha=alpha,
                                        gain=gain)

    if isConvergence:
        #resample point
        if vec.norm() > 1:
            x0 = np.random.rand(3)
        else:
            fixed_points.append(np.array(vec))
            print(np.array(vec))
            current_iter += 1
            print(current_iter)

    index1, index2 = random.randint(0, 2), random.randint(0, 2)

    zeros = np.zeros(shape=(3))
    zeros[index2] = delta[index1]

    # update x0
    x0 += zeros

# let's plot the manifold

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')

for fixed in fixed_points:
    # print(fixed[0][0], fixed[1][0], fixed[2][0])

    ax.scatter(int(fixed[0][0]), int(fixed[1][0]), int(fixed[2][0]), alpha=0.5)

#ax.set_xlim3d(left=-1, right=1)
#ax.set_ylim3d(bottom=-1, top=1)
#ax.set_zlim3d(bottom=-1, top=1)

plt.show()