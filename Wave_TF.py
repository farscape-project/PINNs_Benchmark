#!/usr/bin/env python
# coding: utf-8

"""
---------------------------------------------------------------------------------------
Physics Informed Neural Networks (PINNs) -2D Wave Equation - TensorFlow
---------------------------------------------------------------------------------------
Training Neural Network to converge towards a well-defined solution of a PDE by way of 
 minimising for the residuals across the spatio-temporal domain.
Initial and Boundary conditions are met by introducing them into the loss function along
 with the PDE residuals.

**Using TensorFlow**

Equation:
-----------------------------------------------------------------------------------------
u_tt = u_xx + u_yy on [-1,1] x [-1,1]

Dirichlet Boundary Conditions :
u=0
#
Initial Distribution :
u(x,y,t=0) = exp(-40(x-4)^2 + y^2)

Initial Velocity Condition :
u_t(x,y,t=0) = 0

m*N layers for mth order PDE
-----------------------------------------

Parameter changes to play with:
--------------------------------------------------------------------------------------
CommandLineArgs class gives 7 parameters that can be changed to edit performance
3 are sample sizes for training 
3 are domain specific
1 is for training loops

Can also change size of NN by changing PINN class

Note
-------------------------------------------------------------------------------------
Building the numerical solution by solving the Wave Equation using a spectral solver 
implemented on numpy.
Numerical Method - Spectral Solver using FFT with solution code from Boston University 
with their permission

The numerical solution will not form the training data but will be used for comparing
 against the PINN solution.
"""

import os
import sys
import time
import argparse

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from pyDOE import lhs
import tensorflow as tf

__author__ = "Lucy Harris, Vignesh Gopakumar"
__license__ = "GPL 2"
__email__ = "lucy.harris@ukaea.uk"

class CommandLineArgs:
    """
    Take arguments from command line

    Parameters:
    ---------------------------------------------------------
    Epochs                 (-E, --epochs,    default=20000)
    Inital Sampling No.    (-I, --inital,    default=1000)
    Boundary Sampling No.  (-B, --boundary,  default=1000)
    Domain Sampling No.    (-D, --domain,    default=20000)
    Spatial Discretisation (-N, --n-steps,   default=50)
    Simulation Time (s)    (-T, --time,      default=1)
    Grid size              (-G, --grid-size, default=50)
    """

    def __init__(self):
        pinn_parser = argparse.ArgumentParser(
            description="PINN solver for Schrödinger's wave equation",
            fromfile_prefix_chars="@",
            allow_abbrev=False,
            epilog="Enjoy the program! :)",
        )
        pinn_parser.add_argument(
            "-E",
            "--epochs",
            action="store",
            type=int,
            default=20000,
            help="set number of epochs for training",
        )

        pinn_parser.add_argument(
            "-I",
            "--initial",
            action="store",
            type=int,
            default=1000,
            help="set no. initial samples N_i",
        )

        pinn_parser.add_argument(
            "-B",
            "--boundary",
            action="store",
            type=int,
            default=1000,
            help="set no. boundary samples N_b",
        )

        pinn_parser.add_argument(
            "-D",
            "--domain",
            action="store",
            type=int,
            default=20000,
            help="set no. domain samples N_f",
        )
        pinn_parser.add_argument(
            "-N",
            "--n-steps",
            action="store",
            type=int,
            default=30,
            help="set spatial discretisation of domain",
        )

        pinn_parser.add_argument(
            "-T",
            "--time",
            action="store",
            type=int,
            default=1,
            help="set simulation time of domain",
        )

        pinn_parser.add_argument(
            "-G",
            "--grid-size",
            action="store",
            type=int,
            default=50,
            help="set grid size for domain",
        )

        self.args = pinn_parser.parse_args()
        print("Command grid_sizeine Arguments: ", vars(self.args))
        sys.stdout.flush()


class WaveEquation:
    """
    Numerical method for Schrödinger's 2D wave equation with 
    spectral solver using FFT.
    Code from Boston University with permission

    Parameters:
    -----------------------------------------------------------
    spatial_discretisation (int) : size of discrete resolution in domain
    simulation_time (int) : number of seconds of simulation
    grid_size (int) : full size of sample grid domain

    Return:
    -----------------------------------------------------------
    numerical solution of output u (array)
    """

    def __init__(self, spatial_discretisation, simulation_time, grid_size):
        self.spatial_discretisation = spatial_discretisation
        self.simulation_time = simulation_time
        self.grid_size = grid_size
        self.x0 = -1.0
        self.xf = 1.0
        self.y0 = -1.0
        self.yf = 1.0
        self.initialization()
        self.initCond()

    def initialization(self):
        k = np.arange(self.spatial_discretisation + 1)
        self.x = np.cos(k * np.pi / self.spatial_discretisation)
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.dt = 6 / self.spatial_discretisation**2
        self.plotgap = round((1 / 3) / self.dt)
        self.dt = (1 / 3) / self.plotgap

    def initCond(self):
        self.vv = np.exp(-40 * ((self.xx - 0.4) ** 2 + self.yy**2))
        self.vvold = self.vv.copy()

    def solve(self):

        u_list = []

        tc = 0
        nstep = round(self.simulation_time / self.dt) + 1

        while tc < nstep:

            xxx = np.linspace(self.x0, self.xf, self.grid_size)

            yyy = np.linspace(self.y0, self.yf, self.grid_size)
            vvv = interpolate.interp2d(self.x, self.y, self.vv, kind="cubic")
            Z = vvv(xxx, yyy)

            uxx = np.zeros(
                (self.spatial_discretisation + 1, self.spatial_discretisation + 1)
            )
            uyy = np.zeros(
                (self.spatial_discretisation + 1, self.spatial_discretisation + 1)
            )
            ii = np.arange(1, self.spatial_discretisation)

            for i in range(1, self.spatial_discretisation):
                v = self.vv[i, :]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real

                r1 = np.arange(self.spatial_discretisation)
                r2 = 1j * np.hstack((r1, 0, -r1[:0:-1])) * U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.spatial_discretisation + 1)
                s2 = np.hstack((s1, -s1[self.spatial_discretisation - 1 : 0 : -1]))
                s3 = -(s2**2) * U
                W2 = np.fft.ifft(s3)
                W2 = W2.real

                uxx[i, ii] = W2[ii] / (1 - self.x[ii] ** 2) - self.x[ii] * W1[ii] / (
                    1 - self.x[ii] ** 2
                ) ** (3 / 2)

            for j in range(1, self.spatial_discretisation):
                v = self.vv[:, j]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real

                r1 = np.arange(self.spatial_discretisation)
                r2 = 1j * np.hstack((r1, 0, -r1[:0:-1])) * U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.spatial_discretisation + 1)
                s2 = np.hstack((s1, -s1[self.spatial_discretisation - 1 : 0 : -1]))
                s3 = -(s2**2) * U
                W2 = np.fft.ifft(s3)
                W2 = W2.real

                uyy[ii, j] = W2[ii] / (1 - self.y[ii] ** 2) - self.y[ii] * W1[ii] / (
                    1 - self.y[ii] ** 2
                ) ** (3 / 2)

            vvnew = 2 * self.vv - self.vvold + self.dt**2 * (uxx + uyy)
            self.vvold = self.vv.copy()
            self.vv = vvnew.copy()
            tc += 1

            u_list.append(Z)
        return np.asarray(u_list)


class NumericalSol:
    """
    Generating numerical solution for Schrödinger's 2D wave equation

    Parameters:
    --------------------------------------------------------------------------
    spatial_discretisation (int) : size of discrete resolution
    simulation_time (int) : number of seconds of simulation
    grid_size (int) : full size of sample grid

    Public variable:
    --------------------------------------------------------------------------
    dictionary of solution space:
    x, y, t, upper bound, lower bound, and u solution

    Returns:
    --------------------------------------------------------------------------
    numerical solution of u (array)

    """

    def __init__(self, spatial_discretisation, simulation_time, grid_size):
        self.spatial_discretisation = spatial_discretisation
        self.simulation_time = simulation_time
        self.grid_size = grid_size  # length of array

    def solve_numerical(self):
        simulator = WaveEquation(
            self.spatial_discretisation, self.simulation_time, self.grid_size
        )
        self.u_sol = simulator.solve()

        lb = np.asarray([-1.0, -1.0, 0])  # [x, y, t] Lower Bounds of the domain
        ub = np.asarray([1.0, 1.0, self.simulation_time])  # Upper Bounds of the domain

        dt = (
            6 / self.spatial_discretisation**2
        )  # spatial_discretisation and dt are fixed for ensuring numerical stability.

        lb = np.asarray([-1.0, -1.0, 0])  # [x, y, t] Lower Bounds of the domain
        ub = np.asarray([1.0, 1.0, self.simulation_time])  # Upper Bounds of the domain

        x = np.linspace(-1, 1, self.grid_size)
        y = x.copy()
        t = np.arange(lb[2], ub[2] + dt, dt)

        U_sol = self.u_sol

        # Storing the problem and solution information.
        self.sol_dict = {
            "x": x,
            "y": y,
            "t": t,
            "lower_range": lb,
            "upper_range": ub,
            "U_sol": U_sol,
        }

        return self.u_sol


class PINN(tf.keras.Model):
    """
    Creating neural network model

    Size:
    --------------------------------------------------------------------------
    3 inputs
    4 layers with 100 hidden nodes each
    1 output

    Activation always Tanh

    Returns:
    --------------------------------------------------------------------------
    Neural network model

    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.output_layer(x)

        return x


class LossFunctions:
    """
    Calculation of loss functions for PINN

    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    model (tf.keras.Model) : NN representation

    Returns:
    --------------------------------------------------------------------------
    LHS_Sampling function -> Collocation point sampling (array)
    other functions -> return initial, boundary, pde, recostruction loss
    type is tf Tensor
    """

    def __init__(self, simulation_time, model):
        self.x_range = [-1.0, 1.0]
        self.y_range = [-1.0, 1.0]
        self.t_range = [0.0, simulation_time]
        self.D = 1.0

        self.model = model

        self.lb = np.asarray(
            [self.x_range[0], self.y_range[0], self.t_range[0]]
        )  # lower bounds
        self.ub = np.asarray(
            [self.x_range[1], self.y_range[1], self.t_range[1]]
        )  # Upper bounds

    def LHS_Sampling(self, sample_size):
        """
        Function to sample collocation points across the spatio-temporal domain 
        using a Latin Hypercube
        """
        return self.lb + (self.ub - self.lb) * lhs(3, sample_size)

    @tf.function
    def pde(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        u = self.model(tf.concat([x, y, t], 1))

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        pde_loss = u_tt - self.D * (u_xx + u_yy)

        return tf.reduce_mean(tf.square(pde_loss))

    def boundary(self, X):
        u = self.model(X)
        bc_loss = u - 0
        return tf.reduce_mean(tf.square(bc_loss))

    @tf.function
    def initial_velocity(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        u = self.model(tf.concat([x, y, t], 1))

        u_t = tf.gradients(u, t)[0]
        initial_cond_loss = u_t

        return tf.reduce_mean(tf.square(initial_cond_loss))

    def reconstruction(self, X, Y):
        u = self.model(X)
        recon_loss = u - Y
        return tf.reduce_mean(tf.square(recon_loss))


class DataPrep:
    """
    Preparing data for training

    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    u_sol (array) : numerical solution of wave
    sol_dict (dict) : dictionary of numerical solution inputs and output
    sample_dict (dict) : dictionary of sample sizes, Ni, Nb, Nf
    model (tf.Keras.Model) : NN representation

    Returns:
    --------------------------------------------------------------------------
    data_list (list) : prepared sizes of inputs and outputs ready for training

    """

    def __init__(self, simulation_time, u_sol, sol_dict, sample_dict, model):
        self.u_sol = u_sol
        self.x = sol_dict["x"]
        self.y = sol_dict["y"]
        self.t = sol_dict["t"]

        self.grid_length = len(self.x)

        # Samples taken from each region for optimisation purposes.
        self.N_i = sample_dict["N_i"]  # Initial
        self.N_b = sample_dict["N_b"]  # Boundary
        self.N_f = sample_dict["N_f"]  # Domain

        self.loss_fnc = LossFunctions(simulation_time, model)

    def prep_io(self):
        self.u = np.asarray(self.u_sol)
        X, Y = np.meshgrid(self.x, self.y)
        self.XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

        T_star = np.expand_dims(np.repeat(self.t, len(self.XY_star)), 1)
        X_star_tiled = np.tile(self.XY_star, (len(self.t), 1))

        self.X_star = np.hstack((X_star_tiled, T_star))
        self.u_actual = np.expand_dims(self.u.flatten(), 1)

    def prep_initial(self):
        X_IC = np.hstack(
            (self.XY_star, np.zeros(len(self.XY_star)).reshape(len(self.XY_star), 1))
        )
        u_IC = self.u[0].flatten()
        u_IC = np.expand_dims(u_IC, 1)

        idx = np.random.choice(X_IC.shape[0], self.N_i, replace=False)
        self.X_i = X_IC[idx]
        self.u_i = u_IC[idx]

    def prep_boundary(self):
        X_left = self.loss_fnc.LHS_Sampling(self.N_b)
        X_left[:, 0:1] = self.loss_fnc.x_range[0]

        X_right = self.loss_fnc.LHS_Sampling(self.N_b)
        X_right[:, 0:1] = self.loss_fnc.x_range[1]

        X_bottom = self.loss_fnc.LHS_Sampling(self.N_b)
        X_bottom[:, 1:2] = self.loss_fnc.y_range[0]

        X_top = self.loss_fnc.LHS_Sampling(self.N_b)
        X_top[:, 1:2] = self.loss_fnc.y_range[1]

        self.X_b = np.vstack((X_right, X_top, X_left, X_bottom))
        np.random.shuffle(self.X_b)

    def prep_domain(self):
        self.X_f = self.loss_fnc.LHS_Sampling(self.N_f)

    def convert_tensors(self):
        self.X_i = tf.convert_to_tensor(self.X_i, dtype=tf.float32)
        self.Y_i = tf.convert_to_tensor(self.u_i, dtype=tf.float32)
        self.X_b = tf.convert_to_tensor(self.X_b, dtype=tf.float32)
        self.X_f = tf.convert_to_tensor(self.X_f, dtype=tf.float32)

    def prepare(self):
        self.prep_io()
        self.prep_initial()
        self.prep_boundary()
        self.prep_domain()
        self.convert_tensors()
        data_list = [self.X_star, self.u_actual, self.X_i, self.Y_i, self.X_b, self.X_f]
        return data_list


class Training:
    """
    Unsupervised training with customised training loss

    loss = initial loss + boundary loss + domain loss

    Parameters:
    --------------------------------------------------------------------------
    simulation_time (int) : number of seconds of simulation
    model (tf.Keras.Model) : NN representation
    data_list (list) : list of array sizes for inputs and output
    epochs (int) : number of cyles of training

    Returns:
    --------------------------------------------------------------------------
    u_pred (array) : predicted output solution for PINN

    """

    def __init__(self, simulation_time, model, data_list, epochs):
        self.model = model

        # random seed
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.90)
        self.epochs = epochs
        self.loss_list = []

        self.X_star, self.u_actual, self.X_i, self.Y_i, self.X_b, self.X_f = data_list

        self.loss_fnc = LossFunctions(simulation_time, self.model)

    def training_loop(self):
        it = 0
        start_time = time.time()
        while it < self.epochs:
            with tf.GradientTape() as tape:
                initial_loss = self.loss_fnc.reconstruction(
                    self.X_i, self.Y_i
                ) + self.loss_fnc.initial_velocity(self.X_i)
                boundary_loss = self.loss_fnc.boundary(self.X_b)
                domain_loss = self.loss_fnc.pde(self.X_f)

                print("Type of initial: ", type(initial_loss))
                print("Type of boundary: ", type(boundary_loss))
                print("Type of domain: ", type(domain_loss))

                loss = initial_loss + boundary_loss + domain_loss

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimiser.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

            self.loss_list.append(loss)

            it += 1

            print(
                "It: %d, Init: %.3e, Bound: %.3e, Domain: %.3e"
                % (it, initial_loss, boundary_loss, domain_loss)
            )

        self.train_time = time.time() - start_time

        return self.loss_list

    def trained_output(self):
        u_pred = self.model(tf.convert_to_tensor(self.X_star, dtype=tf.float32)).numpy()
        l2_error = np.mean((self.u_actual - u_pred) ** 2)

        print("Training Time: %d seconds, L2 Error: %.3e" % (self.train_time, l2_error))

        u_pred = u_pred.reshape(len(data.u), data.grid_length, data.grid_length)

        return u_pred


class Plotting:
    """
    Generating plots
    1. L2 loss over epochs
    2. Numerical solution against PINN for 3 sample points

    Parameters:
    --------------------------------------------------------------------------
    lost_list (list) : generated list of all loss over epochs
    u_pred (array) : predicted output solution for PINN
    u_sol (array) : numerical solution of u (array)
    sol_dict (dict) : dictionary of numerical solution inputs and output

    """

    def training_loss(self, loss_list):
        plt.plot(loss_list)
        plt.xlabel("Iterations")
        plt.ylabel("L2 Error")
        plt.show()

    def num_vs_pinn(self, u_sol, u_pred, sol_dict):
        u_field = u_sol
        t = sol_dict["t"]

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(u_field[0])
        ax.title.set_text("Initial")
        ax.set_ylabel("Solution")

        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(u_field[int(len(t) / 2)])
        ax.title.set_text("Middle")

        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(u_field[-1])
        ax.title.set_text("Final")

        u_field = u_pred

        ax = fig.add_subplot(2, 3, 4)
        ax.imshow(u_field[0])
        ax.set_ylabel("PINN")

        ax = fig.add_subplot(2, 3, 5)
        ax.imshow(u_field[int(len(t) / 2)])

        ax = fig.add_subplot(2, 3, 6)
        ax.imshow(u_field[-1])

        plt.show()


if __name__ == "__main__":

    print("Tf test gpu name: ", tf.test.gpu_device_name())

    command_line = CommandLineArgs()

    epochs = command_line.args.epochs
    spartial_discretisation = command_line.args.n_steps
    simulation_time = command_line.args.time
    grid_size = command_line.args.grid_size
    sample_dict = {
        "N_i": command_line.args.initial,
        "N_b": command_line.args.boundary,
        "N_f": command_line.args.domain,
    }

    numerical_sol = NumericalSol(spartial_discretisation, simulation_time, grid_size)
    u_sol = numerical_sol.solve_numerical()

    model = PINN()
    model.build(input_shape=(None, 3))
    model.summary()

    data = DataPrep(simulation_time, u_sol, numerical_sol.sol_dict, sample_dict, model)
    data_list = data.prepare()

    train_model = Training(simulation_time, model, data_list, epochs)
    lost_list = train_model.training_loop()

    u_pred = train_model.trained_output()

    plots = Plotting()
    plots.training_loss(lost_list)
    plots.num_vs_pinn(u_sol, u_pred, numerical_sol.sol_dict)
