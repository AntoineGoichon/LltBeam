import time
from typing import Callable, Tuple, Type, TypeVar

import numpy as np
from numpy import ndarray
from tqdm import tqdm


class NewMarkLin:
    """This class is used to solve the equation of motion of a 1D euler beam using the FEM method and the Newmark time integration scheme."""

    def __init__(
        self,
        M: ndarray,
        K: ndarray,
        C: ndarray,
        Ut: ndarray,
        dUt: ndarray,
        ddUt: ndarray,
        F: ndarray,
        t: ndarray,
        dt: float,
        gamma: float,
        beta: float,
        verbose=True,
    ) -> None:
        """__init__ method for NewMarkLin class. This class is used to solve the equation of motion of a 1D euler beam using the FEM method and the Newmark time integration scheme.

        parameters:
            M: np.array (2 * (Ne + 1), 2 * (Ne + 1)) [kg] mass matrix
            K: np.array (2 * (Ne + 1), 2 * (Ne + 1)) [N/m] stiffness matrix
            C: np.array (2 * (Ne + 1), 2 * (Ne + 1)) [N.s/m] damping matrix
            Un: np.array (2 * (Ne + 1), 1) [m] initial displacement
            dUt: np.array (2 * (Ne + 1), 1) [m/s] initial velocity
            ddUt: np.array (2 * (Ne + 1), 1) [m/s**2] initial acceleration
            t: np.array (Nt, 1) [s] time vector
            dt: float [s] time step
            gamma: float [-] gamma coefficient
            beta: float [-] beta coefficient
            Ne: int [-] number of elements
            array_loads: np.array (Nt, 1) [N] array of loads
            verbose: bool [-] verbose mode (default = True), if True, print the size of the matrices and vectors used in the resolution of the equation of motion and the resolution progress.
        """

        self.M_shape = M.shape[0]  # [-] shape of the mass matrix
        self.len_t = len(t)  # [-] number of time steps
        self.verbose = verbose  # [-] verbose mode

        if self.verbose:
            # print the size of the matrices and vectors used in the resolution of the equation of motion to check the memory usage.
            print("----beam fem init------")
            print("F size in Gb", F.shape[0] * F.shape[1] * 8 / 1e9)
            print("Ui size in Gb", self.M_shape * self.len_t * 8 / 1e9)
            print("ddUi size in Gb", self.M_shape * self.len_t * 8 / 1e9)
            print("dUi size in Gb", self.M_shape * self.len_t * 8 / 1e9)
            print("M size in Gb", self.M_shape * self.M_shape * 8 / 1e9)
            print("K size in Gb", self.M_shape * self.M_shape * 8 / 1e9)
            print("C size in Gb", self.M_shape * self.M_shape * 8 / 1e9)
            print("Sinv size in Gb", self.M_shape * self.M_shape * 8 / 1e9)
            print("K_static size in Gb", self.M_shape * self.M_shape * 8 / 1e9)
            print(
                "total size in Gb",
                (F.shape[0] * F.shape[1] * 8 / 1e9)
                + (self.M_shape * self.len_t * 8 / 1e9) * 4
                + (self.M_shape * self.M_shape * 8 / 1e9) * 5,
            )

        t_start = time.time()

        # init for static resolution:
        self.iter = 0

        print(gamma, beta)
        # time parameters:
        self.gamma = gamma  # [-] gamma coefficient
        self.beta = beta  # [-] beta coefficient

        # time parameters:
        self.dt = dt  # [s] time step
        self.len_t = len(t)  # [-] number of time steps

        # matrices and vectors of the system of equations of motion (initialization) :
        self.M = M.copy()  # [kg] mass matrix
        self.K = K.copy()  # [N/m] stiffness matrix
        self.C = C.copy()  # [N.s/m] damping matrix
        self.K_static = K.copy()  # [N/m] stiffness matrix for static resolution
        self.K_static[0, 0] = 1  # to avoid singularity
        self.K_static[1, 1] = 1  # to avoid singularity
        self.invK1_static = np.linalg.inv(
            self.K_static
        )  # inverse of the stiffness matrix for static resolution

        # exterior forces, data:
        self.F = F  # [N] array of loads

        # inverse matrix for resolution of newmark scheme:
        self.Sinv = np.linalg.inv(
            self.M + self.gamma * self.dt * self.C + self.beta * (self.dt**2) * self.K
        )

        # initial displacement, velocity and acceleration:
        self.Ut = Ut  # [m] initial displacement shape (2 * (Ne + 1), 1)
        self.dUt = dUt  # [m/s] initial velocity shape (2 * (Ne + 1), 1)
        self.ddUt = ddUt  # [m/s^2] initial acceleration shape (2 * (Ne + 1), 1)

        # Create NaN-filled matrices for displacements, velocities and accelerations:
        self.ddUi = np.full((self.M_shape, self.len_t), np.nan)  # [m/s^2] acceleration
        self.Ui = np.full((self.M_shape, self.len_t), np.nan)  # [m]   displacement
        self.dUi = np.full((self.M_shape, self.len_t), np.nan)  # [m/s] velocity
        self.U_static = np.full((int(self.M.shape[0] / 2), len(t)), np.nan)

        # first step saving:
        self.ddUi[:, 0] = self.ddUt[:, 0]  # [m/s^2] acceleration at t = 0
        self.Ui[:, 0] = self.Ut[:, 0]
        self.dUi[:, 0] = self.dUt[:, 0]

        # Create NaN-filled matrices for kinetic, elastic, external work and viscous dissipation and friction force work:
        self.T = np.full(len(t), np.nan)  # kinetic energy
        self.V = np.full(len(t), np.nan)  # elastic energy
        self.W = np.full(len(t), np.nan)  # external work
        self.D = np.full(len(t), np.nan)  # viscous dissipation
        self.KE = np.full(len(t), np.nan)  # kinetic energy
        self.EE = np.full(len(t), np.nan) # elastic energy

        # first iteration to zero:
        self.T[0] = 0
        self.V[0] = 0
        self.W[0] = 0
        self.D[0] = 0

        # time vector:
        self.t = t
    
    def newark_lin_step(self, Fext, Ut, dUt, ddUt, dt) -> Tuple:
        """Newmark time integration

        parameters:
            U: freedom degrees vector at t
            dU: freedom degrees velocity vector at t
            ddU: freedom degrees vector acceleration at t
            dt: float [s] time step

        returns:
            U1: freedom degrees vector at t + dt
            dU1: freedom degrees velocity vector at t + dt
            ddU1: freedom degrees vector acceleration at t + dt
            T1: float [J] kinetic energy at t + dt
            V1: float [J] elastic energy at t + dt
            W1: float [J] external work at t + dt
            D1: float [J] viscous dissipation at t + dt

        """

        # partial prediction of velocity and displacement for acceleration computation:
        dUt1 = dUt + (1 - self.gamma) * dt * ddUt
        Ut1 = Ut + dt * dUt + (0.5 - self.beta) * dt**2 * ddUt

        # computation of acceleration:
        ddUt1 = self.Sinv @ (Fext - self.K @ Ut1 - self.C @ dUt1)

        # correction of velocity and displacement (complete prediction):
        dUt1 = dUt1 + self.gamma * dt * ddUt1
        Ut1 = Ut1 + self.beta * dt**2 * ddUt1

        # Calcul variation energie par la methode des trapezes
        Tt1 = 0.5 * (dUt1 - dUt).T @ self.M @ (dUt1 + dUt)  # kinetic
        Vt1 = 0.5 * (Ut1 - Ut).T @ self.K @ (Ut1 + Ut)  # elastic
        Wt1 = (Ut1 - Ut).T @ Fext  # external work
        Dt1 = dt * (dUt1 + dUt).T / 2 @ self.C @ (dUt1 + dt) / 2  # viscous dissipation
        ket1 = 0.5 * dUt1.T @ self.M @ dUt1
        eet1 = 0.5 * Ut1.T @ self.K @ Ut1
        return Ut1, dUt1, ddUt1, Tt1, Vt1, Wt1, Dt1, ket1, eet1

    def run(self) -> bool:
        """Run the resolution of the equation of motion using the Newmark time integration scheme.

        Returns:
            bool: True if the resolution is done.
        """

        if self.verbose:
            print("----Newmark run------")
        t_start = time.time()

        if self.iter == 0:
            self.U_static = self.invK1_static @ self.F
            if self.verbose:
                print("Static resolution done")
        if self.verbose:
            print("Start dynamic resolution using newmark scheme:")
        for iter in tqdm(range(self.len_t - 1), disable=not self.verbose):
            Ut1, dUt1, ddUt1, Tt1, Vt1, Wt1, Dt1, ket1, eet1 = self.newark_lin_step(
                self.F[:, iter],
                self.Ui[:, iter],
                self.dUi[:, iter],
                self.ddUi[:, iter],
                self.dt,
            )
            self.Ui[:, iter + 1] = Ut1[:]
            self.dUi[:, iter + 1] = dUt1[:]
            self.ddUi[:, iter + 1] = ddUt1[:]

            self.T[iter + 1] = Tt1
            self.V[iter + 1] = Vt1
            self.W[iter + 1] = Wt1
            self.D[iter + 1] = Dt1
            self.KE[iter + 1] = ket1
            self.EE[iter + 1] = eet1

        t_end = time.time()
        if self.verbose:
            print(f"elapsed time = {t_end - t_start:.2f} s\n----END Newmark run------")
        return True

    def save_data(self, path_save) -> bool:
        """Save the data of the resolution of the equation of motion using the Newmark time integration scheme.

        Args:
            path_save (str): path to save the data.

        Returns:
            bool: True if the data is saved.
        """
        np.save(file=path_save + "Ui.npy", arr=self.Ui)
        np.save(file=path_save + "dUi.npy", arr=self.dUi)
        np.save(file=path_save + "ddUi.npy", arr=self.ddUi)
        np.save(file=path_save + "F.npy", arr=self.F)
        np.save(file=path_save + "T.npy", arr=self.T)
        np.save(file=path_save + "V.npy", arr=self.V)
        np.save(file=path_save + "W.npy", arr=self.W)
        np.save(file=path_save + "D.npy", arr=self.D)
        np.save(file=path_save + "U_static.npy", arr=self.U_static)
        np.save(file=path_save + "t.npy", arr=self.t)
        np.save(file=path_save + "KE.npy", arr=self.KE)
        np.save(file=path_save + "EE.npy", arr=self.EE)
        return True

    def save_data_with_dowsampling_factor(self, path_save, factor) -> bool:
        """Save the data of the resolution of the equation of motion using the Newmark time integration scheme and downsampling the data by a factor.

        Args:
            path_save (str): path to save the data.

        Returns:
            bool: True if the data is saved.
        """

        np.save(file=path_save + "U_static.npy", arr=self.U_static[:, ::factor])
        np.save(file=path_save + "Ui.npy", arr=self.Ui[:, ::factor])
        np.save(file=path_save + "dUi.npy", arr=self.dUi[:, ::factor])
        np.save(file=path_save + "ddUi.npy", arr=self.ddUi[:, ::factor])
        np.save(file=path_save + "F.npy", arr=self.F[:, ::factor])
        np.save(file=path_save + "T.npy", arr=self.T[::factor])
        np.save(file=path_save + "V.npy", arr=self.V[::factor])
        np.save(file=path_save + "W.npy", arr=self.W[::factor])
        np.save(file=path_save + "D.npy", arr=self.D[::factor])
        np.save(file=path_save + "t.npy", arr=self.t[::factor])
        np.save(file=path_save + "KE.npy", arr=self.KE[::factor])
        np.save(file=path_save + "EE.npy", arr=self.EE[::factor])
        return True