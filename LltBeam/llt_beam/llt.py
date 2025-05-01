import logging
import os
import sys
import time
from functools import cached_property, reduce
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from tqdm import tqdm


class Llt:
    """
    Lifting Line Theory (LLT) class for computing aerodynamic loads on a wing 
    based on time-resolved flight data.

    This class implements a discretized version of the lifting line theory to compute 
    spanwise aerodynamic loads (e.g., lift) using the wing geometry and a set of flight 
    points (angle of attack, speed, etc.). It supports batch processing and optional 
    memory- and compressibility-related settings. Results can be saved to disk if 
    a path is provided.

    Parameters
    ----------
    wing : pd.DataFrame
        DataFrame containing the geometric and aerodynamic parameters of the wing 
        (e.g., spanwise sections, chord lengths, twist).
    flight_points : pd.DataFrame
        Time-resolved flight data used as aerodynamic input (e.g., velocity, AoA).
    fourrier_coefficients_number : int
        Number of Fourier coefficients used in the LLT formulation (resolution in circulation distribution).
    path_saving : str
        Directory path where intermediate or final results will be saved.
    verbose : bool, optional
        If True, progress messages are printed. Default is True.
    batch_size : int, optional
        Number of flight points to process per batch. Default is 1024.
    low_memory : bool, optional
        If True, reduces memory usage at the cost of computation speed. Default is False.
    """

    def __init__(
        self,
        wing: pd.DataFrame,
        flight_points: pd.DataFrame,
        fourrier_coefficients_number: int,
        path_saving: str,
        verbose=True,
        batch_size=1024,
        low_memory = False,
    ) -> None:
        """
        Initialize the Llt class for computing aerodynamic loads using lifting line theory.

        This constructor sets up the lifting line theory solver by initializing the 
        wing geometry, computing the discretization in theta space for the Fourier 
        representation, and preparing the matrices required to solve the linear 
        aerodynamic problem. It also sets up the data structures for storing the 
        results (loads and aerodynamic coefficients) for all flight points.

        Parameters
        ----------
        wing : pd.DataFrame
            Spanwise wing data including chord length, zero-lift angle, aileron effectiveness, and slope.
        flight_points : pd.DataFrame
            Time-resolved flight data (e.g., airspeed, air density, angle of attack).
        fourrier_coefficients_number : int
            Number of Fourier coefficients used in the LLT formulation. Higher values provide more resolution.
        path_saving : str
            Directory path where intermediate and final results will be saved.
        verbose : bool, optional
            If True, prints initialization and timing information. Default is True.
        batch_size : int, optional
            Number of flight points to process per batch. Default is 1024.
        low_memory : bool, optional
            If True, enables memory-efficient operations. Default is False.
        """
        self.verbose = verbose
        self.batchsize = batch_size
        self.path_saving = path_saving
        self.low_memory = low_memory
        
        t_start = time.time()

        # for flight points:
        self.flight_points = flight_points

        # wing caracteristique
        self.wing = self.create_symetric_wing(wing) # symetric wing caracteristics.
        self.x = self.wing.index.values # position along the wing [m]
        self.dx = self.x[1] - self.x[0] # distance between two points [m]
        self.winglength = self.wing.index.max() # wing length [m]
        self.wingspan = 2 * self.winglength # wingspan [m]
        self.chord_x = self.wing["chord"].values  # chord of the wing at each x [m]
        self.ar = self.wingspan / self.chord_x.mean() # aspect ratio

        self.zero_lift_angle = self.wing["zero_lift_angle"].values # zero lift angle at each x [rad]
        self.ailerons_effectiveness = self.wing["ailerons_effectiveness"].values # ailerons effectiveness at each x

        # llt resolution parameters:
        self.Fourrier_coefficients_number = (
            fourrier_coefficients_number - 1
        )  # -1 because we don't take into account the first coefficient (n=0) in the lifting line theory.
        self.sum_n_fourrier = np.arange(1, self.Fourrier_coefficients_number + 1) # n values for the fourrier coefficients
        self.thetas = np.linspace( 
            np.pi / fourrier_coefficients_number,
            np.pi * (fourrier_coefficients_number - 1) / fourrier_coefficients_number,
            (fourrier_coefficients_number - 1),
        ) # theta values for the fourrier coefficients
        self.thetas_matrix = np.repeat(
            self.thetas[None, :].transpose(),
            repeats=self.Fourrier_coefficients_number,
            axis=1,
        ) # matrix of thetas

        self.x_from_thetas = -(self.wingspan / 2) * np.cos(self.thetas) # x values from the thetas used in the lifting line theory.
        self.wing_adapted_to_An = self.adapt_to_thetas(self.wing, self.x_from_thetas) # wing caracteristics adapted to the thetas used in the lifting line theory.
        self.chord_theta = self.wing_adapted_to_An["chord"].values # chord at each theta.
        self.zeros_lift = self.wing_adapted_to_An["zero_lift_angle"].values # zero lift angle at each theta.

        self.chord_theta_matrix = np.repeat(
            self.chord_theta[None, :].transpose(),
            repeats=self.Fourrier_coefficients_number,
            axis=1,
        ) # matrix of chord at each theta.

        self.slope_theta = self.wing_adapted_to_An["slope"].values # slope at each theta.
        self.slope_theta_matrix = np.repeat(
            self.slope_theta[None, :].transpose(),
            repeats=self.Fourrier_coefficients_number,
            axis=1,
        ) # matrix of slope at each theta.

        # spatial résolutions:
        self.Matrix_sum_sin_n_theta = self.sum_sin_n_theta(
            self.wing.index.values, self.Fourrier_coefficients_number, self.wingspan
        ) # matrix of the sum of sin(n*theta) for each x.

        # initialization for the linear problem to be solved
        self.U = self.calcul_matrix_U() # matrix U used in the linear problem to be solved.
        self.nb_structure = len(self.x) # number of structure to compute the loads.

        # saving generated data and columns names for the DataFrame.
        self.loads_cols = [f"load{i}" for i in range(self.nb_structure)] # columns names for the loads DataFrame.
        self.cls = np.zeros((self.nb_structure, self.flight_points.shape[0])) # lift coefficient at each section of the wing.
        self.cls_cols = [f"cl{i}" for i in range(self.nb_structure)] # columns names for the cls DataFrame.
        self.cds = np.zeros((self.nb_structure, self.flight_points.shape[0])) # drag coefficient at each section of the wing.
        self.cds_cols = [f"cd{i}" for i in range(self.nb_structure)] # columns names for the cds DataFrame.
        self.total_cl = np.zeros((self.flight_points.shape[0])) # total lift coefficient at each flight point.
        self.total_loads = np.zeros((self.flight_points.shape[0])) # total aerodynamic loads at each flight point.

        t_end = time.time()

    def calcul_matrix_U(self) -> np.ndarray:
        """
        Computes the matrix U used in the linear system formulation of the lifting line theory.

        This matrix is derived from the Fourier representation of the circulation distribution 
        and is used in solving the aerodynamic load distribution along the wing span.

        Returns
        -------
        np.ndarray
            Matrix U involved in the LLT linear problem, representing the discretized influence 
            of circulation modes over the spanwise wing sections.
        """

        self.thetas_matrix_n = self.thetas_matrix * self.sum_n_fourrier
        self.divide_sin_matrix = np.sin(self.thetas_matrix)

        U = (
            (2 * self.wingspan) / (self.slope_theta_matrix * self.chord_theta_matrix)
        ) * np.sin(self.thetas_matrix_n) + self.sum_n_fourrier * np.divide(
            np.sin(self.thetas_matrix_n),
            self.divide_sin_matrix,
            out=np.ones_like(self.thetas_matrix_n) * np.sin(self.thetas[-1]),
            where=self.divide_sin_matrix != 0,
        )

        return U
    

        
    def run(self) -> None:
        """
        Executes the lifting line theory computation over all flight points.

        This method iterates over each flight point (or batch of points) provided during 
        initialization and computes the spanwise aerodynamic loads using the lifting line theory.
        For each batch, it calculates the effective angle of attack, solves the linear LLT system, 
        computes circulation, lift coefficients, and aerodynamic loads, and saves the results to disk.

        The computation accounts for dynamic effects such as roll rate and aileron deflection, 
        and can operate in memory-efficient mode if specified.

        Results are stored as HDF5 files, including:
        - Spanwise loads (`load.h5`)
        - Sectional lift coefficients (`cls.h5`)
        - Total lift coefficient (`total_cl.h5`)
        - Total aerodynamic load (`total_loads.h5`)
        """
        if self.verbose:
            print("----LLT run------")
        t_start = time.time()

        # already used parameters in V1 (not so) stable release.
        aoas = self.flight_points["Incidence angle alpha, rad"].values
        rhos = self.flight_points["Air Density, kg/m^3"].values
        v_infs = self.flight_points["Air speed, m/s"].values
        roll_rate = self.flight_points["roll rate (rad/s)"].values
        # new parameters to improve aerodynamic loads prédiction.
        ailerons = self.flight_points[
            "Aileron, dAr, rad, positive: left trailing edge down"
        ].values
        ailerons_effectiveness = self.wing_adapted_to_An[
            "ailerons_effectiveness"
        ].values


        for i in tqdm(range(self.flight_points.shape[0]//self.batchsize + 1)):
            start = i * self.batchsize
            end = (i + 1) * self.batchsize
            # hande the last batch:
            if end > self.flight_points.shape[0]:
                end = self.flight_points.shape[0]
                
            if i == 0:
                self.run_batch(
                    aoas[start:end],
                    ailerons[start:end],
                    roll_rate[start:end],
                    v_infs[start:end],
                    rhos[start:end],
                    first = True
                )
            else:
                self.run_batch(
                    aoas[start:end],
                    ailerons[start:end],
                    roll_rate[start:end],
                    v_infs[start:end],
                    rhos[start:end],
                    first=False
                )
                

        t_end = time.time()
        if self.verbose:
            print(f"elapsed time = {t_end - t_start:.2f} s\n----END LLT run------")
            
    def run_batch(self, aoas, ailerons, roll_rate, v_infs, rhos, first = False) -> None:
        """
        Processes a batch of flight points and computes aerodynamic loads using LLT.

        For the given batch of flight conditions, this method computes the adapted angle 
        of attack (including aileron and roll effects), solves the LLT system for Fourier 
        coefficients, and derives spanwise circulation, lift coefficients, and total loads.

        Results are saved to HDF5 files. The first batch creates new files; subsequent batches append.

        Parameters
        ----------
        aoas : np.ndarray
            Array of angle of attack values for the batch [rad].
        ailerons : np.ndarray
            Aileron deflections for the batch [rad].
        roll_rate : np.ndarray
            Roll rate values for the batch [rad/s].
        v_infs : np.ndarray
            Free-stream velocities for the batch [m/s].
        rhos : np.ndarray
            Air density values for the batch [kg/m³].
        first : bool, optional
            If True, creates new result files. If False, appends to existing ones. Default is False.
        """
        aoas_batch = np.repeat(aoas[:, None], self.Fourrier_coefficients_number, axis=1)
        aileron_batch = np.repeat(ailerons[:, None], self.Fourrier_coefficients_number, axis=1)
        roll_rate_batch = np.repeat(roll_rate[:, None], self.Fourrier_coefficients_number, axis=1)
        ailerons_effectiveness = self.wing_adapted_to_An["ailerons_effectiveness"].values

        self.adapted_aoa = (
            aoas_batch
            + aileron_batch * ailerons_effectiveness
            - 2*roll_rate_batch * (self.wing_adapted_to_An.index.values / v_infs[:,None])
        )
        V = self.calcul_V_batch(self.adapted_aoa).transpose()

        An = self.solve_UX_V(self.U, V)

        self.circulation_distribution_of_x = (
            2 * self.wingspan * v_infs * np.dot(self.Matrix_sum_sin_n_theta, An)
        )
        
        self.lift_coefficient_section_wise = (2 * self.circulation_distribution_of_x) / (
            (self.chord_x[:,None]* np.repeat(v_infs[None,:],self.nb_structure,axis = 0))
        )
        
        self.load_section_wise = v_infs * rhos * self.circulation_distribution_of_x
        
        self.total_loads = np.sum(self.load_section_wise*self.dx,axis=0)
            
        self.total_cl = 2*self.total_loads / (v_infs*v_infs*rhos*np.sum(self.chord_x*self.dx))

        
        # to dataframes:
        self.load_section_wise = pd.DataFrame(self.load_section_wise.T, columns = self.loads_cols)
        
        self.total_loads = pd.DataFrame(self.total_loads, columns = ["total_loads"])
        
        self.total_cl = pd.DataFrame(self.total_cl, columns = ["total_cl"])
        
        self.cls = pd.DataFrame(self.lift_coefficient_section_wise.T, columns = self.cls_cols)
        
        # saving data to hdf5:
        
        if first:
            if not os.path.exists(self.path_saving):
                os.makedirs(self.path_saving)
                
            self.load_section_wise.to_hdf(os.path.join(self.path_saving,"load.h5"), key = "load", mode = "w",format = "table")
            if not self.low_memory:
                self.total_cl.to_hdf(os.path.join(self.path_saving,"total_cl.h5"), key = "total_cl", mode = "w",format = "table")
                self.cls.to_hdf(os.path.join(self.path_saving,"cls.h5"), key = "cls", mode = "w",format = "table")
                self.total_loads.to_hdf(os.path.join(self.path_saving,"total_loads.h5"), key = "total_loads", mode = "w",format = "table")
        else:
            self.load_section_wise.to_hdf(os.path.join(self.path_saving,"load.h5"), key = "load", mode = "r+",append = True)
            if not self.low_memory:
                self.total_cl.to_hdf(os.path.join(self.path_saving,"total_cl.h5"), key = "total_cl", mode = "r+",append = True)
                self.cls.to_hdf(os.path.join(self.path_saving,"cls.h5"), key = "cls", mode = "r+",append = True)
                self.total_loads.to_hdf(os.path.join(self.path_saving,"total_loads.h5"), key = "total_loads", mode = "r+",append = True)
            
    def calcul_V_batch(self, alpha_theta: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side matrix V of the lifting line theory linear system.

        This matrix represents the effective angle of attack at each spanwise location 
        (in theta space) after subtracting the local zero-lift angle.

        Parameters
        ----------
        alpha_theta : np.ndarray
            Effective angle of attack at each theta and for each time step [rad].

        Returns
        -------
        np.ndarray
            Matrix V used in the LLT linear system, containing the angle of attack 
            deviation from zero-lift for each theta.
        """
        # print(alpha_theta - self.zeroas_lift_angle_from_thetas)
        V = (alpha_theta - self.zeros_lift) # correction de prandtl glauert to be added here.
        return V
    
    def calcul_V(self, alpha_theta: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side vector V of the lifting line theory linear system 
        for a single flight point.

        This vector represents the effective angle of attack at each spanwise location 
        (in theta space), corrected by the local zero-lift angle.

        Parameters
        ----------
        alpha_theta : np.ndarray
            Effective angle of attack at each theta [rad] for a single time step.

        Returns
        -------
        np.ndarray
            Column vector V used in the LLT linear system, representing the angle 
            of attack deviation from zero-lift along the span.
        """
        # print(alpha_theta - self.zeroas_lift_angle_from_thetas)
        V = (alpha_theta - self.zeros_lift)[None, :].transpose()
        return V

    def solve_UX_V(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Solves the linear system U · An = V in the lifting line theory formulation.

        This method computes the Fourier coefficients (An) of the circulation distribution 
        by solving the linear system derived from the lifting line theory discretization.

        Parameters
        ----------
        U : np.ndarray
            Matrix U representing the system coefficients in the LLT formulation.
        V : np.ndarray
            Right-hand side matrix or vector representing the effective angle of attack deviation.

        Returns
        -------
        np.ndarray
            Solution An of the linear system, corresponding to the Fourier coefficients 
            of the circulation distribution along the wing.
        """
        An = np.linalg.solve(U, V)
        return An

    ### get fonction used after resolution of the linear problems ###

    def get_load_section_wise(self, v_inf: float, rho: float) -> np.ndarray:
        """
        Computes the aerodynamic load distribution along the wing span.

        This method calculates the spanwise aerodynamic loads from the previously 
        computed circulation distribution, using the classical lifting line expression.

        Parameters
        ----------
        v_inf : float
            Free-stream velocity [m/s].
        rho : float
            Air density [kg/m³].

        Returns
        -------
        np.ndarray
            Aerodynamic loads [N/m] at each section along the wing span.
        """
        return v_inf * rho * self.circulation_distribution_of_x

    def get_circulation_distribution_of_x(self, v_inf: float) -> np.ndarray:
        """
        Computes the spanwise circulation distribution from the Fourier coefficients.

        This method reconstructs the circulation distribution at each spanwise section 
        using the Fourier coefficients obtained from the lifting line theory solution.

        Parameters
        ----------
        v_inf : float
            Free-stream velocity [m/s].

        Returns
        -------
        np.ndarray
            Circulation distribution [m²/s] at each spanwise location along the wing.
        """
        circulation_distribution_of_x = (
            2 * self.wingspan * v_inf * np.dot(self.Matrix_sum_sin_n_theta, self.An)
        )
        return circulation_distribution_of_x

    def get_lift_coefficient_section_wise(self, v_inf: float) -> np.ndarray:
        """
        Computes the lift coefficient at each spanwise section of the wing.

        This method reconstructs the circulation distribution from the Fourier coefficients
        and converts it into sectional lift coefficients using the local chord length.

        Parameters
        ----------
        v_inf : float
            Free-stream velocity [m/s].

        Returns
        -------
        np.ndarray
            Lift coefficient at each spanwise section (dimensionless).
        """
        circulation_distribution_of_x = self.get_circulation_distribution_of_x(
            v_inf
        ).transpose()[0]
        lift_coefficient_section_wise = (2 * circulation_distribution_of_x) / (
            v_inf * self.chord_x
        )
        return lift_coefficient_section_wise

    def create_symetric_wing(self, wing: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a symmetric wing by mirroring the input wing geometry about the root.

        This is used to construct a full wing model from a half-wing definition.

        Parameters
        ----------
        wing : pd.DataFrame
            Spanwise wing characteristics (e.g., chord, zero-lift angle, aileron effectiveness).

        Returns
        -------
        pd.DataFrame
            Symmetric wing characteristics covering both sides of the span.
        """
        wing_sym = wing.copy()
        wing_sym.transpose()
        wing_sym.index = -wing_sym.index
        wing_sym["ailerons_effectiveness"] = -wing_sym["ailerons_effectiveness"]
        wing_sym = wing_sym[::-1]
        wing_sym = wing_sym.iloc[:-1]
        wing = pd.concat([wing_sym, wing])
        return wing

    def adapt_to_thetas(
        self, wing: pd.DataFrame, x_from_thetas: np.ndarray
    ) -> pd.DataFrame:
        """
        Interpolates the wing characteristics at the theta-based spanwise locations.

        Required to align wing parameters with the discretization used in the LLT formulation.

        Parameters
        ----------
        wing : pd.DataFrame
            Original wing data indexed by spanwise location.
        x_from_thetas : np.ndarray
            x-positions corresponding to theta sampling (cosine-spaced).

        Returns
        -------
        pd.DataFrame
            Wing data interpolated and adapted to theta-based spanwise locations.
        """

        effectiveness = wing[wing.index > 0]["ailerons_effectiveness"].max()
        df_x_from_thetas = pd.DataFrame(index=x_from_thetas)

        for col in wing.columns:
            if col == "aileron":
                None
            else:
                df_x_from_thetas[col] = np.interp(x_from_thetas, wing.index, wing[col])
        index_to_change = df_x_from_thetas[
            df_x_from_thetas["ailerons_effectiveness"] != 0
        ].index
        # plt.plot(df_x_from_thetas["ailerons_effectiveness"])
        df_x_from_thetas.loc[index_to_change, "ailerons_effectiveness"] = effectiveness
        df_x_from_thetas.fillna(0, inplace=True)
        # plt.plot(df_x_from_thetas["ailerons_effectiveness"])

        # change the sign of the ailerons_effectiveness for df.index < 0:
        non_zero_index = df_x_from_thetas[
            df_x_from_thetas["ailerons_effectiveness"] != 0
        ].index
        non_zero_negative_index = df_x_from_thetas.loc[non_zero_index][
            df_x_from_thetas.loc[non_zero_index].index < 0
        ].index

        df_x_from_thetas.loc[non_zero_negative_index, "ailerons_effectiveness"] = (
            -1 * df_x_from_thetas.loc[non_zero_negative_index]
        )

        return df_x_from_thetas

    def sum_sin_n_theta(
        self, x: np.ndarray, Fourrier_coefficients_number: int, wingspan: float
    ) -> np.ndarray:
        """
        Constructs the matrix of sine terms used to reconstruct circulation in LLT.

        This matrix contains sin(n·θ) terms evaluated at spanwise points x, for use 
        in reconstructing the circulation distribution from Fourier coefficients.

        Parameters
        ----------
        x : np.ndarray
            Spanwise positions along the wing.
        Fourrier_coefficients_number : int
            Number of Fourier modes used in the LLT formulation.
        wingspan : float
            Total wingspan of the aircraft.

        Returns
        -------
        np.ndarray
            Matrix of sin(n·θ(x)) values used to reconstruct circulation distribution.
        """

        # self.thetas_from_x_matrix_n est la matrice des thetas_n multiplier par n ou on a (m)ij = theta_i * j
        self.thetas_from_x = np.arccos(((-2 * x) / wingspan))
        self.thetas_from_x_matrix = np.repeat(
            self.thetas_from_x[None, :].transpose(),
            repeats=Fourrier_coefficients_number,
            axis=1,
        )
        thetas_from_x_matrix_n = np.sin(self.thetas_from_x_matrix * self.sum_n_fourrier)

        return thetas_from_x_matrix_n
