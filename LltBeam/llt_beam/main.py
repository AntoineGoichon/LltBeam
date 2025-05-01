import os
import shutil

import numpy as np
import pandas as pd

from .newmark import NewMarkLin
from .utils import get_intial_static_displacement
from .llt import Llt


class LltBeam:
    """
    Computes aerodynamic loads using lifting line theory (LLT) and simulates
    the structural response of a beam using the Newmark method.

    This class processes a set of flight data files stored in a folder. For each flight,
    it computes aerodynamic loads with LLT, projects them onto a beam structure using
    a projection matrix, simulates the structural response, and saves the results.
    Optional features include gravitational loading, strain computation, and automatic
    cleanup of temporary files.

    Parameters
    ----------
    wing : object
        Object defining the wing geometry and aerodynamic properties.
    K : np.ndarray
        Stiffness matrix of the beam.
    M : np.ndarray
        Mass matrix of the beam.
    C : np.ndarray
        Damping matrix of the beam.
    P : np.ndarray
        Projection matrix mapping aerodynamic loads to structural DOFs.
    gamma_newmark : float
        Gamma parameter for the Newmark integration scheme.
    beta_newmark : float
        Beta parameter for the Newmark integration scheme.
    path_fp_folder : str
        Path to the folder containing raw flight data files.
    path_experimentation : str
        Root directory where results and temporary files will be stored.
    llt_resolution : int
        Number of spanwise sections for the lifting line theory computation.
    frequency : float
        Sampling frequency used for the dynamic simulation.
    batch_size : int, optional
        Batch size for LLT computation; defaults to None (5% of the total data flight point).
    downsampling_factor : int, optional
        Factor for reducing temporal resolution of the outputs; defaults to 1.
    clear_temp : bool, optional
        Whether to delete temporary files after processing each flight; defaults to True.
    B : np.ndarray, optional
        Matrix for computing projected strain quantities; defaults to None.
    add_gravity : bool, optional
        Whether to include gravitational force in the structural simulation; defaults to False.

    Attributes
    ----------
    flight_list : list of str
        List of flight filenames automatically loaded from `path_fp_folder`.
    """

    def __init__(
        self,
        wing,
        K,
        M,
        C,
        P,
        gamma_newmark,
        beta_newmark,
        path_fp_folder,
        path_experimentation,
        llt_resolution,
        frequency,
        batch_size=None,
        downsampling_factor=1,
        clear_temp=True,
        B=None,
        add_gravity=False,
    ):

        self.wing = wing
        self.M = M
        self.K = K
        self.C = C
        self.P = P
        self.B = B
        self.gamma_newmark = gamma_newmark
        self.beta_newmark = beta_newmark
        self.add_gravity = add_gravity
        self.path_fp_folder = path_fp_folder
        self.llt_resolution = llt_resolution
        self.frequency = frequency
        self.path_experimentation = path_experimentation
        self.llt_resolution = llt_resolution

        self.flight_list = os.listdir(path_fp_folder)
        self.batch_size = batch_size
        self.downsampling_factor = downsampling_factor
        self.clear_temp = clear_temp

        if self.downsampling_factor is None:
            self.downsampling_factor = 1

        if not os.path.exists(path_experimentation):
            os.makedirs(path_experimentation)

    def run(self):
        """
        Executes the full load computation and structural response pipeline for each flight.

        For each flight in the flight list, this method:
        1. Loads the corresponding flight data.
        2. Computes aerodynamic loads using lifting line theory (LLT).
        3. Projects the loads onto the structural beam using a projection matrix.
        4. Optionally adds gravitational effects to the loading.
        5. Removes boundary conditions from the loads.
        6. Computes the initial static displacement.
        7. Simulates the dynamic response using the Newmark method.
        8. Saves the computed results and flight data.
        9. Optionally generates and saves a combined results dataframe.
        10. Optionally deletes temporary files after processing.

        This method is the main entry point for processing all flights in a batch.
        """
        for i, flight in enumerate(self.flight_list):
            print(f"file {i+1}/{len(self.flight_list)}, ", flight)

            # read the flight_data
            fp = pd.read_hdf(
                self.path_fp_folder + flight, index_col=0, encoding="ISO-8859-1"
            )
            t = fp["time"].values
            
            # set batch size for LLT computation
            if self.batch_size is None:
                batch_size = fp.shape[0] // 20
                
            # compute the loads using the lifting line theory
            llt1 = Llt(
                self.wing,
                fp,
                self.llt_resolution,
                path_saving=self.path_experimentation + "temp/" + flight,
                verbose=True,
                batch_size=batch_size,
            )

            llt1.run()

            # read the loads computed by the lifting line theory
            loads = pd.read_hdf(
                self.path_experimentation + "temp/" + flight + "/" + "load.h5"
            ).values.T

            # Adapt the load to the beam (Load is compute for the two wings).
            loads = loads[(loads.shape[0] // 2) :, :]

            # Use the P matrix to project the load on the beam
            loads = self.P @ (loads)

            # if the gravity is added, we need to add the gravity load
            if self.add_gravity:
                g_corrected = np.zeros(self.M.shape[0])
                g_corrected[0::2] = -9.81
                gavity = (self.M @ g_corrected)[:, None] * np.ones_like(loads)

                loads = loads + gavity

            # remove boundary conditions from the loads
            loads[0, :] = 0
            loads[1, :] = 0

            # copy and modify the stiffness matrix to compute the static displacement
            K_static = self.K.copy()
            K_static[0, 0] = 1
            K_static[1, 1] = 1
            Un = get_intial_static_displacement(K_static, loads)
            dUn = np.zeros((self.M.shape[0], 1))

            # run the newmark method
            newmarklin = NewMarkLin(
                self.M,
                self.K,
                self.C,
                Ut=Un,
                dUt=dUn,
                ddUt=dUn,
                t=t,
                dt=1 / self.frequency,
                gamma=self.gamma_newmark,
                beta=self.beta_newmark,
                F=loads,
            )

            newmarklin.run()

            # save the results
            if not os.path.exists(self.path_experimentation + "temp/" + flight + "/"):
                os.makedirs(self.path_experimentation + "temp/" + flight + "/")

            newmarklin.save_data_with_dowsampling_factor(
                self.path_experimentation + "temp/" + flight + "/",
                factor=self.downsampling_factor,
            )

            # save the flight data
            if not os.path.exists(self.path_experimentation + "flight_data/"):
                os.makedirs(self.path_experimentation + "flight_data/")

            fp.to_hdf(
                self.path_experimentation + "flight_data/" + flight, key="df", mode="w"
            )

            # generate the dataframe
            self.generate_dataframe(flight)

            # remove the temporary files
            if self.clear_temp:
                self.clear_flight_temp(flight)

    def generate_dataframe(self, flight):
        """
        Generates and saves a simulation dataframe for a given flight.

        This method loads multiple input arrays (e.g., displacements, forces, energies, etc.)
        associated with the given flight identifier, constructs a unified pandas DataFrame,
        optionally computes strain projections using the B matrix (if provided),
        and merges the result with downsampled flight data. The final dataframe is then
        saved as an HDF5 file in the 'simulation_results/' directory.

        Parameters
        ----------
        flight : str
            Identifier of the flight whose data will be loaded and processed.
        """

        path_flight_data = self.path_experimentation + "flight_data/"
        fp = pd.read_hdf(os.path.join(path_flight_data, flight), key="df")[
            :: self.downsampling_factor
        ]
        fp = fp.reset_index(drop=True)
        Ui = np.load(os.path.join(self.path_experimentation, "temp/", flight, "Ui.npy"))
        dUi = np.load(
            os.path.join(self.path_experimentation, "temp/", flight, "dUi.npy")
        )
        ddUi = np.load(
            os.path.join(self.path_experimentation, "temp/", flight, "ddUi.npy")
        )
        U_static = np.load(
            os.path.join(self.path_experimentation, "temp/", flight, "U_static.npy")
        )
        F = np.load(os.path.join(self.path_experimentation, "temp/", flight, "F.npy"))
        T = np.load(os.path.join(self.path_experimentation, "temp/", flight, "T.npy"))[
            None, :
        ]
        V = np.load(os.path.join(self.path_experimentation, "temp/", flight, "V.npy"))[
            None, :
        ]
        W = np.load(os.path.join(self.path_experimentation, "temp/", flight, "W.npy"))[
            None, :
        ]
        t = np.load(os.path.join(self.path_experimentation, "temp/", flight, "t.npy"))[
            None, :
        ]
        Ee = np.load(
            os.path.join(self.path_experimentation, "temp/", flight, "EE.npy")
        )[None, :]
        Ke = np.load(
            os.path.join(self.path_experimentation, "temp/", flight, "KE.npy")
        )[None, :]

        # columns names
        Ui_col = [f"Ui_{i}" for i in range(Ui.shape[0])]
        dUi_col = [f"dUi_{i}" for i in range(dUi.shape[0])]
        ddUi_col = [f"ddUi_{i}" for i in range(ddUi.shape[0])]
        F_col = [f"F_{i}" for i in range(F.shape[0])]
        U_static_col = [f"U_static_{i}" for i in range(U_static.shape[0])]

        # energy
        T_col = ["T"]
        V_col = ["V"]
        W_col = ["W"]
        t_col = ["t"]
        EE_col = ["EE"]
        KE_col = ["KE"]

        cols = (
            Ui_col
            + dUi_col
            + ddUi_col
            + U_static_col
            + F_col
            + T_col
            + V_col
            + W_col
            + t_col
            + EE_col
            + KE_col
        )

        Big_array = np.concatenate(
            (Ui, dUi, ddUi, U_static, F, T, V, W, t, Ee, Ke), axis=0
        )
        df = pd.DataFrame(Big_array.T, columns=cols)
        df = df.reset_index(drop=True)

        if self.B is not None:
            eps_static = [f"eps_static_{i}" for i in range(U_static.shape[0] // 2)]
            eps_dynamic = [f"eps_{i}" for i in range(U_static.shape[0] // 2)]
            strain_static = self.B @ df[U_static_col].T
            strain_dynamic = self.B @ df[Ui_col].T
            df[eps_static] = strain_static.T
            df[eps_dynamic] = strain_dynamic.T

        del Big_array
        fp.index = df.index
        df = pd.concat((df, fp), axis=1)

        if not os.path.exists(self.path_experimentation + "simulation_results/"):
            os.makedirs(self.path_experimentation + "simulation_results/")

        save_path = os.path.join(
            self.path_experimentation + "simulation_results/", flight
        )
        # replace .csv by .h5
        flight_h5 = flight.replace(".csv", ".h5")
        
        df.to_hdf(
        self.path_experimentation + "simulation_results/" + flight_h5, key="df", mode="w"
        )
        print("file successfully saved")

    def clear_flight_temp(self, flight):
        """
        Deletes temporary files (.h5 and .npy) from the 'temp/' directory
        associated with the current experiment, then removes the directory itself.

        This method helps free up disk space and prevent conflicts by
        cleaning intermediate data generated during training or evaluation.
        """
        for file in os.listdir(self.path_experimentation + "temp/" + flight):
            if file.endswith(".h5"):
                os.remove(self.path_experimentation + "temp/" + flight + "/" + file)
            if file.endswith(".npy"):
                os.remove(self.path_experimentation + "temp/" + flight + "/" + file)

        os.rmdir(self.path_experimentation + "temp/" + flight)
