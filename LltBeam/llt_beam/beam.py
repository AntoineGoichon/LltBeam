from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas
from numpy import int64, ndarray, single


class AssembleBeamMatrix:
    """
    This class is used to assemble the beam matrices of a 1D euler beam using the FEM method from beam Dataframe and stiffness_elementary_matrix_function and mass_elementary matrix function.
    it is based on mechanical vibrations theory and application to dynamics, third edition, from Michel Geradin and Daniel J. Rixen p 379.
    """

    def __init__(
        self,
        beam: pandas.core.frame.DataFrame,
        stiffness_elementary_matrix_function: Callable,
        mass_elementary_matrix_function: Callable,
    ):
        """

        Args:
            beam (pandas.core.frame.DataFrame): DataFrame with the following columns: E (young modulus), Ixx (quadratic moment), ml (linear mass), for each index wich represent the position on the beam.
            stiffness_elementary_matrix_function (Callable): function that returns the stiffness elementary matrix for a given element. should be a function of the length of the element, the EI at the beginning of the element and the EI at the end of the element.
            mass_elementary_matrix_function (Callable): function that returns the mass elementary matrix for a given element. should be a function of the length of the element, the linear mass at the beginning of the element and the linear mass at the end of the element.
        """

        self.beam = beam  # beam is a dict with the following keys: E, Ixx, ml,
        self.mass_elementary_matrix_function = mass_elementary_matrix_function
        self.stiffness_elementary_matrix_function = stiffness_elementary_matrix_function
        self.Ne = beam.shape[0] - 1  # number of elements
        self.EI = beam["E"].values * beam["Ixx"].values
        self.ml = beam["ml"].values
        self.le = beam.index.values[1:] - beam.index.values[:-1]

        self.nddl = (self.Ne + 1) * 2
        self.ddl_per_element = 2

        # check type:
        assert isinstance(
            self.beam, pandas.core.frame.DataFrame
        ), "beam should be a dict"
        assert isinstance(
            self.mass_elementary_matrix_function, Callable
        ), "mass_elementary_matrix_function should be a Callable"
        assert isinstance(
            self.stiffness_elementary_matrix_function, Callable
        ), "stiffness_elementary_matrix_function should be a Callable"

    def K_M_Q(self) -> Tuple[ndarray, ndarray, ndarray]:
        """This function returns the global stiffness matrix, the global mass matrix and the global force vector using the fem1d_matrix function.

        Returns:
            Tuple[ndarray, ndarray, ndarray]: global stiffness matrix, the global mass matrix and the global force vector.
        """
        k, m, q = self.fem1d_matrix()

        return k, m, q

    def fem1d_matrix(self) -> Tuple[ndarray, ndarray, ndarray]:
        """This function returns the global stiffness matrix, the global mass matrix and the global force vector using the fem1d_matrix function.

        Returns:
            Tuple[ndarray, ndarray, ndarray]: global stiffness matrix, the global mass matrix and the global force vector.
        """
        k_fem_matrix = np.zeros((self.nddl, self.nddl))
        m_fem_matrix = np.zeros((self.nddl, self.nddl))

        for jje in range(self.Ne):  # loop over the elements

            # index of the global matrix
            idx = np.arange(4) + 2 * jje

            # EI and ml at the beginning and the end of the element:
            EI1 = self.EI[jje]
            EI2 = self.EI[jje + 1]
            ml1 = self.ml[jje]
            ml2 = self.ml[jje + 1]
            le = self.le[jje]

            # stiffness and mass elementary matrix:
            kel = self.stiffness_elementary_matrix_function(le, EI1, EI2)
            mel = self.mass_elementary_matrix_function(le, ml1, ml2)

            # assembly of the global matrix:
            k_fem_matrix[np.ix_(idx, idx)] += kel
            m_fem_matrix[np.ix_(idx, idx)] += mel

            #
            q = 0

        return k_fem_matrix, m_fem_matrix, q
