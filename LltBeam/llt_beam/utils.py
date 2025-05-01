from typing import Callable, Tuple

import sympy as sym
import numpy as np
from numpy import int64, ndarray, single
from scipy.linalg import eig
import scipy


def get_elementary_stiffness_matrix_function() -> Callable:
    """This function returns the stiffness elementary matrix function for a given element. should be a function of the length of the element, the EI at the beginning of the element and the EI at the end of the element.

    Returns:
        Callable: stiffness elementary matrix function.
    """

    l = sym.Symbol("l")
    xi = sym.Symbol("xi")
    Phi = sym.Matrix(
        [
            1 - 3 * xi**2 + 2 * xi**3,
            l * xi * (1 - xi) ** 2,
            xi**2 * (3 - 2 * xi),
            l * xi**2 * (xi - 1),
        ]
    )
    dd_Phi = Phi.diff(xi, 2)

    EI1 = sym.Symbol("EI1")
    EI2 = sym.Symbol("EI2")

    EI1EI2 = sym.Matrix([[EI1, EI2]])
    Emat = sym.Matrix([[1 - xi], [xi]])

    k_integrale = (1 / l**3) * sym.integrate(
        (EI1EI2 * Emat)[0, 0] * dd_Phi * dd_Phi.T, (xi, 0, 1)
    )
    kelemf = sym.lambdify((l, EI1, EI2), k_integrale)

    return kelemf


def get_elementary_mass_matrix_function() -> Callable:
    """This function returns the mass elementary matrix function for a given element. should be a function of the length of the element, the linear mass at the beginning of the element and the linear mass at the end of the element.

    Returns:
        _type_: mass elementary matrix function.
    """
    l = sym.Symbol("l")
    xi = sym.Symbol("xi")
    Phi = sym.Matrix(
        [
            1 - 3 * xi**2 + 2 * xi**3,
            l * xi * (1 - xi) ** 2,
            xi**2 * (3 - 2 * xi),
            l * xi**2 * (xi - 1),
        ]
    )

    m1 = sym.Symbol("m1")
    m2 = sym.Symbol("m2")

    m1m2 = sym.Matrix([[m1, m2]])
    mmat = sym.Matrix([[1 - xi], [xi]])

    m_integral = l * sym.integrate((m1m2 * mmat)[0, 0] * Phi * Phi.T, (xi, 0, 1))
    melemf = sym.lambdify((l, m1, m2), m_integral)

    return melemf


def get_force_vector_function() -> Callable:
    """This function returns the force vector elementary function for a given element. should be a function of the length of the element, the linear force at the beginning of the element and the linear force at the end of the element.

    Returns:
        Callable: force vector elementary function.
    """
    l = sym.Symbol("l")
    xi = sym.Symbol("xi")
    Phi = sym.Matrix(
        [
            1 - 3 * xi**2 + 2 * xi**3,
            l * xi * (1 - xi) ** 2,
            xi**2 * (3 - 2 * xi),
            l * xi**2 * (xi - 1),
        ]
    )

    p1 = sym.Symbol("p1")
    p2 = sym.Symbol("p2")

    p1p2 = sym.Matrix([[p1, p2]])
    pmat = sym.Matrix([[1 - xi], [xi]])

    pintegral = l * sym.integrate(Phi * (p1p2 * pmat)[0, 0], (xi, 0, 1))

    pf = sym.lambdify((l, p1, p2), pintegral)


def get_proper_mode(K: ndarray, M: ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function returns the eigenvalues and the eigenvectors of the generalized eigenvalue problem K@r = eigenvalues*M@r

    Args:
        K (ndarray): stiffness matrix.
        M (ndarray): mass matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: proper frequency and proper vectors.
    """

    K_mode_propre = K.copy()
    eigenvalues, right_eigenvectors = eig(K_mode_propre, M)
    print(" shape eigenvalues", eigenvalues.shape)
    print(" shape right_eigenvectors", right_eigenvectors.shape)

    # check the results is correct:
    print(
        "check resolve equation K@r = eigenvalues*M@r : ",
        np.allclose(
            K @ right_eigenvectors[:, 0],
            eigenvalues[0] * M @ right_eigenvectors[:, 0],
            atol=1e-03,
            equal_nan=False,
        ),
    )

    eigenvalues = np.abs(np.real(eigenvalues))
    sorted_indice = np.argsort(eigenvalues)

    sorted_eigenvalues = eigenvalues[sorted_indice]
    sorted_freq = np.sqrt(sorted_eigenvalues) / (2 * np.pi)
    sorted_right_eigenvectors = right_eigenvectors[:, sorted_indice]

    return sorted_freq, sorted_right_eigenvectors


def get_intial_static_displacement(K: ndarray, F: ndarray):
    """This function returns the static displacement of the system for a given stiffness matrix and force vector.

    Args:
        K (ndarray): stiffness matrix [N,N].
        F (ndarray): force vector [N, 1]

    Returns:
        U (ndarray): static displacement.
    """

    K_static = K.copy()
    invK1_static = np.linalg.inv(K_static)
    U_static = (invK1_static @ F[:, 0])[:, None]

    return U_static


def get_P_matrix(beam: object) -> np.ndarray:
    """function to calculate the P matrix for the beam model.

    Args:
        beam (object): object of the beam model.

    Returns:
        np.ndarray: array of the P matrix with shape (nb_dofs, nb_dofs).
    """
    P = np.zeros((beam.nddl, beam.Ne + 1))
    for i in range(beam.Ne):
        le = beam.le[i]
        P[
            i * beam.ddl_per_element : (i + 1) * beam.ddl_per_element
            + beam.ddl_per_element,
            i : (i + 1) + 1,
        ] += le * np.array(
            [
                [7 / 20, 3 / 20],
                [le / 20, le / 30],
                [3 / 20, 7 / 20],
                [-le / 30, -le / 20],
            ]
        )
    return P


def get_pf():

    p1, p2, xi, l = sym.symbols("p1 p2 xi l")

    Phi = sym.Matrix(
        [
            1 - 3 * xi**2 + 2 * xi**3,
            l * xi * (1 - xi) ** 2,
            xi**2 * (3 - 2 * xi),
            l * xi**2 * (xi - 1),
        ]
    )

    p1p2 = sym.Matrix([[p1, p2]])
    pmat = sym.Matrix([[1 - xi], [xi]])

    pintegral = l * sym.integrate(Phi * (p1p2 * pmat)[0, 0], (xi, 0, 1))

    pf = sym.lambdify((l, p1, p2), pintegral)
    return pf


def get_strain_displacement_matrix(xi, l):
    """
    Calculate the strain-displacement matrix B for a beam element.

    Args:
    xi : float
        The normalized coordinate (xi = x / l, 0 <= xi <= 1).
    l : float
        The length of the beam element.

    Returns:
    B : numpy array
        The strain-displacement matrix B.
    """
    B = (
        np.array(
            [
                [
                    6 * (2 * xi - 1),
                    2 * l * (3 * xi - 2),
                    6 * (1 - 2 * xi),
                    2 * l * (3 * xi - 1),
                ]
            ]
        )
        / l**2
    )
    return B


def elementary_B_matrix(beam):
    """
    Assemble the global strain-displacement matrix for a 3-element beam.

    Args:
    element_lengths : list of floats
        List of lengths of the elements.
    xi_values : list of floats
        List of xi values to evaluate the strain-displacement matrix for each element.

    Returns:
    B_global : numpy array
        Global strain-displacement matrix for the system with 3 elements and 4 nodes.
    """
    # Initialize a 3x8 global B matrix (3 strains, 8 DOFs for 4 nodes)
    B_global = np.zeros((beam.Ne + 1, beam.ddl_per_element * (beam.Ne + 1)))

    # Local contributions from each element
    for i in range(beam.Ne + 1):

        if i == 0:
            B_global[
                i, i * beam.ddl_per_element : (i + 2) * beam.ddl_per_element
            ] += get_strain_displacement_matrix(0, beam.le[i])[0, :]
        elif i == beam.Ne:
            B_global[
                i, (i - 1) * beam.ddl_per_element : (i + 1) * beam.ddl_per_element
            ] += get_strain_displacement_matrix(1, beam.le[i - 1])[0, :]
        else:
            B_global[
                i, (i - 1) * beam.ddl_per_element : (i + 1) * beam.ddl_per_element
            ] += get_strain_displacement_matrix(1, beam.le[i - 1])[0, :]
            B_global[
                i, i * beam.ddl_per_element : (i + 2) * beam.ddl_per_element
            ] += get_strain_displacement_matrix(0, beam.le[i])[0, :]
    return B_global


def get_proper_mode(K: ndarray, M: ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function returns the eigenvalues and the eigenvectors of the generalized eigenvalue problem K@r = eigenvalues*M@r

    Args:
        K (ndarray): stiffness matrix.
        M (ndarray): mass matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: proper frequency and proper vectors.
    """

    K_mode_propre = K.copy()
    eigenvalues, right_eigenvectors = scipy.linalg.eigh(K_mode_propre, M)

    eigenvalues = np.abs(np.real(eigenvalues))
    sorted_indice = np.argsort(eigenvalues)

    sorted_eigenvalues = eigenvalues[sorted_indice]
    sorted_freq = np.sqrt(sorted_eigenvalues) / (2 * np.pi)
    sorted_right_eigenvectors = right_eigenvectors[:, sorted_indice]

    return sorted_freq, sorted_right_eigenvectors


def mass_normalisation(
    M: np.ndarray, eigvecs: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """
    This function normalizes the eigenvectors of the system by the mass matrix.

    Args:
        M (np.ndarray): mass matrix.
        eigvecs (np.ndarray): eigenvectors.

    Returns:
        np.ndarray: normalized eigenvectors.
    """
    # Normalize the eigenvectors
    mass_wise_eigvecs = np.zeros_like(eigvecs)
    for i in range(eigvecs.shape[0]):
        mass_wise_eigvecs[:, i] = eigvecs[:, i] / np.sqrt(
            eigvecs[:, i].T @ M @ eigvecs[:, i]
        )
    if verbose:
        for i in range(eigvecs.shape[0]):
            print(
                f"normed mode {i} : {mass_wise_eigvecs[:,i]@M@mass_wise_eigvecs[:,i].T:.2f}"
            )
    return mass_wise_eigvecs
