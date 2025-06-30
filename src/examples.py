import numpy as np

# QP example 
# min x^2 + y^2 + (z + 1)^2
# s.t. x + y + z = 1
# x, y, z >= 0


def qp_objective(x):
    """objective function for the QP problem."""
    val = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    grad = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    hess = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    return val, grad, hess


# inequality constraints must be in the form f(x) <= 0
# x >= 0  --> -x <= 0
# y >= 0  --> -y <= 0
# z >= 0  --> -z <= 0 
def qp_ineq_constraint_x(x):
    """inequality constraint -x <= 0"""
    val = -x[0]
    grad = np.array([-1.0, 0.0, 0.0])
    hess = np.zeros((3, 3))
    return val, grad, hess


def qp_ineq_constraint_y(x):
    """Inequality constraint -y <= 0"""
    val = -x[1]
    grad = np.array([0.0, -1.0, 0.0])
    hess = np.zeros((3, 3))
    return val, grad, hess


def qp_ineq_constraint_z(x):
    """Inequality constraint -z <= 0"""
    val = -x[2]
    grad = np.array([0.0, 0.0, -1.0])
    hess = np.zeros((3, 3))
    return val, grad, hess


# equality constraint: x + y + z = 1
# matrix A for Ax = b
qp_eq_constraints_mat = np.array([[1.0, 1.0, 1.0]])
# vector b for Ax = b
qp_eq_constraints_rhs = np.array([1.0])


# --- Linear Programming (LP) Example ---
# max(x + y) --> min -(x + y)
# s.t. y >= -x + 1  -->  -x - y + 1 <= 0
#      y <= 1        -->  y - 1 <= 0
#      x <= 2        -->  x - 2 <= 0
#      y >= 0        -->  -y <= 0


def lp_objective(x):
    """Objective function for the LP problem."""
    val = -x[0] - x[1]
    grad = np.array([-1.0, -1.0])
    hess = np.zeros((2, 2))
    return val, grad, hess


# Inequality constraints
def lp_ineq_1(x):
    """Inequality constraint -x - y + 1 <= 0"""
    val = -x[0] - x[1] + 1
    grad = np.array([-1.0, -1.0])
    hess = np.zeros((2, 2))
    return val, grad, hess


def lp_ineq_2(x):
    """Inequality constraint y - 1 <= 0"""
    val = x[1] - 1
    grad = np.array([0.0, 1.0])
    hess = np.zeros((2, 2))
    return val, grad, hess


def lp_ineq_3(x):
    """Inequality constraint x - 2 <= 0"""
    val = x[0] - 2
    grad = np.array([1.0, 0.0])
    hess = np.zeros((2, 2))
    return val, grad, hess


def lp_ineq_4(x):
    """Inequality constraint -y <= 0"""
    val = -x[1]
    grad = np.array([0.0, -1.0])
    hess = np.zeros((2, 2))
    return val, grad, hess