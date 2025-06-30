# src/constrained_min.py
import numpy as np

# parameters for the backtracking line search
ALPHA = 0.1
BETA = 0.5
# tolerance for the outer loop, duality gap
TOLERANCE = 1e-6
# tolerance for the inner loop, newton decrement
NEWTON_TOLERANCE = 1e-6
MAX_ITER = 100


def _centering_step(
    t,
    func,
    ineq_constraints,
    eq_constraints_mat,
    eq_constraints_rhs,
    x0,
):
    """
    Solves the centering problem using Newton's method.

    This is the "inner loop" of the interior point method. It minimizes
    the log-barrier objective for a fixed t, subject to equality constraints.
    """
    x = x0
    for _ in range(MAX_ITER):
        
        f0_val, f0_grad, f0_hess = func(x)

        # inequality constraints
        phi_grad = t * f0_grad
        phi_hess = t * f0_hess
        for f_ineq in ineq_constraints:
            fi_val, fi_grad, fi_hess = f_ineq(x)
            if fi_val >= 0:
                raise ValueError(
                    "Current point is not strictly feasible for inequality constraints."
                )
            phi_grad += -1.0 / fi_val * fi_grad
            phi_hess += (
                (1.0 / fi_val**2) * np.outer(fi_grad, fi_grad)
                - (1.0 / fi_val) * fi_hess
            )

        # form and solve the KKT system to find the Newton step
        if eq_constraints_mat is not None:
            A = eq_constraints_mat
            n = len(x)
            p = A.shape[0]
            kkt_mat = np.zeros((n + p, n + p))
            kkt_mat[:n, :n] = phi_hess
            kkt_mat[:n, n:] = A.T
            kkt_mat[n:, :n] = A
            rhs = np.concatenate([-phi_grad, np.zeros(p)])
        else:
            # no equality constraints ==> unconstrained Newton
            kkt_mat = phi_hess
            rhs = -phi_grad

        try:
            solution = np.linalg.solve(kkt_mat, rhs)
            newton_step = solution[: len(x)]
        except np.linalg.LinAlgError:
            print("Warning: KKT matrix is singular. Stopping inner loop.")
            break

        # check stopping criterion (Newton decrement)
        newton_decrement_sq = -phi_grad.T @ newton_step
        if newton_decrement_sq / 2.0 <= NEWTON_TOLERANCE:
            break

        s = 1.0
        # calculate objective value at current point for line search
        phi_x = t * f0_val - sum(np.log(-f(x)[0]) for f in ineq_constraints)

        while True:
            x_new = x + s * newton_step
            is_feasible = all(f(x_new)[0] < 0 for f in ineq_constraints)
            if not is_feasible:
                s *= BETA
                continue

            f0_new_val = func(x_new)[0]
            phi_x_new = t * f0_new_val - sum(
                np.log(-f(x_new)[0]) for f in ineq_constraints
            )
            if phi_x_new <= phi_x + ALPHA * s * (-newton_decrement_sq):
                break
            s *= BETA
        x = x + s * newton_step

    return x


def interior_pt(
    func,
    ineq_constraints,
    eq_constraints_mat,
    eq_constraints_rhs,
    x0,
):
    """
    Implements the interior point method using a log-barrier.

    Args:
        func (callable): The objective function to minimize. Returns a tuple of
            (value, gradient, hessian).
        ineq_constraints (list of callables): Inequality constraints of the
            form f(x) <= 0. Each has the same interface as func.
        eq_constraints_mat (np.ndarray): Matrix A for equality constraints Ax = b.
        eq_constraints_rhs (np.ndarray): Vector b for equality constraints Ax = b.
        x0 (np.ndarray): starting point.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final optimized solution vector x.
            - list: A list of np.ndarray, representing the central path.
    """
    x = np.array(x0, dtype=float)
    t = 1.0
    mu = 10.0
    m = len(ineq_constraints)
    central_path = [x]

    print("Starting Interior Point Method...")
    print(f"Initial t={t}, mu={mu}, m={m} inequalities")
    print("-" * 40)
    print(f"Iter |   t    | Objective Value")
    print("-" * 40)

    for i in range(MAX_ITER):
        x = _centering_step(
            t,
            func,
            ineq_constraints,
            eq_constraints_mat,
            eq_constraints_rhs,
            x,
        )
        central_path.append(x)

        objective_value = func(x)[0]
        print(f"{i:4d} | {t:6.1e} | {objective_value:15.8f}")

        if m / t < TOLERANCE:
            print("-" * 40)
            print(f"Stopping. Duality gap m/t = {m/t:.2e} < {TOLERANCE:.2e}")
            break

        t *= mu

    return x, central_path