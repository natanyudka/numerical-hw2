import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src. constrained_min import interior_pt
from src.examples import (
    qp_objective,
    qp_ineq_constraint_x,
    qp_ineq_constraint_y,
    qp_ineq_constraint_z,
    qp_eq_constraints_mat,
    qp_eq_constraints_rhs,
    lp_objective,
    lp_ineq_1,
    lp_ineq_2,
    lp_ineq_3,
    lp_ineq_4,
)


def run_and_plot_qp():
    """Runs the QP problem and generates the required plots and prints."""
    print("\n--- Solving Quadratic Programming Problem ---")
    x0 = np.array([0.1, 0.2, 0.7])
    ineq_constraints = [
        qp_ineq_constraint_x,
        qp_ineq_constraint_y,
        qp_ineq_constraint_z,
    ]

    x_final, central_path = interior_pt(
        qp_objective,
        ineq_constraints,
        qp_eq_constraints_mat,
        qp_eq_constraints_rhs,
        x0,
    )

    path_arr = np.array(central_path)

    # feasible region and central path plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # vertices of the feasible region, a triangle
    verts = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
    ax.add_collection3d(
        Poly3DCollection(
            verts,
            alpha=0.2,
            facecolor="cyan",
            linewidths=1,
            edgecolors="r",
        )
    )

    ax.plot(
        path_arr[:, 0],
        path_arr[:, 1],
        path_arr[:, 2],
        "o-",
        label="Central Path",
    )
    ax.plot(
        [x_final[0]],
        [x_final[1]],
        [x_final[2]],
        "r*",
        markersize=15,
        label="Final Solution",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("QP: Feasible Region and Central Path")
    ax.legend()
    plt.savefig("qp_central_path.png")
    plt.show()

    # objective value vs. iteration plot
    obj_values = [qp_objective(x)[0] for x in central_path]
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(obj_values)), obj_values, "o-")
    plt.xlabel("Outer Iteration Number")
    plt.ylabel("Objective Function Value")
    plt.title("QP: Objective Value vs. Outer Iteration")
    plt.grid(True)
    plt.savefig("qp_objective_value.png")
    plt.show()

    # --- final print line ---
    print("\n--- QP Final Results ---")
    print(f"Final solution x: {x_final}")
    obj_val, _, _ = qp_objective(x_final)
    print(f"Objective value f(x): {obj_val}")
    print("Constraint values at final candidate:")
    print(f"  x + y + z - 1 = {np.sum(x_final) - 1:.2e}")
    print(f"  -x = {-x_final[0]:.2e} (<= 0)")
    print(f"  -y = {-x_final[1]:.2e} (<= 0)")
    print(f"  -z = {-x_final[2]:.2e} (<= 0)")


def run_and_plot_lp():
    """Runs the LP problem and generates the required plots and prints."""
    print("\n--- Solving Linear Programming Problem ---")
    x0 = np.array([0.5, 0.75])
    ineq_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]

    x_final, central_path = interior_pt(
        lp_objective, ineq_constraints, None, None, x0
    )
    path_arr = np.array(central_path)

    # feasible region and central path plot
    plt.figure(figsize=(8, 8))
    x_plot = np.linspace(0, 3, 400)
    y1 = -x_plot + 1
    plt.fill_between(
        [0, 1, 2, 2, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], color="cyan", alpha=0.3
    )
    plt.plot(x_plot, y1, label="y = -x + 1")
    plt.axhline(y=1, color="gray", linestyle="--", label="y = 1")
    plt.axvline(x=2, color="gray", linestyle="--", label="x = 2")
    plt.axhline(y=0, color="gray", linestyle="--", label="y = 0")

    #central path
    plt.plot(
        path_arr[:, 0], path_arr[:, 1], "o-", color="blue", label="Central Path"
    )
    plt.plot(
        x_final[0],
        x_final[1],
        "r*",
        markersize=15,
        label="Final Solution",
    )

    plt.xlim(0, 2.5)
    plt.ylim(0, 1.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("LP: Feasible Region and Central Path")
    plt.legend()
    plt.grid(True)
    plt.savefig("lp_central_path.png")
    plt.show()

    obj_values = [lp_objective(x)[0] for x in central_path]
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(obj_values)), obj_values, "o-")
    plt.xlabel("Outer Iteration Number")
    plt.ylabel("Objective Function Value (min -x-y)")
    plt.title("LP: Objective Value vs. Outer Iteration")
    plt.grid(True)
    plt.savefig("lp_objective_value.png")
    plt.show()

    print("\n--- LP Final Results ---")
    print(f"Final solution x: {x_final}")
    # we minimized -(x+y), so max(x+y) is the negative of this
    obj_val, _, _ = lp_objective(x_final)
    print(f"Objective value min(-x-y): {obj_val:.6f}")
    print(f"Objective value max(x+y): {-obj_val:.6f}")
    print("Constraint values at final candidate:")
    print(f"  -x - y + 1 = {-x_final[0] - x_final[1] + 1:.2e} (<= 0)")
    print(f"  y - 1 = {x_final[1] - 1:.2e} (<= 0)")
    print(f"  x - 2 = {x_final[0] - 2:.2e} (<= 0)")
    print(f"  -y = {-x_final[1]:.2e} (<= 0)")


if __name__ == "__main__":
    run_and_plot_qp()
    run_and_plot_lp()