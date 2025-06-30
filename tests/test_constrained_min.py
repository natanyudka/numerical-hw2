import unittest
import numpy as np


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


from src.constrained_min import interior_pt

class TestConstrainedMinimization(unittest.TestCase):
    def test_qp(self):
        """
        Tests the interior point solver on the QP example.
        The main purpose is to run the solver and check that it completes.
        The actual analysis will be done via plotting.
        """
        print("\n--- Running Quadratic Programming (QP) Test ---")
        x0 = np.array([0.1, 0.2, 0.7])
        ineq_constraints = [
            qp_ineq_constraint_x,
            qp_ineq_constraint_y,
            qp_ineq_constraint_z,
        ]

        x_final, _ = interior_pt(
            qp_objective,
            ineq_constraints,
            qp_eq_constraints_mat,
            qp_eq_constraints_rhs,
            x0,
        )

        self.assertIsNotNone(x_final)
        np.testing.assert_almost_equal(
            qp_eq_constraints_mat @ x_final, qp_eq_constraints_rhs, decimal=5
        )
        print(f"QP test completed.")

    def test_lp(self):
        """
        Tests the interior point solver on the LP example.
        """
        print("\n--- Running Linear Programming (LP) Test ---")
        x0 = np.array([0.5, 0.75])
        ineq_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]

        # no equalities in this problem
        eq_mat = None
        eq_rhs = None

        x_final, _ = interior_pt(
            lp_objective, ineq_constraints, eq_mat, eq_rhs, x0
        )

        self.assertIsNotNone(x_final)
        print(f"LP test completed.")


if __name__ == "__main__":
    unittest.main()