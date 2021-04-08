import numpy as np
from qpsolvers import solve_qp


def main():
    #### Order 2 Polynomial, No Velocity Constraints ####

    P2 = np.array([[1, 1], [1, 4/3]])
    A2 = np.array([1, 1])
    b2 = np.array([1])
    G2 = np.zeros((2, 2))
    h2 = np.zeros((2,))
    q2 = np.zeros((2,))

    p2_star = solve_qp(P2, q2, G2, h2, A2, b2)
    print("QP for polynomial of degree 2, no velocity constraints")
    print("Optimal value of cost function: {}".format(p2_star.T @ P2 @ p2_star))
    print("QP Solver for N=2, No Velocity Constraints: {} \n".format(p2_star))


    #### Order 3 Polynomial, No Velocity Constraints ####

    P3 = np.array([[1, 1, 1], [1, 4/3, 5/4], [1, 5/4, 9/5]])
    A3 = np.array([1, 1, 1])
    b3 = np.array([1])
    G3 = np.zeros((3, 3))
    h3 = np.zeros((3,))
    q3 = np.zeros((3,))

    p3_star = solve_qp(P3, q3, G3, h3, A3, b3)
    print("QP for polynomial of degree 3, no velocity constraints")
    print("Optimal value of cost function: {}".format(p3_star.T @ P3 @ p3_star))
    print("QP Solver for N=3, No Velocity Constraints: {} \n".format(p3_star))


    #### Order 2 Polynomial, Velocity Constraints ####

    P2v = np.array([[1, 1], [1, 4/3]])
    A2v = np.array([[1, 2], [1, 1]])
    b2v = np.array([-2, 1]).T
    G2v = np.zeros((2, 2))
    h2v = np.zeros((2,))
    q2v = np.zeros((2,))

    p2v_star = solve_qp(P2v, q2v, G2v, h2v, A2v, b2v)
    print("QP for polynomial of degree 2, velocity constraints")
    print("Optimal value of cost function: {}".format(p2v_star.T @ P2v @ p2v_star))
    print("QP Solver for N=2, Velocity Constraints: {} \n".format(p2v_star))


    #### Order 3 Polynomial, Velocity Constraints ####

    P3v = np.array([[1, 1, 1], [1, 4/3, 5/4], [1, 5/4, 9/5]])
    A3v = np.array([[1, 2, 3], [1, 1, 1]])
    b3v = np.array([-2, 1]).T
    G3v = np.zeros((3, 3))
    h3v = np.zeros((3,))
    q3v = np.zeros((3,))

    p3v_star = solve_qp(P3v, q3v, G3v, h3v, A3v, b3v)
    print("QP for polynomial of degree 3, velocity constraints")
    print("Optimal value of cost function: {}".format(p3v_star.T @ P3v @ p3v_star))
    print("QP Solver for N=3, Velocity Constraints: {} \n".format(p3v_star))

if __name__ == "__main__":
    main()

