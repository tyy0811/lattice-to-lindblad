import numpy as np
from numpy.linalg import eigh
from functools import reduce

# ----------------------------
# Utilities
# ----------------------------
def kron_all(mats):
    """Kronecker product of a list of matrices."""
    return reduce(np.kron, mats)

def embed_1site(N, site, op, I2):
    """Embed a 2x2 single-site operator into an N-qubit Hilbert space."""
    ops = [I2] * N
    ops[site] = op
    return kron_all(ops)

def embed_2site(N, site, op2, I2):
    """
    Embed a 4x4 nearest-neighbor operator acting on (site, site+1)
    into an N-qubit Hilbert space.
    """
    if site < 0 or site >= N - 1:
        raise ValueError("site must satisfy 0 <= site <= N-2 for a 2-site embedding.")
    ops = []
    s = 0
    while s < N:
        if s == site:
            ops.append(op2)  # 4x4 block replaces sites (site, site+1)
            s += 2
        else:
            ops.append(I2)
            s += 1
    return kron_all(ops)

def staggered_vacuum_state(N):
    """
    |Omega> = |0101...> in computational basis (|0> empty, |1> occupied).
    With Z|0>=+|0>, Z|1>=-|1>, this corresponds to n=(1-Z)/2.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for the staggered vacuum |0101...>.")
    dim = 2 ** N
    bits = [(i % 2) for i in range(N)]  # 0,1,0,1,...
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    psi = np.zeros(dim, dtype=complex)
    psi[idx] = 1.0
    return psi

# ----------------------------
# Schwinger (gauge-eliminated) Hamiltonian
# ----------------------------
def get_spin_H_matrix(N, m, g, w=1.0, E0=0.0, hopping_convention="fermionic"):
    """
    Constructs the N-site gauge-eliminated Schwinger Hamiltonian matrix.

    Parameters
    ----------
    N : int
        Number of lattice sites (must be even).
    m : float
        Staggered mass parameter.
    g : float
        Gauge coupling (electric term coefficient uses g^2/2).
    w : float
        Hopping strength parameter.
    E0 : float
        Background electric field (use E0=0 for vacuum; E0!=0 for string backgrounds).
    hopping_convention : {"fermionic", "xx_yy"}
        - "fermionic": interpret w as the coefficient of -w*(c†c + h.c.),
          which maps to -(w/2)*(XX+YY) in qubits.
        - "xx_yy": interpret w as the coefficient directly multiplying (XX+YY),
          i.e. H_hop = -w*(XX+YY).

    Notes
    -----
    - Qubit convention: Z|0>=+|0>, Z|1>=-|1>.
      Fermion occupation: n = (1 - Z)/2.
    - Charge operator (with staggered background subtraction):
        q_n = ((-1)^n - Z_n) / 2
    - Electric field (open boundary):
        E_n = E0 + sum_{i<=n} q_i
      Electric energy:
        H_el = (g^2/2) * sum_{n=0}^{N-2} E_n^2
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")
    dim = 2 ** N

    # Pauli matrices (complex dtype for safety)
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Local operators
    n_op = 0.5 * (I2 - Z)  # number operator

    # Precompute embedded single-site ops we need
    Z_full = [embed_1site(N, i, Z, I2) for i in range(N)]
    n_full = [embed_1site(N, i, n_op, I2) for i in range(N)]

    # Charge operators q_i = ((-1)^i * I - Z_i)/2
    I_full = np.eye(dim, dtype=complex)
    q_full = []
    for i in range(N):
        q_full.append(0.5 * (((-1) ** i) * I_full - Z_full[i]))

    # ----------------------------
    # 1) Hopping term
    # ----------------------------
    XX_YY_pair = np.kron(X, X) + np.kron(Y, Y)  # 4x4

    if hopping_convention == "fermionic":
        hop_coeff = w / 2.0
    elif hopping_convention == "xx_yy":
        hop_coeff = w
    else:
        raise ValueError("hopping_convention must be 'fermionic' or 'xx_yy'.")

    H_hop = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H_hop -= hop_coeff * embed_2site(N, i, XX_YY_pair, I2)

    # ----------------------------
    # 2) Mass term
    #    H_mass = m * sum_n (-1)^n * n_n
    # ----------------------------
    H_mass = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H_mass += m * ((-1) ** i) * n_full[i]

    # ----------------------------
    # 3) Electric term
    #    H_el = (g^2/2) * sum_{n=0}^{N-2} E_n^2,  E_n = E0 + sum_{i<=n} q_i
    # ----------------------------
    H_el = np.zeros((dim, dim), dtype=complex)
    for n in range(N - 1):  # links 0..N-2
        E_n = E0 * I_full
        for i in range(n + 1):
            E_n = E_n + q_full[i]
        H_el += (g ** 2 / 2.0) * (E_n @ E_n)

    H = H_hop + H_mass + H_el
    return H

# ----------------------------
# Diagnostics 
# ----------------------------
def expectation(psi, A):
    return np.vdot(psi, A @ psi)

if __name__ == "__main__":
    # --- Run Check for N=4 ---
    N = 4
    m = 0.1
    g = 0.5
    w = 1.0
    E0 = 0.0

    # 1) Build Hamiltonian (using fermionic convention: -(w/2)(XX+YY))
    H = get_spin_H_matrix(N, m, g, w=w, E0=E0, hopping_convention="fermionic")

    # Hermiticity check
    herm_err = np.linalg.norm(H - H.conj().T)
    print(f"Hermiticity ||H-H†|| = {herm_err:.3e}")

    # 2) Diagonalize
    evals, evecs = eigh(H)
    print(f"\n--- N={N} Hamiltonian Check ---")
    print(f"Parameters: m={m}, g={g}, w={w} (fermionic convention => -(w/2)(XX+YY)), E0={E0}")
    print("Lowest 5 Eigenvalues:")
    print(evals[:5].real)

    # 3) Vacuum expectation check: <Omega|H|Omega> should equal -2m for E0=0,
    #    because <Omega|H_hop|Omega>=0 (off-diagonal) and electric term is zero in the neutral staggered vacuum.
    omega = staggered_vacuum_state(N)
    E_vac_expect = expectation(omega, H).real
    print(f"\nVacuum expectation <Omega|H|Omega> = {E_vac_expect:.6f}")
    print(f"Expected <Omega|H|Omega> at E0=0 (any w): {-m * (N/2):.6f}")

    # 4) Static-limit ground state check: set w=0 and confirm ground energy is -2m (for E0=0).
    H_w0 = get_spin_H_matrix(N, m, g, w=0.0, E0=E0, hopping_convention="fermionic")
    evals_w0, _ = eigh(H_w0)
    print(f"\nStatic-limit (w=0) ground energy E0 = {evals_w0[0].real:.6f}")
    print(f"Expected ground energy at w=0, E0=0: {-m * (N/2):.6f}")

    
