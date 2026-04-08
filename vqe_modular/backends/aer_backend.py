from __future__ import annotations

def build_aer_backend(p1q: float, p2q: float, p01: float, p10: float, seed_sim: int = 123):
    """AerSimulator with depolarizing gate noise + symmetric readout error."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except Exception as e:
        raise RuntimeError("Install Aer with: python -m pip install qiskit-aer") from e

    noise_model = NoiseModel()

    if p1q > 0:
        e1 = depolarizing_error(p1q, 1)
        for g in ["x", "ry", "rz", "h", "sx", "s", "sdg"]:
            noise_model.add_all_qubit_quantum_error(e1, g)

    if p2q > 0:
        e2 = depolarizing_error(p2q, 2)
        noise_model.add_all_qubit_quantum_error(e2, "cx")

    ro = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    noise_model.add_all_qubit_readout_error(ro)

    return AerSimulator(noise_model=noise_model, seed_simulator=seed_sim)
