from __future__ import annotations

from typing import List

def get_qi_backend(qi_backend_name: str):
    """Return a Quantum Inspire backend using the qiskit-quantuminspire provider."""
    try:
        from qiskit_quantuminspire.qi_provider import QIProvider
    except Exception as e:
        raise RuntimeError(
            "Quantum Inspire provider not found. Install with:\n"
            "  python -m pip install qiskit-quantuminspire\n"
            "Then login once using: qi login"
        ) from e

    return QIProvider().get_backend(qi_backend_name)

def list_qi_backends() -> List[str]:
    from qiskit_quantuminspire.qi_provider import QIProvider
    return [b.name for b in QIProvider().backends()]
