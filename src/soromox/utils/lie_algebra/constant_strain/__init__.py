r"""Closed-form Lie operators for constant-strain rod segments.

The algebra-specific :mod:`.se2` and :mod:`.se3` namespaces provide concise
public functions for planar and spatial rod kinematics. Both modules return
:class:`ConstantStrainOperators` from their bundled ``operators`` evaluator.
"""

from . import se2, se3
from ._shared import ConstantStrainOperators

__all__ = [
    "ConstantStrainOperators",
    "se2",
    "se3",
]
