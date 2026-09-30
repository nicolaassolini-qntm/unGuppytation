"""Experimental explicit uncomputation of scalar qubits.

``forget(q)`` requests dependency-aware cleanup; it is not measurement or discard.
Unsupported dependencies and operations raise ``RuntimeError`` during compilation.
See ``docs/uncomputation.md`` for the supported subset and algorithm details.
"""

from typing import no_type_check

from guppylang_internals.decorator import custom_function
from guppylang_internals.std._internal.compiler.uncomputation import ForgetCompiler
from guppylang_internals.tys import Effect

from guppylang.std.lang import owned
from guppylang.std.quantum import qubit


@custom_function(ForgetCompiler(), higher_order_value=False, effects=[Effect.ANY])
@no_type_check
def forget(q: qubit @ owned) -> None:
    """Uncompute a qubit using its producers, then deallocate it in the zero state.

    Currently supports scalar qubits computed within one basic block. No purity
    annotation is required. Basis-preserving producers are recognised conservatively;
    controllability by itself does not establish that a value may be forgotten.
    """
