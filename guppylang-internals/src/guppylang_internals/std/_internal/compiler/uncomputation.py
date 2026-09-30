"""Sentinel compiler for declarative uncomputation requests."""

from hugr import Wire

from guppylang_internals.compiler.uncomputation import UncomputationError
from guppylang_internals.definition.custom import CustomInoutCallCompiler
from guppylang_internals.definition.value import CallReturnWires


class ForgetCompiler(CustomInoutCallCompiler):
    def compile_with_inouts(self, args: list[Wire]) -> CallReturnWires:
        raise UncomputationError(
            "forget must be a standalone statement in a supported uncomputation block"
        )
