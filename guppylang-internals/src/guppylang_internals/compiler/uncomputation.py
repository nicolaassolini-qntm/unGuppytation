"""Scalar SSA uncomputation, following Figure 7 of arXiv:2406.14227.

This layer knows neither Python places nor HUGR wires. A register denotes a *value*,
not a physical qubit. Conserved arguments are computational-basis controls, not
ordinary Guppy mutable borrows. The adapter supplies the operation contracts.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace


class UncomputationError(RuntimeError):
    """The requested uncomputation is unsupported or has unavailable dependencies."""


class UncomputationSynthesisError(UncomputationError):
    """A failed precondition, with the producer path that required its value."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.producers: list[Statement] = []


class UncomputationDependencyError(UncomputationSynthesisError):
    """An input value has no available local producer."""


@dataclass(frozen=True)
class Operation:
    name: str
    qfree: bool = False
    daggerable: bool = False


ALLOC = Operation("alloc", qfree=True, daggerable=True)
DUP = Operation("dup", qfree=True, daggerable=True)
FORGET = Operation("forget")


@dataclass(eq=False)
class Statement:
    op: Operation
    conserved: tuple[int, ...] = ()
    consumed: tuple[int, ...] = ()
    produced: tuple[int, ...] = ()
    dagger: bool = False
    payload: object = None
    # Retained for a future garbage-erasure lowering. Pairing by identity avoids
    # accidentally pairing two syntactically identical calls.
    partner: "Statement | None" = field(default=None, repr=False)

    def substitute(self, mapping: dict[int, int]) -> None:
        self.conserved = tuple(mapping.get(r, r) for r in self.conserved)
        self.consumed = tuple(mapping.get(r, r) for r in self.consumed)
        self.produced = tuple(mapping.get(r, r) for r in self.produced)


def validate(
    statements: Sequence[Statement], inputs: Sequence[int], outputs: Sequence[int]
) -> None:
    """Check SSA definitions, availability, aliasing and linear consumption."""
    alive = set(inputs)
    defined = set(inputs)
    if len(alive) != len(inputs) or len(set(outputs)) != len(outputs):
        raise UncomputationError("Aliased input or output registers")
    for s in statements:
        uses = s.conserved + s.consumed
        if len(set(uses)) != len(uses):
            raise UncomputationError(f"Aliased quantum arguments to {s.op.name}")
        if not set(uses) <= alive:
            raise UncomputationError(f"Unavailable quantum input to {s.op.name}")
        if set(s.produced) & defined or len(set(s.produced)) != len(s.produced):
            raise UncomputationError(f"Repeated SSA definition in {s.op.name}")
        alive.difference_update(s.consumed)
        alive.update(s.produced)
        defined.update(s.produced)
    if alive != set(outputs):
        raise UncomputationError(
            "Quantum values must be returned, consumed or forgotten"
        )


def synthesize(
    statements: Sequence[Statement], inputs: Sequence[int], outputs: Sequence[int]
) -> list[Statement]:
    """Replace forget with inverses, extending dependency lifetimes as necessary.

    Each original producer is undone at most once. Mutation is confined to copies;
    the input program can be reused after either success or failure. Dup is a
    computational-basis copy (QAlloc + CX), never an arbitrary quantum clone.
    """
    validate(statements, inputs, outputs)
    original = [replace(s, partner=None) for s in statements]
    result = original.copy()
    producers = {r: s for s in original for r in s.produced}
    next_register = max([*inputs, *producers], default=-1) + 1
    alive = set(outputs)
    undone: set[Statement] = set()
    extended: dict[int, int] = {}
    restored: list[tuple[int, int, Statement]] = []

    def fresh() -> int:
        nonlocal next_register
        r = next_register
        next_register += 1
        return r

    def ensure(anchor: Statement, r: int) -> None:
        if r not in producers:
            raise UncomputationDependencyError(
                f"Cannot forget register {r}: dependency is a consumed function "
                "or basic-block input"
            )
        compute = producers[r]
        if compute in undone:
            return
        if not compute.op.qfree:
            error = UncomputationSynthesisError(
                f"Cannot forget register {r}: {compute.op.name} is not known to "
                "preserve the computational basis without phases"
            )
            error.producers.append(compute)
            raise error
        if not compute.op.daggerable:
            raise UncomputationError(
                f"Cannot forget register {r}: {compute.op.name} has no dagger"
            )
        undone.add(compute)
        try:
            for dep in compute.conserved:
                if dep not in alive:
                    ensure(anchor, dep)
            for dep in compute.consumed:
                ensure(anchor, dep)
        except UncomputationSynthesisError as error:
            error.producers.append(compute)
            raise

        undo = Statement(
            compute.op,
            compute.conserved,
            tuple(fresh() for _ in compute.produced),
            tuple(fresh() for _ in compute.consumed),
            not compute.dagger,
            compute.payload,
            compute,
        )
        compute.partner = undo
        copies = []
        for r, copy in zip(compute.produced, undo.consumed, strict=True):
            if r in alive:
                copies.append(Statement(DUP, conserved=(r,), produced=(copy,)))
            else:
                extended[r] = copy
        pos = result.index(anchor) + 1
        result[pos:pos] = [*copies, undo]
        restored.extend(
            (r, copy, undo)
            for r, copy in zip(compute.consumed, undo.produced, strict=True)
        )

    for s in reversed(original):
        if s.op == FORGET:
            if s.conserved or s.produced or len(s.consumed) != 1 or s.dagger:
                raise UncomputationError("Malformed forget statement")
            ensure(s, s.consumed[0])
        alive.difference_update(s.produced)
        alive.update(s.consumed)

    for r, copy, undo in restored:
        pos = result.index(undo) + 1
        result.insert(
            pos, Statement(DUP, conserved=(r,), consumed=(copy,), dagger=True)
        )

    # Resolve the extended lifetimes in a separate forward pass. In particular,
    # rewrites do not affect the producer/dependency graph used above.
    mapping: dict[int, int] = {}
    resolved = []
    for s in result:
        if s.op == FORGET:
            r = s.consumed[0]
            mapping[extended.pop(r)] = r
            continue
        after = {}
        for r in s.consumed:
            if r in extended:
                inverse_input = extended.pop(r)
                copy = fresh()
                resolved.append(Statement(DUP, conserved=(r,), produced=(copy,)))
                mapping[r] = copy
                after[inverse_input] = r
                after[r] = r
        s.substitute(mapping)
        mapping.update(after)
        resolved.append(s)
    if extended:
        raise UncomputationError("Unresolved uncomputation lifetimes")
    validate(resolved, inputs, outputs)
    return resolved
