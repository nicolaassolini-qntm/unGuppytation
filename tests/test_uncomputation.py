"""Algorithm tests with an independent computational-basis interpreter."""

import itertools
import random

import pytest
from guppylang_internals.compiler.uncomputation import (
    ALLOC,
    DUP,
    FORGET,
    Operation,
    Statement,
    synthesize,
)

X = Operation("X", qfree=True, daggerable=True)
CX = Operation("CX", qfree=True, daggerable=True)
PAIR = Operation("pair", qfree=True, daggerable=True)


def evaluate(program, values):
    """Interpret basis permutations and require *every* deallocation to be zero."""
    values = values.copy()
    for s in program:
        controls = [values[r] for r in s.conserved]
        args = [values.pop(r) for r in s.consumed]
        if s.op == FORGET:
            continue  # Declarative reference semantics on basis states.
        if s.op == ALLOC:
            if s.dagger:
                assert args == [0]
                out = []
            else:
                out = [0]
        elif s.op == DUP:
            if s.dagger:
                assert args == controls
                out = []
            else:
                out = controls
        elif s.op == X:
            out = [args[0] ^ 1]
        elif s.op == CX:
            out = [args[0] ^ controls[0]]
        elif s.op == PAIR:
            out = [v ^ controls[0] for v in args]
        else:
            raise AssertionError(s.op)
        values.update(zip(s.produced, out, strict=True))
    return values


def test_previous_forget_extends_dependency():
    # b is forgotten before c, although uncomputing c still needs b.
    program = [
        Statement(DUP, conserved=(0,), produced=(1,)),
        Statement(DUP, conserved=(1,), produced=(2,)),
        Statement(FORGET, consumed=(1,)),
        Statement(FORGET, consumed=(2,)),
    ]
    plan = synthesize(program, [0], [0])
    for bit in [0, 1]:
        assert evaluate(plan, {0: bit}) == {0: bit}
    assert all(s.op != FORGET for s in plan)
    assert sum(s.dagger for s in plan) == 2


def test_live_sibling_output_is_preserved():
    program = [
        Statement(ALLOC, produced=(1,)),
        Statement(ALLOC, produced=(2,)),
        Statement(PAIR, conserved=(0,), consumed=(1, 2), produced=(3, 4)),
        Statement(FORGET, consumed=(3,)),
    ]
    plan = synthesize(program, [0], [0, 4])
    for bit in [0, 1]:
        assert evaluate(plan, {0: bit}) == {0: bit, 4: bit}
    assert sum(s.op == PAIR and s.dagger for s in plan) == 1
    assert all(s.partner.partner is s for s in plan if s.partner is not None)


@pytest.mark.parametrize("seed", range(30))
def test_random_basis_programs(seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic compiler fuzzing
    program = []
    alive = [0, 1]
    fresh = 2
    for _ in range(16):
        if len(alive) < 3 or rng.random() < 0.4:
            program.append(
                Statement(DUP, conserved=(rng.choice(alive),), produced=(fresh,))
            )
            alive.append(fresh)
            fresh += 1
        elif rng.random() < 0.5:
            target = rng.choice(alive[2:])
            control = rng.choice([r for r in alive if r != target])
            program.append(
                Statement(
                    CX, conserved=(control,), consumed=(target,), produced=(fresh,)
                )
            )
            alive[alive.index(target)] = fresh
            fresh += 1
        else:
            r = rng.choice(alive[2:])
            alive.remove(r)
            program.append(Statement(FORGET, consumed=(r,)))
    # Keep one local result alive to exercise inverses of multi-use dependencies.
    program.extend(Statement(FORGET, consumed=(r,)) for r in alive[3:])
    outputs = alive[:3]
    before = [(s.conserved, s.consumed, s.produced, s.partner) for s in program]
    plan = synthesize(program, [0, 1], outputs)
    for a, b in itertools.product([0, 1], repeat=2):
        assert evaluate(plan, {0: a, 1: b}) == evaluate(program, {0: a, 1: b})
    assert before == [(s.conserved, s.consumed, s.produced, s.partner) for s in program]
    assert len(plan) <= 6 * len(program)
    assert sum(s.partner is not None and s.dagger for s in plan) <= len(program)


@pytest.mark.parametrize(
    ("program", "inputs", "outputs", "message"),
    [
        ([Statement(FORGET, consumed=(0,))], [0], [], "dependency"),
        (
            [Statement(CX, conserved=(0,), consumed=(0,), produced=(1,))],
            [0],
            [1],
            "Aliased",
        ),
        ([Statement(ALLOC, produced=(0,))], [0], [0], "Repeated SSA"),
        ([Statement(FORGET, consumed=(1,))], [0], [0], "Unavailable"),
        ([Statement(ALLOC, produced=(0,))], [], [], "returned, consumed or forgotten"),
        (
            [
                Statement(Operation("opaque"), produced=(0,)),
                Statement(FORGET, consumed=(0,)),
            ],
            [],
            [],
            "computational basis",
        ),
        (
            [
                Statement(Operation("no dagger", qfree=True), produced=(0,)),
                Statement(FORGET, consumed=(0,)),
            ],
            [],
            [],
            "no dagger",
        ),
    ],
)
def test_invalid_programs(program, inputs, outputs, message):
    with pytest.raises(RuntimeError, match=message):
        synthesize(program, inputs, outputs)
