"""End-to-end coverage of explicit scalar uncomputation."""

import pytest

from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit, toffoli, x
from guppylang.std.uncomputation import forget


def test_alloc_forget(validate):
    @guppy
    def main() -> None:
        q = qubit()
        forget(q)

    validate(main.compile_function())


def test_compute_use_forget(validate, run_int_fn):
    @guppy
    def clean(a: qubit, result: qubit) -> None:
        tmp = qubit()
        cx(a, tmp)
        cx(tmp, result)
        forget(tmp)

    @guppy
    def main() -> int:
        a, result = qubit(), qubit()
        x(a)
        clean(a, result)
        av = measure(a)
        rv = measure(result)
        return int(bool(av)) + int(bool(rv))

    validate(main.compile_function())
    run_int_fn(main, 2, num_qubits=16)


def test_coherent_cleanup(run_int_fn):
    @guppy
    def clean(a: qubit) -> None:
        tmp = qubit()
        cx(a, tmp)
        forget(tmp)

    @guppy
    def main() -> int:
        a = qubit()
        h(a)
        clean(a)
        h(a)
        return int(bool(measure(a)))

    run_int_fn(main, 0, num_qubits=16)


def test_forget_quantum_effect():
    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        forget(q)

    with pytest.raises(RuntimeError, match="computational basis"):
        main.compile_function()


def test_unavailable_input():
    @guppy
    def clean(a: qubit) -> None:
        tmp = qubit()
        cx(a, tmp)
        x(a)
        forget(tmp)

    with pytest.raises(RuntimeError, match="consumed function or basic-block input"):
        clean.compile_function()


def test_modular_helper(validate, run_int_fn):
    @guppy(unitary=True)
    def compute(a: qubit, target: qubit) -> None:
        cx(a, target)
        x(target)

    @guppy
    def clean(a: qubit, result: qubit) -> None:
        tmp = qubit()
        compute(a, tmp)
        cx(tmp, result)
        forget(tmp)

    @guppy
    def main() -> int:
        a, result = qubit(), qubit()
        clean(a, result)
        av = measure(a)
        rv = measure(result)
        return int(bool(av)) + int(bool(rv))

    validate(main.compile_function())
    run_int_fn(main, 1, num_qubits=16)


def test_forgotten_dependency_and_return(validate, run_int_fn):
    @guppy
    def chain(a: qubit) -> qubit:
        first = qubit()
        cx(a, first)
        second = qubit()
        cx(first, second)
        result = qubit()
        cx(second, result)
        forget(first)
        forget(second)
        return result

    @guppy
    def main() -> int:
        a = qubit()
        h(a)
        result = chain(a)
        cx(a, result)
        h(a)
        av = measure(a)
        rv = measure(result)
        return int(bool(av)) + int(bool(rv))

    validate(main.compile_function())
    run_int_fn(main, 0, num_qubits=16)


def test_live_output_of_modular_call(run_int_fn):
    @guppy(unitary=True)
    def pair(a: qubit, b: qubit, c: qubit) -> None:
        cx(a, b)
        cx(a, c)

    @guppy
    def clean(a: qubit) -> qubit:
        b = qubit()
        c = qubit()
        pair(a, b, c)
        forget(b)
        return c

    @guppy
    def main() -> int:
        a = qubit()
        h(a)
        c = clean(a)
        cx(a, c)
        h(a)
        av = measure(a)
        cv = measure(c)
        return int(bool(av)) + int(bool(cv))

    run_int_fn(main, 0, num_qubits=16)


@pytest.mark.parametrize("bit", [False, True])
def test_toffoli_and_classical_arguments(bit, run_int_fn):
    @guppy(unitary=True)
    def compute(unused: int, a: qubit, b: qubit, target: qubit) -> None:
        toffoli(a, b, target)

    @guppy
    def clean(a: qubit, b: qubit, target: qubit) -> None:
        n = 7
        tmp = qubit()
        compute(n, a, b, tmp)
        n = 9
        cx(tmp, target)
        forget(tmp)

    @guppy
    def main(bit: bool) -> int:
        a, b, target = qubit(), qubit(), qubit()
        if bit:
            x(a)
            x(b)
        clean(a, b, target)
        av, bv, tv = measure(a), measure(b), measure(target)
        return int(bool(av)) + int(bool(bv)) + int(bool(tv))

    run_int_fn(main, 3 if bit else 0, args=[bit], num_qubits=16)


def test_noncontrollable_operation():
    @guppy
    def main() -> bool:
        q = qubit()
        other = qubit()
        result = measure(other)
        forget(q)
        return bool(result)

    with pytest.raises(RuntimeError, match="not controllable"):
        main.compile_function()


def test_nested_measurement_rejected():
    @guppy
    def main() -> bool:
        q = qubit()
        other = qubit()
        result = bool(measure(other))
        forget(q)
        return result

    with pytest.raises(RuntimeError, match="Nested quantum"):
        main.compile_function()


def test_missing_dagger():
    @guppy(controllable=True)
    def compute(q: qubit) -> None:
        x(q)

    @guppy
    def main() -> None:
        q = qubit()
        compute(q)
        forget(q)

    with pytest.raises(RuntimeError, match="no dagger"):
        main.compile_function()


def test_quantum_effect_on_unrelated_output(validate):
    @guppy
    def clean(a: qubit, b: qubit) -> None:
        tmp = qubit()
        cx(a, tmp)
        h(b)
        forget(tmp)

    validate(clean.compile_function())


def test_qubit_move_and_import_alias(validate):
    from guppylang.std.uncomputation import forget as cleanup

    @guppy
    def clean(a: qubit) -> None:
        q = qubit()
        cx(a, q)
        moved = q
        cleanup(moved)

    validate(clean.compile_function())


def test_nested_modifier_rejected():
    from guppylang.std.builtins import dagger

    @guppy
    def clean(a: qubit) -> None:
        q = qubit()
        cx(a, q)
        with dagger:
            x(a)
        forget(q)

    with pytest.raises(RuntimeError, match="Nested blocks and modifiers"):
        clean.compile_function()


def test_cross_block_dependency_rejected():
    @guppy
    def clean(a: qubit, flag: bool) -> None:
        q = qubit()
        if flag:
            cx(a, q)
        forget(q)

    with pytest.raises(RuntimeError, match="basic-block input"):
        clean.compile_function()


def test_allocating_helper_rejected():
    @guppy
    def allocate() -> qubit:
        return qubit()

    @guppy
    def main() -> None:
        q = allocate()
        forget(q)

    with pytest.raises(RuntimeError, match="garbage-aware adjoints"):
        main.compile_function()


def test_name_is_not_a_purity_contract():
    @guppy(unitary=True)
    def x(q: qubit) -> None:
        h(q)

    @guppy
    def main() -> None:
        q = qubit()
        x(q)
        forget(q)

    with pytest.raises(RuntimeError, match="computational basis"):
        main.compile_function()


@pytest.mark.parametrize("superposed", [False, True])
@pytest.mark.parametrize("prepare_inside", [False, True])
def test_helper_updates_source_before_copy(
    superposed, prepare_inside, validate, run_int_fn, tmp_path
):
    @guppy(unitary=True)
    def use_cx(a: qubit, b: qubit) -> None:
        x(a)
        cx(a, b)

    @guppy
    def clean_copy(a: qubit, result: qubit) -> None:
        temporary = qubit()
        use_cx(a, temporary)
        cx(temporary, result)
        forget(temporary)

    if prepare_inside:

        @guppy
        def clean_copy(a: qubit, result: qubit) -> None:
            temporary = qubit()
            h(a)
            use_cx(a, temporary)
            cx(temporary, result)
            forget(temporary)

    # Exercise the user's compilation/export/render workflow as well as execution.
    package = clean_copy.compile_function()
    validate(package)
    program = package.modules[0]
    (tmp_path / "complex_guppy.hugr").write_bytes(program.to_bytes())
    assert program.render_dot().source

    @guppy
    def main(superposed: bool, prepare_inside: bool) -> int:
        a, result = qubit(), qubit()
        if superposed:
            h(a)
        clean_copy(a, result)
        cx(a, result)
        # Undo only the X on the surviving a, then undo state preparation.
        x(a)
        if prepare_inside:
            h(a)
        if superposed:
            h(a)
        av, rv = measure(a), measure(result)
        return int(bool(av)) + int(bool(rv))

    run_int_fn(main, 0, args=[superposed, prepare_inside], num_qubits=16)


def test_helper_nested_update_with_classical_argument(validate, run_int_fn):
    @guppy(unitary=True)
    def inner(unused: int, a: qubit, b: qubit) -> None:
        x(a)
        cx(a, b)

    @guppy(unitary=True)
    def outer(n: int, a: qubit, b: qubit) -> None:
        inner(n + 1, a, b)

    @guppy
    def clean(a: qubit, result: qubit) -> None:
        tmp = qubit()
        outer(2, a, tmp)
        cx(tmp, result)
        forget(tmp)

    @guppy
    def main() -> int:
        a, result = qubit(), qubit()
        clean(a, result)
        av, rv = measure(a), measure(result)
        return int(bool(av)) + int(bool(rv))

    validate(main.compile_function())
    run_int_fn(main, 2, num_qubits=16)


def test_helper_copy_before_source_update_remains_rejected():
    @guppy(unitary=True)
    def use_cx(a: qubit, b: qubit) -> None:
        cx(a, b)
        x(a)

    @guppy
    def clean(a: qubit) -> None:
        tmp = qubit()
        use_cx(a, tmp)
        forget(tmp)

    # This fallback exposes existing SSA provenance; it does not invent a way to
    # reconstruct an unavailable version by inverting arbitrary live outputs.
    with pytest.raises(RuntimeError, match="basic-block input"):
        clean.compile_function()


def test_helper_loses_source_information_rejected():
    @guppy(unitary=True)
    def move_information(a: qubit, b: qubit) -> None:
        cx(a, b)
        cx(b, a)

    @guppy
    def clean(a: qubit) -> None:
        tmp = qubit()
        move_information(a, tmp)
        forget(tmp)

    # a ends at zero; tmp contains the sole copy of the original input. Forgetting
    # it cannot preserve arbitrary input states and must never be accepted.
    with pytest.raises(RuntimeError, match="basic-block input"):
        clean.compile_function()
