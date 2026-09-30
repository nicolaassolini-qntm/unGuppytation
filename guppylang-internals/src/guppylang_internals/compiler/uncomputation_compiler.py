"""Translate checked scalar-qubit statements to SSA, synthesize, and emit HUGR.

The existing DaggerModifier implements inverses of balanced calls. Allocation and
basis copies have explicit inverses. No changes to Guppy's dagger semantics occur.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from hugr import Wire, ops
from hugr import tys as ht

from guppylang_internals.ast_util import get_type, get_type_opt
from guppylang_internals.checker.core import Variable
from guppylang_internals.compiler.builder import pure
from guppylang_internals.compiler.core import CompilerContext, DFContainer
from guppylang_internals.compiler.uncomputation import (
    ALLOC,
    DUP,
    FORGET,
    Operation,
    Statement,
    UncomputationError,
    UncomputationSynthesisError,
    synthesize,
)
from guppylang_internals.definition.custom import CustomFunctionDef
from guppylang_internals.definition.value import CompiledCallableDef
from guppylang_internals.engine import DEF_STORE
from guppylang_internals.metadata.common import add_unitary_metadata
from guppylang_internals.nodes import GlobalCall, PlaceNode
from guppylang_internals.std._internal.compiler.tket_exts import MODIFIER_EXTENSION
from guppylang_internals.std._internal.compiler.uncomputation import ForgetCompiler
from guppylang_internals.std._internal.util import quantum_op
from guppylang_internals.tys import Effect
from guppylang_internals.tys.qubit import contain_qubit_ty, is_qubit_ty
from guppylang_internals.tys.ty import InputFlags, NoneType, UnitaryFlags

if TYPE_CHECKING:
    from guppylang_internals.compiler.stmt_compiler import StmtCompiler


def is_forget(node: ast.AST) -> bool:
    return (
        isinstance(node, GlobalCall)
        and isinstance(node.defn, CustomFunctionDef)
        and isinstance(node.defn.call_compiler, ForgetCompiler)
    )


def has_forget(stmts: Sequence[ast.stmt]) -> bool:
    return any(is_forget(n) for s in stmts for n in ast.walk(s))


@dataclass(frozen=True)
class Contract:
    qfree: bool
    conserved: frozenset[int] = frozenset()


class Contracts:
    """Fixed-point summaries shared with the recursive variant compiler."""

    def __init__(self, ctx: CompilerContext) -> None:
        from guppylang_internals.compiler.uncomputation_variants import Variants

        self.variants = Variants.get(ctx)

    def get(self, call: GlobalCall) -> Contract:
        result = self.variants.summary(call)
        return Contract(result.qfree, result.conserved)


@dataclass
class Call:
    node: GlobalCall
    quantum: tuple[int, ...]
    conserved: tuple[int, ...]
    target: ast.expr | None = None
    classical: dict[int, Wire] = field(default_factory=dict)
    projection: frozenset[int] | None = None
    residual: bool = False
    source: "Call | None" = None
    garbage: Wire | None = None


@dataclass
class Classical:
    stmt: ast.stmt


def factor_call(
    s: Statement, outputs: set[int], contracts: Contracts
) -> list[Statement] | None:
    """Factor f into its closed retained-output projection P and P†;f.

    The residual conserves the retained outputs. This makes the paper's conserved
    argument boundary explicit without exposing, cloning, or unrolling f's body.
    """
    call = s.payload
    if not isinstance(call, Call) or call.projection is not None:
        return None
    key = (call.node.def_id, call.node.type_args)
    if key not in contracts.variants.functions:
        return None
    targets = [i for i in call.quantum if i not in call.conserved]
    kept = frozenset(
        i for i, r in zip(targets, s.produced, strict=True) if r in outputs
    )
    selected = kept | frozenset(call.conserved)
    if (
        not kept
        or kept == set(targets)
        or not contracts.variants.can_project(call.node, selected)
    ):
        return None
    before = dict(zip(targets, s.consumed, strict=True))
    after = dict(zip(targets, s.produced, strict=True))
    controls = dict(zip(call.conserved, s.conserved, strict=True))
    prefix = replace(call, quantum=tuple(sorted(selected)), projection=selected)
    residual = replace(
        call,
        conserved=tuple(sorted(selected)),
        projection=selected,
        residual=True,
        source=prefix,
    )
    return [
        Statement(
            s.op,
            s.conserved,
            tuple(before[i] for i in targets if i in kept),
            tuple(after[i] for i in targets if i in kept),
            payload=prefix,
        ),
        Statement(
            s.op,
            tuple(controls[i] if i in controls else after[i] for i in sorted(selected)),
            tuple(before[i] for i in targets if i not in kept),
            tuple(after[i] for i in targets if i not in kept),
            payload=residual,
        ),
    ]


def compile_uncomputation(
    stmts: Sequence[ast.stmt],
    compiler: "StmtCompiler",
    dfg: DFContainer,
) -> DFContainer:
    """Compile a basic block containing a standalone forget request."""
    ctx = compiler.ctx
    contracts = Contracts(ctx)
    statements: list[Statement] = []
    current: dict[Variable.Id, int] = {}
    places: dict[Variable.Id, Variable] = {}
    inputs: dict[int, Wire] = {}
    counter = 0
    returns: list[ast.Return] = []

    def fresh() -> int:
        nonlocal counter
        r = counter
        counter += 1
        return r

    def place(node: ast.expr) -> Variable:
        if not isinstance(node, PlaceNode) or not isinstance(node.place, Variable):
            raise UncomputationError(
                "Uncomputation currently requires scalar qubit variables"
            )
        if not is_qubit_ty(node.place.ty):
            raise UncomputationError("Uncomputation currently requires scalar qubits")
        places[node.place.id] = node.place
        return node.place

    def register(node: ast.expr) -> int:
        p = place(node)
        if p.id not in current:
            if p not in dfg:
                raise UncomputationError(f"Unavailable quantum variable {p.name}")
            current[p.id] = fresh()
            inputs[current[p.id]] = dfg[p]
        return current[p.id]

    def classical(stmt: ast.stmt) -> None:
        # Looking only at the outer call's signature misses e.g. bool(measure(q)).
        # Never let quantum work bypass the SSA model through a classical result.
        for node in ast.walk(stmt):
            ty = get_type_opt(node)
            if (ty is not None and contain_qubit_ty(ty)) or is_forget(node):
                raise UncomputationError(
                    "Nested quantum expressions are unsupported in uncomputation blocks"
                )
        statements.append(Statement(Operation("classical"), payload=Classical(stmt)))

    for stmt in stmts:
        if not isinstance(
            stmt, ast.Assign | ast.AnnAssign | ast.Expr | ast.Return | ast.Pass
        ):
            raise UncomputationError(
                "Nested blocks and modifiers are unsupported in uncomputation blocks"
            )
        if isinstance(stmt, ast.Return):
            # Returns are emitted after synthesis so returned quantum wires are the
            # final live versions. Calls in return expressions need explicit locals.
            if stmt.value is not None:
                for n in ast.walk(stmt.value):
                    if isinstance(n, GlobalCall):
                        raise UncomputationError(
                            "Bind return expressions before forget/return"
                        )
                    if isinstance(n, PlaceNode) and contain_qubit_ty(n.place.ty):
                        register(n)
            returns.append(stmt)
            continue
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(stmt, ast.Expr):
            value = stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value = stmt.target, stmt.value

        if (
            isinstance(value, PlaceNode)
            and is_qubit_ty(value.place.ty)
            and target is not None
        ):
            r = register(value)
            current.pop(place(value).id)
            current[place(target).id] = r
            continue
        if not isinstance(value, GlobalCall):
            if any(
                isinstance(n, PlaceNode) and contain_qubit_ty(n.place.ty)
                for n in ast.walk(stmt)
            ) or any(is_forget(n) for n in ast.walk(stmt)):
                raise UncomputationError(
                    "Unsupported quantum statement in uncomputation block"
                )
            classical(stmt)
            continue

        call = value
        quantum = tuple(
            i for i, a in enumerate(call.args) if contain_qubit_ty(get_type(a))
        )
        regs = {i: register(call.args[i]) for i in quantum}
        if is_forget(call):
            if target is not None:
                raise UncomputationError("forget must be a standalone statement")
            statements.append(Statement(FORGET, consumed=(regs[0],)))
            current.pop(place(call.args[0]).id)
            continue
        if contain_qubit_ty(get_type(call)):
            from guppylang.std.quantum import qubit

            # Identify the actual allocator, not a user function named __new__.
            allocator = DEF_STORE.type_members[qubit.id]["__new__"]
            if call.def_id != allocator or quantum or target is None:
                raise UncomputationError(
                    "Quantum-returning calls require garbage-aware adjoints"
                )
            r = fresh()
            statements.append(Statement(ALLOC, produced=(r,)))
            current[place(target).id] = r
            continue
        if not quantum:
            classical(stmt)
            continue

        for i, arg in enumerate(call.args):
            if i not in quantum:
                for node in ast.walk(arg):
                    ty = get_type_opt(node)
                    if ty is not None and contain_qubit_ty(ty):
                        raise UncomputationError(
                            "Quantum work in classical gate arguments"
                        )

        fn = ctx.build_compiled_def(call.def_id, call.type_args)
        if not isinstance(fn, CompiledCallableDef):
            raise UncomputationError("Unsupported callable in uncomputation block")
        ty = fn.ty
        contract = contracts.get(call)
        controlled = UnitaryFlags.Control in ty.unitary_flags
        if not controlled and not contract.qfree:
            raise UncomputationError(f"{fn.name} is not controllable")
        if not isinstance(ty.output, NoneType) or any(
            InputFlags.Inout not in ty.inputs[i].flags for i in quantum
        ):
            raise UncomputationError(
                "Quantum calls must borrow scalar qubits and return None"
            )
        if any(
            InputFlags.Inout in inp.flags or InputFlags.Comptime in inp.flags
            for i, inp in enumerate(ty.inputs)
            if i not in quantum
        ):
            raise UncomputationError(
                "Classical call inputs must be immutable runtime values"
            )
        conserved = tuple(i for i in quantum if i in contract.conserved)
        consumed = tuple(i for i in quantum if i not in contract.conserved)
        produced = tuple(fresh() for _ in consumed)
        statements.append(
            Statement(
                Operation(
                    fn.name,
                    contract.qfree,
                    contract.qfree or UnitaryFlags.Dagger in ty.unitary_flags,
                ),
                tuple(regs[i] for i in conserved),
                tuple(regs[i] for i in consumed),
                produced,
                payload=Call(call, quantum, conserved, target),
            )
        )
        for i, r in zip(consumed, produced, strict=True):
            current[place(call.args[i]).id] = r

    while True:
        try:
            plan = synthesize(statements, tuple(inputs), tuple(current.values()))
            break
        except UncomputationSynthesisError as error:
            for producer in error.producers:
                factored = factor_call(producer, set(current.values()), contracts)
                if factored is not None:
                    index = next(
                        i
                        for i, s in enumerate(statements)
                        if s.payload is producer.payload
                    )
                    statements[index : index + 1] = factored
                    break
            else:
                raise
    wires = inputs.copy()
    for s in plan:
        with dfg.builder.set_ast_context(
            s.payload.node if isinstance(s.payload, Call) else None
        ):
            if s.op == ALLOC:
                if s.dagger:
                    _quantum(dfg, ctx, "QFree", [wires.pop(s.consumed[0])], 0, True)
                else:
                    wires[s.produced[0]] = _quantum(dfg, ctx, "QAlloc", [], 1, True)[0]
            elif s.op == DUP:
                control = s.conserved[0]
                target_wire = (
                    wires.pop(s.consumed[0])
                    if s.dagger
                    else _quantum(dfg, ctx, "QAlloc", [], 1, True)[0]
                )
                c, t = _quantum(dfg, ctx, "CX", [wires[control], target_wire], 2)
                wires[control] = c
                if s.dagger:
                    _quantum(dfg, ctx, "QFree", [t], 0, True)
                else:
                    wires[s.produced[0]] = t
            elif isinstance(s.payload, Classical):
                compiler.visit(s.payload.stmt)
            elif isinstance(s.payload, Call):
                _emit_call(s, s.payload, wires, compiler, dfg)
            else:
                raise UncomputationError(f"No lowering for {s.op.name}")
    for name, r in current.items():
        dfg[places[name]] = wires[r]
    for stmt in returns:
        compiler.visit_Return(stmt)
    return dfg


def _quantum(
    dfg: DFContainer,
    ctx: CompilerContext,
    name: str,
    args: list[Wire],
    outputs: int,
    effect: bool = False,
) -> list[Wire]:
    ty = ht.FunctionType([ht.Qubit] * len(args), [ht.Qubit] * outputs)
    op = quantum_op(name)(ty, (), ctx)
    return list(dfg.builder.add_op((op, [Effect.ANY] if effect else []), *args))


def _emit_call(
    s: Statement,
    call: Call,
    wires: dict[int, Wire],
    compiler: "StmtCompiler",
    dfg: DFContainer,
) -> None:
    ctx = compiler.ctx
    fn = ctx.build_compiled_def(call.node.def_id, call.node.type_args)
    assert isinstance(fn, CompiledCallableDef)
    targets = tuple(i for i in call.quantum if i not in call.conserved)
    registers = dict(zip(call.conserved, s.conserved, strict=True)) | dict(
        zip(targets, s.consumed, strict=True)
    )
    if not s.dagger:
        call.classical = (
            call.source.classical
            if call.source
            else {
                i: compiler.expr_compiler.compile(a, dfg)
                for i, a in enumerate(call.node.args)
                if not is_qubit_ty(get_type(a))
            }
        )
    from guppylang_internals.compiler.uncomputation_variants import Variants

    variants = Variants.get(ctx)
    key = (call.node.def_id, call.node.type_args)
    summary = variants.summary(call.node)
    if (
        key in variants.functions
        and summary.qfree
        and summary.garbage
        and s.partner is not None
    ):
        from guppylang_internals.compiler.uncomputation_garbage import (
            GarbageVariants,
            new_bin,
            finish_bin,
        )

        args_by_index = call.classical | {i: wires[r] for i, r in registers.items()}
        bin = call.garbage if s.dagger else new_bin(dfg.builder)
        assert bin is not None
        quantum_results, bin = GarbageVariants.get(variants).emit(
            call.node, args_by_index, bin, dfg, s.dagger
        )
        if s.dagger:
            finish_bin(dfg.builder, bin)
        else:
            call.garbage = bin
    elif key in variants.functions and summary.qfree and not summary.garbage:
        args_by_index = call.classical | {i: wires[r] for i, r in registers.items()}
        if call.residual:
            assert call.projection is not None
            stages = (
                [(None, True), (call.projection, False)]
                if s.dagger
                else [(call.projection, True), (None, False)]
            )
        else:
            stages = [(call.projection, s.dagger)]
        for selected, inverse in stages:
            indices = sorted(selected) if selected is not None else list(call.quantum)
            results = variants.emit(
                call.node, args_by_index, dfg, inverse=inverse, selected=selected
            )
            args_by_index.update(zip(indices, results, strict=True))
        quantum_results = [args_by_index[i] for i in call.quantum]
    else:
        quantum_results = _emit_default_dagger(
            s, call, registers, wires, compiler, dfg, fn
        )
    out_regs = dict(zip(call.conserved, s.conserved, strict=True)) | dict(
        zip(targets, s.produced, strict=True)
    )
    for r in s.consumed:
        wires.pop(r)
    for i, wire in zip(call.quantum, quantum_results, strict=True):
        wires[out_regs[i]] = wire


def _emit_default_dagger(
    s: Statement,
    call: Call,
    registers: dict[int, int],
    wires: dict[int, Wire],
    compiler: "StmtCompiler",
    dfg: DFContainer,
    fn: CompiledCallableDef,
) -> list[Wire]:
    ctx = compiler.ctx
    args = [
        wires[registers[i]] if i in registers else call.classical[i]
        for i in range(len(call.node.args))
    ]
    if not s.dagger:
        result = fn.compile_call(args, dfg, ctx, call.node)
        quantum_results = result.inout_returns
        if call.target is not None:
            # A None-valued assignment is allowed by Guppy.
            compiler._assign(call.target, dfg.builder.add_op(pure(ops.MakeTuple())))
    else:
        # Reorder to the ABI expected by the *existing* DaggerModifier: mutable
        # inputs first, immutable classical inputs second.
        order = [*call.quantum, *call.classical]
        in_types = [fn.ty.inputs[i].ty.to_hugr(ctx) for i in order]
        out_types: ht.TypeRow = [ht.Qubit] * len(call.quantum)
        wrapper = dfg.builder.define_function(
            f"__uncompute_{fn.name}_{len(ctx.module.hugr)}", in_types, out_types
        )
        add_unitary_metadata(
            wrapper._raw.hugr[wrapper].metadata, fn.ty.unitary_flags.value
        )
        wrapper_args = dict(zip(order, wrapper.inputs(), strict=True))
        result = fn.compile_call(
            [wrapper_args[i] for i in range(len(args))],
            DFContainer(wrapper, ctx),
            ctx,
            call.node,
        )
        wrapper.set_outputs(*result.inout_returns)
        ft = ht.FunctionType(in_types, out_types)
        loaded = dfg.builder.load_function(wrapper, ft)
        dagger = dfg.builder.add_op(
            pure(
                ops.ExtOp(
                    MODIFIER_EXTENSION.get_op("DaggerModifier"),
                    ht.FunctionType([ft], [ft]),
                    [
                        ht.ListArg([t.type_arg() for t in out_types]),
                        ht.ListArg([t.type_arg() for t in in_types[len(out_types) :]]),
                    ],
                )
            ),
            loaded,
        )
        quantum_results = list(
            dfg.builder.add_op(
                (ops.CallIndirect(), fn.call_effects), dagger, *(args[i] for i in order)
            )
        )
    return list(quantum_results)
