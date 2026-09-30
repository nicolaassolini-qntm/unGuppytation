"""Modular summaries and recursive adjoints for checked Guppy control flow.

A reverse block first replays classical work, calls the reverse successor, then
undoes its own quantum work. Declarations are cached before bodies are emitted:
recursive calls and CFG back edges remain calls, irrespective of runtime depth.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from dataclasses import dataclass

from hugr import Wire
from hugr import tys as ht

from guppylang_internals.ast_util import get_type, get_type_opt
from guppylang_internals.checker.core import Place, Variable
from guppylang_internals.compiler.core import CompilerContext, DFContainer
from guppylang_internals.compiler.stmt_compiler import StmtCompiler
from guppylang_internals.compiler.uncomputation import UncomputationError
from guppylang_internals.definition.function import CheckedFunctionDef
from guppylang_internals.definition.value import CompiledCallableDef
from guppylang_internals.engine import DEF_STORE, ENGINE, MonoDefId
from guppylang_internals.nodes import GlobalCall, PlaceNode
from guppylang_internals.tys.qubit import contain_qubit_ty, is_qubit_ty
from guppylang_internals.tys.ty import InputFlags, NoneType


if TYPE_CHECKING:
    from guppylang_internals.checker.cfg_checker import CheckedBB
    from guppylang_internals.compiler.builder import DFBuilder, FunctionBuilder
    from guppylang_internals.compiler.uncomputation_garbage import GarbageVariants


@dataclass(frozen=True)
class Summary:
    qfree: bool
    # Parameter i can depend on these parameters. Includes i itself.
    dependencies: tuple[frozenset[int], ...] = ()
    conserved: frozenset[int] = frozenset()
    garbage: bool = False


class Variants:
    def __init__(self, ctx: CompilerContext):
        self.ctx = ctx
        self.functions: dict[MonoDefId, CheckedFunctionDef] = {}
        self.summaries: dict[MonoDefId, Summary] = {}
        self.blocks: dict[
            tuple[MonoDefId, frozenset[int], bool, CheckedBB[Place]], FunctionBuilder
        ] = {}
        self.invalid: set[MonoDefId] = set()
        self.local_garbage: set[MonoDefId] = set()
        self.garbage_variants: GarbageVariants | None = None

    @staticmethod
    def get(ctx: CompilerContext) -> Variants:
        if ctx.uncomputation_variants is None:
            ctx.uncomputation_variants = Variants(ctx)
        return ctx.uncomputation_variants

    def primitive(self, call: GlobalCall) -> Summary | None:
        from guppylang.std.quantum import cx, toffoli, x

        if call.def_id == x.id:
            return Summary(True, (frozenset({0}),))
        if call.def_id == cx.id:
            return Summary(True, (frozenset({0}), frozenset({0, 1})), frozenset({0}))
        if call.def_id == toffoli.id:
            return Summary(
                True,
                (frozenset({0}), frozenset({1}), frozenset({0, 1, 2})),
                frozenset({0, 1}),
            )
        return None

    def summary(self, call: GlobalCall) -> Summary:
        if primitive := self.primitive(call):
            return primitive
        key = (call.def_id, call.type_args)
        if key not in self.summaries:
            self._discover(key)
            # Greatest fixed point for qfree/conservation, least fixed point for
            # dependencies. A recursive SCC is accepted only if *every* leaf is
            # valid; optimistic cycle breaking never escapes this iteration.
            changed = True
            while changed:
                changed = False
                for k, fn in self.functions.items():
                    result = self._analyse(k, fn)
                    if result != self.summaries[k]:
                        self.summaries[k] = result
                        changed = True
        return self.summaries[key]

    def _discover(self, key: MonoDefId) -> None:
        if key in self.summaries:
            return
        fn = ENGINE.get_checked(*key)
        self.summaries[key] = Summary(False)
        if not isinstance(fn, CheckedFunctionDef) or DEF_STORE.custom_modified_defs.get(
            key[0]
        ):
            return
        self.functions[key] = fn
        quantum = frozenset(i for i, p in enumerate(fn.ty.inputs) if is_qubit_ty(p.ty))
        self.summaries[key] = Summary(
            True, tuple(frozenset({i}) for i in range(len(fn.ty.inputs))), quantum
        )
        try:
            self._check_body(key, fn, quantum)
        except UncomputationError:
            self.invalid.add(key)

    def _check_body(
        self, key: MonoDefId, fn: CheckedFunctionDef, quantum: frozenset[int]
    ) -> None:
        if not isinstance(fn.ty.output, NoneType):
            raise UncomputationError("Recursive adjoints require a None return")
        for i, inp in enumerate(fn.ty.inputs):
            if i in quantum:
                if InputFlags.Inout not in inp.flags:
                    raise UncomputationError(
                        "Recursive adjoints require borrowed qubits"
                    )
            elif not inp.ty.copyable or inp.flags:
                raise UncomputationError(
                    "Recursive adjoints require immutable classical inputs"
                )
        for bb in fn.cfg.bbs:
            for p in bb.sig.input_row:
                if not isinstance(p, Variable):
                    raise UncomputationError(
                        "Recursive adjoints require scalar variables"
                    )
            for stmt in bb.statements:
                kind = self.local_kind(stmt)
                if kind:
                    self.local_garbage.add(key)
                    continue
                call = self.quantum_call(stmt)
                if call:
                    if not isinstance(stmt, ast.Expr):
                        raise UncomputationError(
                            "Quantum calls must be standalone statements"
                        )
                    for arg in call.args:
                        if contain_qubit_ty(get_type(arg)):
                            if not isinstance(arg, PlaceNode) or not is_qubit_ty(
                                arg.place.ty
                            ):
                                raise UncomputationError(
                                    "Recursive adjoints require scalar qubit arguments"
                                )
                        else:
                            self.check_classical(arg)
                    if not self.primitive(call):
                        self._discover((call.def_id, call.type_args))
                else:
                    self.check_classical(stmt)
            if bb.branch_pred:
                self.check_classical(bb.branch_pred)

    @staticmethod
    def local_kind(stmt: ast.stmt) -> tuple[str, Place] | None:
        from guppylang.std.quantum import qubit
        from guppylang_internals.compiler.uncomputation_compiler import is_forget

        value = (
            stmt.value
            if isinstance(stmt, ast.Expr | ast.Assign | ast.AnnAssign)
            else None
        )
        if not isinstance(value, GlobalCall):
            return None
        if (
            is_forget(value)
            and isinstance(stmt, ast.Expr)
            and isinstance(value.args[0], PlaceNode)
        ):
            return "forget", value.args[0].place
        if value.def_id == DEF_STORE.type_members[qubit.id]["__new__"]:
            target = (
                stmt.targets[0]
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                else stmt.target
                if isinstance(stmt, ast.AnnAssign)
                else None
            )
            if isinstance(target, PlaceNode) and is_qubit_ty(target.place.ty):
                return "alloc", target.place
        return None

    def check_classical(self, node: ast.AST) -> None:
        for sub in ast.walk(node):
            ty = get_type_opt(sub)
            if ty is not None and contain_qubit_ty(ty):
                raise UncomputationError("Quantum values in classical replay")
            if isinstance(sub, GlobalCall):
                fn = self.ctx.build_compiled_def(sub.def_id, sub.type_args)
                if not isinstance(fn, CompiledCallableDef) or fn.call_effects:
                    raise UncomputationError(
                        "Classical replay must be free of side effects"
                    )
                if any(InputFlags.Inout in p.flags for p in fn.ty.inputs):
                    raise UncomputationError(
                        "Classical replay cannot mutate borrowed inputs"
                    )
        if isinstance(node, ast.stmt) and not isinstance(
            node, ast.Assign | ast.AnnAssign | ast.Expr | ast.Return | ast.Pass
        ):
            raise UncomputationError("Unsupported statement in recursive adjoint")

    @staticmethod
    def quantum_call(stmt: ast.stmt) -> GlobalCall | None:
        value = (
            stmt.value
            if isinstance(stmt, ast.Expr | ast.Assign | ast.AnnAssign)
            else None
        )
        if isinstance(value, GlobalCall) and any(
            contain_qubit_ty(get_type(a)) for a in value.args
        ):
            return value
        return None

    def _analyse(self, key: MonoDefId, fn: CheckedFunctionDef) -> Summary:
        old = self.summaries[key]
        if key in self.invalid or not old.qfree:
            return Summary(False)
        names = {p.name: i for i, p in enumerate(fn.ty.inputs)}
        deps = {p.name: set(old.dependencies[i]) for i, p in enumerate(fn.ty.inputs)}
        conserved = set(old.conserved)
        garbage = key in self.local_garbage
        changed = True
        while changed:
            changed = False
            for bb in fn.cfg.bbs:
                for stmt in bb.statements:
                    if kind := self.local_kind(stmt):
                        deps.setdefault(kind[1].name, set())
                        continue
                    if not (call := self.quantum_call(stmt)):
                        continue
                    summary = (
                        self.primitive(call)
                        or self.summaries[call.def_id, call.type_args]
                    )
                    if not summary.qfree:
                        return Summary(False)
                    garbage |= summary.garbage
                    mapping = {
                        i: a.place.name
                        for i, a in enumerate(call.args)
                        if isinstance(a, PlaceNode) and is_qubit_ty(a.place.ty)
                    }
                    for i, p in mapping.items():
                        if i not in summary.conserved and p in names:
                            conserved.discard(names[p])
                        before = set(deps.setdefault(p, set()))
                        for dep in summary.dependencies[i]:
                            if dep in mapping:
                                deps[p].update(deps.setdefault(mapping[dep], set()))
                        changed |= before != deps[p]
        return Summary(
            True,
            tuple(frozenset(deps[p.name]) for p in fn.ty.inputs),
            frozenset(conserved),
            garbage,
        )

    def can_project(self, call: GlobalCall, selected: frozenset[int]) -> bool:
        s = self.summary(call)
        return (
            s.qfree
            and not s.garbage
            and all(s.dependencies[i] <= selected for i in selected)
        )

    def emit(
        self,
        call: GlobalCall,
        args: dict[int, Wire],
        dfg: DFContainer,
        *,
        inverse: bool,
        selected: frozenset[int] | None = None,
    ) -> list[Wire]:
        key = (call.def_id, call.type_args)
        fn = self.functions[key]
        quantum = frozenset(
            i for i, inp in enumerate(fn.ty.inputs) if is_qubit_ty(inp.ty)
        )
        selected = quantum if selected is None else selected
        if not self.can_project(call, selected):
            raise UncomputationError("Retained outputs depend on forgotten inputs")
        entry = self._block(key, selected, inverse, fn.cfg.entry_bb, dfg.builder)
        classical = [p for p in fn.cfg.entry_bb.sig.input_row if not is_qubit_ty(p.ty)]
        names = {inp.name: i for i, inp in enumerate(fn.ty.inputs)}
        wires = [args[names[p.name]] for p in classical] + [
            args[i] for i in sorted(selected)
        ]
        return list(dfg.builder.call(entry, *wires, effects=[]))

    def _block(
        self,
        key: MonoDefId,
        selected: frozenset[int],
        inverse: bool,
        bb: CheckedBB[Place],
        outer: DFBuilder,
    ) -> FunctionBuilder:
        cache_key = (key, selected, inverse, bb)
        if cache_key in self.blocks:
            return self.blocks[cache_key]
        fn = self.functions[key]
        classical = [p for p in bb.sig.input_row if not is_qubit_ty(p.ty)]
        q_indices = sorted(selected)
        builder = outer.define_function(
            f"__uncompute_{'reverse' if inverse else 'forward'}_{fn.name}_"
            f"{'_'.join(map(str, q_indices))}_bb{fn.cfg.bbs.index(bb)}_{len(self.blocks)}",
            [p.ty.to_hugr(self.ctx) for p in classical] + [ht.Qubit] * len(selected),
            [ht.Qubit] * len(selected),
        )
        self.blocks[cache_key] = builder  # before visiting recursive edges
        dfg = DFContainer(builder, self.ctx)
        wires = list(builder.inputs())
        for p, w in zip(classical, wires[: len(classical)], strict=True):
            dfg[p] = w
        q = dict(zip(q_indices, wires[len(classical) :], strict=True))
        compiler = StmtCompiler(self.ctx)
        compiler.dfg = dfg
        names = {p.name: i for i, p in enumerate(fn.ty.inputs)}
        pending: list[tuple[GlobalCall, dict[int, Wire], dict[int, int]]] = []
        for stmt in bb.statements:
            call = self.quantum_call(stmt)
            if call is None:
                if not isinstance(stmt, ast.Return):
                    compiler.visit(stmt)
                continue
            mapping = {
                i: names[a.place.name]
                for i, a in enumerate(call.args)
                if isinstance(a, PlaceNode) and is_qubit_ty(a.place.ty)
            }
            child_selected = frozenset(i for i, p in mapping.items() if p in selected)
            if not child_selected:
                continue
            classical_args = {
                i: compiler.expr_compiler.compile(a, dfg)
                for i, a in enumerate(call.args)
                if i not in mapping
            }
            if inverse:
                pending.append((call, classical_args, mapping))
            else:
                self._gate(call, classical_args, mapping, q, dfg, False)
        if bb.successors:
            # Classical work is in forward order. In the inverse, successor calls
            # return before the saved gates are emitted in reverse order.
            if len(bb.successors) == 1:
                succ = bb.successors[0]
                target = self._block(key, selected, inverse, succ, builder)
                cargs = [dfg[p] for p in succ.sig.input_row if not is_qubit_ty(p.ty)]
                q = dict(
                    zip(
                        q_indices,
                        builder.call(
                            target, *cargs, *(q[i] for i in q_indices), effects=[]
                        ),
                        strict=True,
                    )
                )
            else:
                assert bb.branch_pred is not None
                predicate = compiler.expr_compiler.compile(bb.branch_pred, dfg)
                cplaces = {
                    p.id: p
                    for succ in bb.successors
                    for p in succ.sig.input_row
                    if not is_qubit_ty(p.ty)
                }
                with builder.add_conditional(
                    predicate,
                    *(dfg[p] for p in cplaces.values()),
                    *(q[i] for i in q_indices),
                ) as conditional:
                    for i, succ in enumerate(bb.successors):
                        case = conditional.add_case(i)
                        case_wires = list(case.inputs())
                        cvalues = dict(
                            zip(cplaces, case_wires[: len(cplaces)], strict=True)
                        )
                        target = self._block(key, selected, inverse, succ, builder)
                        cargs = [
                            cvalues[p.id]
                            for p in succ.sig.input_row
                            if not is_qubit_ty(p.ty)
                        ]
                        result = case.call(
                            target, *cargs, *case_wires[len(cplaces) :], effects=[]
                        )
                        case.set_outputs(*result)
                q = dict(zip(q_indices, conditional, strict=True))
        for call, cargs, mapping in reversed(pending):
            self._gate(call, cargs, mapping, q, dfg, True)
        builder.set_outputs(*(q[i] for i in q_indices))
        return builder

    def _gate(
        self,
        call: GlobalCall,
        cargs: dict[int, Wire],
        mapping: dict[int, int],
        q: dict[int, Wire],
        dfg: DFContainer,
        inverse: bool,
    ) -> None:
        selected = frozenset(i for i, p in mapping.items() if p in q)
        args = cargs | {i: q[p] for i, p in mapping.items() if i in selected}
        if primitive := self.primitive(call):
            if selected <= primitive.conserved:
                return
            assert set(mapping) <= selected, (
                "Projection must contain all gate dependencies"
            )
            fn = self.ctx.build_compiled_def(call.def_id, call.type_args)
            assert isinstance(fn, CompiledCallableDef)
            # X, CX and Toffoli are all self-adjoint.
            result = fn.compile_call(
                [args[i] for i in range(len(call.args))], dfg, self.ctx, call
            )
            outputs = result.inout_returns
        else:
            outputs = self.emit(call, args, dfg, inverse=inverse, selected=selected)
        for i, w in zip(sorted(selected), outputs, strict=True):
            q[mapping[i]] = w
