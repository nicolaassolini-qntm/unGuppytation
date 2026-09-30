"""Garbage-passing recursive variants: forget becomes push, its adjoint pop.

The bin is linear and shared by all recursive calls. Each call site retains the
bin until its paired inverse. No recursive expansion, depth bound, measurement,
or premature cleanup is involved.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from hugr import Wire
from hugr import tys as ht
from hugr.std.collections.list import List, ListVal

from guppylang_internals.compiler.builder import pure
from guppylang_internals.compiler.core import DFContainer
from guppylang_internals.compiler.stmt_compiler import StmtCompiler
from guppylang_internals.compiler.uncomputation import UncomputationError
from guppylang_internals.definition.value import CompiledCallableDef
from guppylang_internals.nodes import PlaceNode
from guppylang_internals.std._internal.compiler.list import list_pop, list_push
from guppylang_internals.std._internal.compiler.prelude import (
    build_expect_none,
    build_unwrap,
)
from guppylang_internals.std._internal.compiler.tket_exts import GUPPY_EXTENSION
from guppylang_internals.tys import Effect
from guppylang_internals.tys.qubit import is_qubit_ty

if TYPE_CHECKING:
    from guppylang_internals.checker.cfg_checker import CheckedBB
    from guppylang_internals.checker.core import Place
    from guppylang_internals.compiler.builder import DFBuilder, FunctionBuilder
    from guppylang_internals.compiler.uncomputation_variants import Variants
    from guppylang_internals.engine import MonoDefId
    from guppylang_internals.nodes import GlobalCall

BIN = List(ht.Qubit)


def new_bin(builder: DFBuilder) -> Wire:
    return builder.load(ListVal([], elem_ty=ht.Qubit))


def finish_bin(builder: DFBuilder, bin: Wire) -> None:
    # The generic affine-container drop is used only after proving at runtime that
    # this linear container is empty. It never discards a live qubit.
    empty, item = builder.add_op(list_pop(ht.Qubit), bin)
    build_expect_none(builder, item, "Uncomputation left nonempty garbage")
    drop = GUPPY_EXTENSION.get_op("drop").instantiate(
        [ht.TypeTypeArg(BIN)], ht.FunctionType([BIN], [])
    )
    builder.add_op(pure(drop), empty)


class GarbageVariants:
    def __init__(self, variants: Variants):
        self.variants = variants
        self.ctx = variants.ctx
        self.blocks: dict[
            tuple[MonoDefId, bool, CheckedBB[Place]], FunctionBuilder
        ] = {}

    @staticmethod
    def get(variants: Variants) -> GarbageVariants:
        if variants.garbage_variants is None:
            variants.garbage_variants = GarbageVariants(variants)
        return variants.garbage_variants

    def emit(
        self,
        call: GlobalCall,
        args: dict[int, Wire],
        bin: Wire,
        dfg: DFContainer,
        inverse: bool,
    ) -> tuple[list[Wire], Wire]:
        key = (call.def_id, call.type_args)
        fn = self.variants.functions[key]
        target = self._block(key, inverse, fn.cfg.entry_bb, dfg.builder)
        names = {inp.name: i for i, inp in enumerate(fn.ty.inputs)}
        cargs = [args[names[p.name]] for p in self.classical(fn.cfg.entry_bb)]
        qrow = self.quantum(fn.cfg.exit_bb if inverse else fn.cfg.entry_bb)
        qargs = [args[names[p.name]] for p in qrow]
        result = list(
            dfg.builder.call(target, *cargs, *qargs, bin, effects=[Effect.ANY])
        )
        output_row = self.quantum(fn.cfg.entry_bb if inverse else fn.cfg.exit_bb)
        by_name = {p.name: w for p, w in zip(output_row, result[:-1], strict=True)}
        return [by_name[p.name] for p in fn.ty.inputs if is_qubit_ty(p.ty)], result[-1]

    @staticmethod
    def classical(bb: CheckedBB[Place]) -> list[Place]:
        return [p for p in bb.sig.input_row if not is_qubit_ty(p.ty)]

    @staticmethod
    def quantum(bb: CheckedBB[Place]) -> list[Place]:
        return sorted(
            (p for p in bb.sig.input_row if is_qubit_ty(p.ty)), key=lambda p: p.name
        )

    def _block(
        self, key: MonoDefId, inverse: bool, bb: CheckedBB[Place], outer: DFBuilder
    ) -> FunctionBuilder:
        cache_key = key, inverse, bb
        if cache_key in self.blocks:
            return self.blocks[cache_key]
        fn = self.variants.functions[key]
        classical = self.classical(bb)
        entry_quantum = self.quantum(bb)
        final_quantum = self.quantum(fn.cfg.exit_bb)
        in_quantum = final_quantum if inverse else entry_quantum
        out_quantum = entry_quantum if inverse else final_quantum
        builder = outer.define_function(
            f"__uncompute_garbage_{'reverse' if inverse else 'forward'}_{fn.name}_bb{fn.cfg.bbs.index(bb)}_{len(self.blocks)}",
            [p.ty.to_hugr(self.ctx) for p in classical]
            + [ht.Qubit] * len(in_quantum)
            + [BIN],
            [ht.Qubit] * len(out_quantum) + [BIN],
        )
        self.blocks[cache_key] = builder
        dfg = DFContainer(builder, self.ctx)
        inputs = list(builder.inputs())
        for p, w in zip(classical, inputs[: len(classical)], strict=True):
            dfg[p] = w
        q = {
            p.name: w
            for p, w in zip(in_quantum, inputs[len(classical) : -1], strict=True)
        }
        bin = inputs[-1]
        compiler = StmtCompiler(self.ctx)
        compiler.dfg = dfg
        pending: list[tuple[ast.stmt, dict[int, Wire]]] = []
        for stmt in bb.statements:
            if self.variants.local_kind(stmt):
                if inverse:
                    pending.append((stmt, {}))
                else:
                    bin = self._statement(stmt, {}, q, bin, dfg, False)
            elif call := self.variants.quantum_call(stmt):
                cargs = {
                    i: compiler.expr_compiler.compile(a, dfg)
                    for i, a in enumerate(call.args)
                    if not isinstance(a, PlaceNode) or not is_qubit_ty(a.place.ty)
                }
                if inverse:
                    pending.append((stmt, cargs))
                else:
                    bin = self._statement(stmt, cargs, q, bin, dfg, False)
            elif not isinstance(stmt, ast.Return):
                compiler.visit(stmt)
        if bb.successors:
            succ_row = self.quantum(bb.successors[0])
            if any(
                [p.name for p in self.quantum(s)] != [p.name for p in succ_row]
                for s in bb.successors
            ):
                raise UncomputationError("Branches require matching quantum interfaces")
            # Reverse continuations carry the final function outputs down to exit,
            # and return predecessor interfaces while unwinding. Allocations may
            # therefore cross branch boundaries without a static garbage shape.
            next_inputs = final_quantum if inverse else succ_row
            next_outputs = succ_row if inverse else final_quantum
            if len(bb.successors) == 1:
                succ = bb.successors[0]
                target = self._block(key, inverse, succ, builder)
                result = list(
                    builder.call(
                        target,
                        *(dfg[p] for p in self.classical(succ)),
                        *(q[p.name] for p in next_inputs),
                        bin,
                        effects=[Effect.ANY],
                    )
                )
            else:
                assert bb.branch_pred is not None
                predicate = compiler.expr_compiler.compile(bb.branch_pred, dfg)
                cplaces = {
                    p.id: p for succ in bb.successors for p in self.classical(succ)
                }
                with builder.add_conditional(
                    predicate,
                    *(dfg[p] for p in cplaces.values()),
                    *(q[p.name] for p in next_inputs),
                    bin,
                ) as conditional:
                    for i, succ in enumerate(bb.successors):
                        case = conditional.add_case(i)
                        case_inputs = list(case.inputs())
                        cvalues = dict(
                            zip(cplaces, case_inputs[: len(cplaces)], strict=True)
                        )
                        target = self._block(key, inverse, succ, builder)
                        case_result = case.call(
                            target,
                            *(cvalues[p.id] for p in self.classical(succ)),
                            *case_inputs[len(cplaces) :],
                            effects=[Effect.ANY],
                        )
                        case.set_outputs(*case_result)
                result = list(conditional)
            q = {p.name: w for p, w in zip(next_outputs, result[:-1], strict=True)}
            bin = result[-1]
        for stmt, cargs in reversed(pending):
            bin = self._statement(stmt, cargs, q, bin, dfg, True)
        builder.set_outputs(*(q[p.name] for p in out_quantum), bin)
        return builder

    def _statement(
        self,
        stmt: ast.stmt,
        cargs: dict[int, Wire],
        q: dict[str, Wire],
        bin: Wire,
        dfg: DFContainer,
        inverse: bool,
    ) -> Wire:
        from guppylang_internals.compiler.uncomputation_compiler import _quantum

        if kind := self.variants.local_kind(stmt):
            action, place = kind
            if action == "forget":
                if inverse:
                    bin, item = dfg.builder.add_op(list_pop(ht.Qubit), bin)
                    q[place.name] = build_unwrap(
                        dfg.builder, item, "Uncomputation garbage underflow"
                    )
                else:
                    bin = dfg.builder.add_op(
                        list_push(ht.Qubit), bin, q.pop(place.name)
                    )
            elif inverse:
                _quantum(dfg, self.ctx, "QFree", [q.pop(place.name)], 0, True)
            else:
                q[place.name] = _quantum(dfg, self.ctx, "QAlloc", [], 1, True)[0]
            return bin
        call = self.variants.quantum_call(stmt)
        assert call is not None
        mapping = {
            i: a.place.name
            for i, a in enumerate(call.args)
            if isinstance(a, PlaceNode) and is_qubit_ty(a.place.ty)
        }
        args = cargs | {i: q[p] for i, p in mapping.items()}
        if self.variants.primitive(call):
            fn = self.ctx.build_compiled_def(call.def_id, call.type_args)
            assert isinstance(fn, CompiledCallableDef)
            outputs = fn.compile_call(
                [args[i] for i in range(len(call.args))], dfg, self.ctx, call
            ).inout_returns
        elif self.variants.summary(call).garbage:
            outputs, bin = self.emit(call, args, bin, dfg, inverse)
        else:
            outputs = self.variants.emit(call, args, dfg, inverse=inverse)
        for i, wire in zip(mapping, outputs, strict=True):
            q[mapping[i]] = wire
        return bin
