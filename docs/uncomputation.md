# Explicit uncomputation with Guppy's existing dagger

## Status and scope

This branch adds an experimental `forget(q)` operation and a working implementation
of the backwards, dependency-driven uncomputation transformation for scalar qubits.
It compiles to ordinary HUGR operations and the existing `DaggerModifier`; the
result executes through Guppy's existing emulator pipeline.

**This is the default-dagger first implementation, not a complete port of the
paper's recursive modular compiler.** It implements the scalar, unconditional
synthesis kernel and integrates it with individual Guppy basic blocks. Calls to
supported reversible helpers normally remain function calls; a selective expansion
fallback handles the coarse-dependency case described below. It does not yet implement
garbage-producing function variants, garbage bins, recursive uncomputation
erasure, or the paper's separate adjoint synthesis pass. Unsupported cases fail
explicitly rather than silently falling back to measurement or arbitrary discard.

The reference is *Modular Synthesis of Efficient Quantum Uncomputation*,
[arXiv:2406.14227](https://arxiv.org/pdf/2406.14227). The implementation work also
used the local reference repository
`oopsla24-modular-uncomputation-synthesis`, particularly
`src/hqir/hqir/passes/materialize_forgets.py`, `make_reverse.py`,
`make_fwdhalf.py`, and `hqir/uncomp.py`.

The branch is `na/modular-uncomputation-default-dagger`. No existing dagger
implementation was changed, so there is no second branch containing a replacement
adjoint implementation. The remaining work is listed explicitly below.

## Public interface

For runnable examples with checked simulator outputs, open
[`examples/uncomputation.ipynb`](../examples/uncomputation.ipynb). It covers local
cleanup, a reversible helper, coherence, dependency lifetime extension, and an
expected compilation error.

Import the new operation from its own standard-library module:

```python
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, h, measure
from guppylang.std.uncomputation import forget

@guppy
def clean_copy(a: qubit, result: qubit) -> None:
    temporary = qubit()
    cx(a, temporary)
    cx(temporary, result)
    forget(temporary)

@guppy
def main() -> int:
    a = qubit()
    result = qubit()
    h(a)
    clean_copy(a, result)
    cx(a, result)
    h(a)
    av = measure(a)
    rv = measure(result)
    return int(bool(av)) + int(bool(rv))

package = main.compile_function()
```

The two final measurements should both be zero. The temporary initially becomes
entangled with `a`; cleanup must preserve that coherence for the final Hadamard
to undo the first one. Replacing `forget` with unconditional discard would not
provide this guarantee.

`forget` consumes its argument, with the signature `qubit @ owned -> None`.
The caller does not subsequently own that qubit. Existing Guppy linearity checks
still reject use after forgetting, double consumption, and missing consumption.
There is no automatic cleanup of an unused qubit: requests must be explicit.

No `pure=True` decorator or purity type is added. Ordinary Guppy functions need
no new decorator option to use `forget`. A supported helper whose inverse must
be synthesized needs the existing controllable and daggerable capabilities,
normally written `@guppy(unitary=True)`.

The operation is experimental in scope, but does not require enabling the global
experimental-features switch. Its dedicated module and explicit operation make
the opt-in local to each program.

## Meaning of errors

`UncomputationError` subclasses Python `RuntimeError`. These exceptions are raised
**while the Python compiler runs**, usually from `compile_function()` or emulator
compilation. They are not quantum-program panic instructions or late checks on
the device. Ordinary Guppy type and ownership errors retain their existing
diagnostics, which can occur before the synthesis pass is reached.

Examples of failures include:

| Condition | Reason |
| --- | --- |
| `q = qubit(); h(q); forget(q)` | The producer creates superposition and is not accepted as a basis-preserving producer. |
| Copy from input `a`, mutate `a`, then forget the copy | The old input value needed by the inverse is unavailable and cannot be reconstructed from a local producer. |
| A producer is controllable but not daggerable | This implementation has no permitted inverse for the producer. |
| Measurement in the synthesized block | Measurement is not controllable. |
| `bool(measure(q))` nested inside a classical expression | Quantum work cannot bypass the SSA dependency model through a classical result. |
| A local value was produced in an earlier basic block | Cross-block provenance is not represented by this adapter. |
| Forgetting the result of an allocating helper | Its inverse changes the quantum signature; the balanced default-dagger adapter cannot implement it. |
| Arrays, field accesses, nested modifier blocks | These require additional lowering before scalar SSA synthesis. |

The backend's `QFree` operation is used only after the synthesis plan has undone
the computation. There is no new device-side assertion that checks a freed qubit
is zero. Consequently, the conservative producer checks and correct dependency
tracking are essential; testing also checks zero-state deallocation independently.

## Files and responsibilities

| File | Change |
| --- | --- |
| `guppylang/src/guppylang/std/uncomputation.py` | Public typed `forget` declaration and API documentation. |
| `guppylang-internals/src/guppylang_internals/std/_internal/compiler/uncomputation.py` | Sentinel `ForgetCompiler`; a request that escapes the block pass raises an error. |
| `guppylang-internals/src/guppylang_internals/compiler/uncomputation.py` | Small scalar SSA model, invariants, and dependency-driven synthesis. |
| `guppylang-internals/src/guppylang_internals/compiler/uncomputation_compiler.py` | Checked-AST adapter, operation summaries, and HUGR emission through the existing dagger. |
| `guppylang-internals/src/guppylang_internals/compiler/stmt_compiler.py` | Dispatch a block containing the actual forget intrinsic to the new compiler. |
| `tests/test_uncomputation.py` | Independent basis interpreter, structural checks, invalid input checks, and deterministic generated programs. |
| `tests/integration/test_uncomputation.py` | HUGR validation, emulator execution, coherence checks, and unsupported-case regressions. |
| `docs/uncomputation.md` | This implementation report. |

The hook runs after existing type and linearity checking. The normal statement
compiler handles blocks without `forget` exactly as before. Detection uses the
registered custom compiler class, not a spelling comparison: importing `forget`
under another name works, and an unrelated function named `forget` does not
activate the transformation.

## The scalar intermediate representation

Each `Statement` contains:

- An `Operation` with a diagnostic name and internal `qfree`/`daggerable` contracts.
- `conserved`: quantum values used as computational-basis controls.
- `consumed`: quantum values whose current SSA versions end at this statement.
- `produced`: freshly defined quantum values.
- A `dagger` bit identifying the inverse orientation.
- An opaque payload used by the Guppy adapter to retain the original checked call.
- An optional partner pointer linking an original computation to its synthesized
  inverse. Partners are matched by statement identity, not textual equality.

Integer registers identify values. They do not identify mutable Python variables
or physical qubit locations. For example, `x(q)` consumes the old register for
`q` and produces a new register, even though the eventual circuit uses the same
physical qubit. A control passed to `cx(control, target)` keeps its logical
register. The HUGR emitter nevertheless updates its physical wire after the call.

The built-in SSA operations are allocation, basis duplication, and forgetting.
Duplication means allocation of a zero qubit followed by CX. It preserves
computational-basis information coherently; it does not clone arbitrary states.
Its inverse applies CX and then frees the zero target.

`validate` checks that:

1. Inputs and outputs do not alias.
2. Every use refers to an available value.
3. Conserved and consumed arguments do not alias within a statement.
4. Each produced register is fresh and defined exactly once.
5. Consumption removes a value from scope.
6. The final live set equals the stated outputs.

Validation runs on both the input and synthesized programs. These are explicit
runtime checks, so disabling Python assertions does not remove them.

## Synthesis in detail

The algorithm uses a backwards analysis followed by a forward substitution pass.
It works on copies of input statements; failing synthesis does not mutate the
original register graph.

### Backwards analysis

`alive` starts as the set of output registers. The producer map associates each
SSA result with the statement that created it. `undone` records producers already
selected for inversion. `extended` records result values whose lifetimes must be
extended, and `restored` records values returned by generated inverse calls.

When the traversal sees `forget(r)`, it finds the producer of `r`. A function or
block input has no local producer, so such a dependency is rejected. A producer
must have a known basis-preserving contract and an available dagger.

The producer's unavailable conserved inputs and all its consumed inputs are
recursively made available. This recursion follows the acyclic SSA producer
graph, not the source program's function-call graph. A producer is selected at
most once, even when several forget requests need it.

The inverse is inserted at the forget anchor, before the newly scheduled cleanup
of its dependencies. Its consumed arguments correspond to the forward call's
results; its produced arguments correspond to the forward call's consumed values.
Conserved arguments remain conserved. The original statement and its inverse are
paired explicitly.

If an original result is still live, a basis duplicate supplies the inverse and
the original result remains available to the rest of the function. If it has
already left scope, the value is marked for lifetime extension. Restored consumed
inputs are cleaned by inverse duplication against their retained originals.

### Forward lifetime resolution

The forward pass removes the declarative forget operations and resolves those
lifetime extensions. When an earlier ordinary statement consumes a value that
must remain available for later cleanup, a basis copy is inserted and the earlier
consumer receives the copy. A substitution map keeps the original value available
to its later inverse. If the earlier consumer was itself a forget, it is removed
and no such copy is needed.

This separation keeps analysis anchored to the original producer graph. Inserted
statements do not recursively become new producers to analyze. Finally, the
synthesized graph is validated again before quantum operations are emitted.

### Why an earlier forget may move

Consider a dependency chain:

```python
first = qubit()
cx(a, first)
second = qubit()
cx(first, second)
result = qubit()
cx(second, result)
forget(first)
forget(second)
return result
```

The last forget needs `first`. During the backwards pass, `first` is therefore
retained until `second` has been undone. The earlier request to forget `first`
disappears, and its eventual inverse is scheduled after its last cleanup use.
The end-to-end test exercises this with `a` in superposition and verifies that
the result remains coherent.

### Why live sibling outputs need copying

A single helper can update two target qubits. If only one target is forgotten,
the helper's inverse still requires both outputs. Simply reversing the helper
would also erase the output the caller wants to keep. The pass duplicates the
live sibling in the computational basis for the inverse, preserving the requested
output. The independent interpreter checks all zero-state frees, and an emulator
test checks coherence through such a helper.

### Complexity and optimization limits

The number of original producers selected for inversion is bounded by the
number of original statements. Inserted copies scale with quantum argument and
result occurrences. The pass therefore avoids recursively repeating an original
producer within a block merely because several cleanup requests depend on it.

This Python implementation uses lists and `list.index` for insertion anchors;
its compiler running time is not claimed to be linear. It also retains explicit
dup/undup operations instead of implementing the reference implementation's
additional cascade and equivalence-class optimizations. More temporary qubits
and CX operations may be generated than strictly necessary.

Most significantly, these bounds describe an individual synthesized block.
Without garbage erasure, they do not imply the paper's interprocedural bounds for
recursive programs or repeated calls to helpers that perform their own cleanup.

## Effects without a public purity system

The adapter uses `UnitaryFlags.Control` for the reversible quantum-effect boundary.
A non-controllable quantum call in a synthesized block is rejected. A call that
must be inverted must additionally have `UnitaryFlags.Dagger`.

Controllability does not establish that a value is forgettable: H is controllable
but can create superposition. The adapter therefore uses a small internal
conservative contract table, rather than introducing new source-level purity
annotations or accepting every controllable operation as pure.

The actual registered X, CX, and Toffoli definitions are recognised as
basis-preserving. CX conserves its first quantum argument; Toffoli conserves its
first two. Their targets acquire new SSA versions. A user function named `x` is
not automatically trusted.

For an ordinary straight-line helper returning `None`, the adapter inspects its
checked body. If every statement is an accepted basis-preserving call, it derives
a summary. An input is conserved only when every use conserves it. Function
calls normally remain calls in the resulting HUGR. Summaries are cached within a
block compilation. If the whole-call summary rejects an unavailable input
dependency, the compiler can selectively expand a supported helper as described
below.

Unknown, recursive, branching, allocating, or custom-modifier helpers receive a
conservative non-qfree summary. They cannot serve as producers to undo. An
otherwise supported quantum operation with quantum effects may still occur on
an unrelated live output; only a dependency that actually needs uncomputation
must satisfy the stronger producer contract.

The public `forget` declaration reports `Effect.ANY`. This accounts for the
allocations and deallocations that synthesis can introduce and keeps Guppy's
existing effect/order-edge analysis conservative. Forward-only classical
statements stay in their original order; they are not copied or reversed.

### Helpers that update a source before using it as a control

Consider this valid example:

```python
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
```

The initial adapter rejected it because the summary for `use_cx` consumes both
input versions. Reversing that whole call requires the original function input
`a`, which has no local producer. This summary loses information about the
intermediate value: `temporary` actually depends on the **updated** `a`, and that
value is still available. A CX controlled by the updated `a` can clear the
temporary without reversing the X on the surviving input.

The compiler now records the producer path on `UncomputationSynthesisError`, a
`RuntimeError` subclass, for unavailable inputs and non-basis-preserving producers.
Before emitting a failed synthesis plan, it looks for a
supported helper on this path, substitutes its checked body into that call site,
and retries. The expanded body exposes the separate X and CX dependencies. The
original checked function is copied, not mutated. Classical arguments are bound
to fresh internal variables once at the original call site before substitution;
unused arguments are still evaluated, and repeated uses do not repeat argument
evaluation.

This also handles `h(a)` immediately before `use_cx(a, temporary)`: a whole-call
inverse would incorrectly walk back to H and reject it. Exposing the helper shows
that the temporary only needs the current, still-live `a` as a control. H remains
on the surviving qubit and is not part of the temporary's cleanup. Directly
forgetting a qubit whose producer is H remains rejected.

This fallback only expands standalone calls already accepted as qfree and
daggerable. It does not expand arbitrary opaque, recursive, effectful, or
custom-modifier functions. Expansion is bounded at 64 call sites per block;
exceeding that bound raises a `RuntimeError`. Calls whose existing summaries are
sufficient remain calls, including the modular helper examples above. Exposing
helper bodies can increase code size; this is a conservative precision fallback,
not the paper's modular garbage-erasure construction.

The fallback still rejects genuine information loss, such as copying the sole
input into a temporary, clearing the input using that temporary, and then asking
to forget the temporary. It also does not reconstruct arbitrary unavailable
input versions by inverting live outputs: some physically valid programs remain
outside the supported precondition.

## Reusing the existing dagger

Each balanced call inverse is lowered through a small function wrapper:

1. Put all quantum arguments first and immutable classical arguments second.
2. Compile the original Guppy call inside the wrapper using its normal compiler.
3. Return the quantum inout results in their original quantum-argument order.
4. Attach the existing unitary metadata to the wrapper.
5. Load the wrapper and apply the existing HUGR `DaggerModifier` extension.
6. Invoke the resulting callable through `CallIndirect`.

The argument reordering matters when a helper's source signature interleaves
classical and quantum parameters. The wrapper reconstructs the source order for
the contained call. Quantum controls are physically inout values in HUGR even
when their basis values are conserved in the synthesis representation.

Classical arguments are evaluated once at the original call. Their wires are
saved in that call's payload and reused by its inverse. Reassigning a classical
Python variable later must not change the parameters of an earlier computation's
inverse.

Allocation and basis-copy inverses are emitted directly, outside any dagger
wrapper. This avoids changing dagger's existing balanced-signature restrictions.
Custom modifier dispatch is not expanded at this late compiler stage: introducing
a previously unregistered custom dagger implementation would require changes to
the earlier modifier-use/call-graph analysis. Such helpers are excluded from the
qfree summaries for this first implementation.

## Supported source subset

The following are supported inside a block containing `forget`:

- Explicit scalar allocations assigned to individual variables.
- Direct named calls borrowing scalar qubits and returning `None`.
- X, CX, Toffoli, and straight-line helper compositions of those operations as
  forgettable producers.
- Other controllable gates on values whose producers need not be undone.
- Immutable runtime classical call arguments, including classical parameters
  before quantum parameters.
- Classical statements containing no nested quantum work.
- Scalar qubit moves such as `other = temporary`.
- Returning live scalar qubit variables, and multiple explicit cleanup requests.
- Block-local cleanup inside a larger program whose surrounding functions have
  branches or perform measurement.

The adapter rejects unsupported quantum expressions and nested modifier blocks.
Direct return expressions containing calls must first be bound to locals. Arrays,
struct fields, higher-order quantum calls, and allocating callees are outside
the accepted scalar adapter. Cross-block cleanup dependencies are rejected rather
than guessed from mutable variable names.

## Validation performed

The algorithm tests use a separate computational-basis interpreter. Unlike the
emulator's general free operation, this interpreter asserts that every generated
allocation inverse frees zero and every undup matches its conserved value. It
compares the synthesized program with declarative forget semantics on all basis
inputs of each generated two-input test. Thirty deterministic seeds exercise
dependency chains, updates, early forgets, and retained results.

Targeted tests also cover a shared dependency's single inverse, preservation of a
live sibling output, partner symmetry, input immutability, malformed SSA,
unavailable inputs, unknown effects, and missing daggers.

Integration tests validate serialized HUGR and execute representative programs
with Selene's statevector simulator. They cover zero cleanup, copying through
temporary values, superposition/coherence, helper calls, delayed dependencies,
Toffoli, classical argument ordering, import aliases, qubit moves, and explicit
unsupported-case errors. The coherence cases expect deterministic zero after
interference; they are stronger checks than checking a classical truth table.

The existing modifier tests are also run to check that default dagger, control,
custom modifier call graphs, metadata, and modifier error diagnostics still work.

Results for this implementation:

| Check | Result |
| --- | --- |
| Algorithm tests | 39 passed |
| New integration tests | 26 passed |
| Example notebook | All cells executed, including the expected-error example |
| Existing modifier integration, call-graph, metadata, and error tests | 183 passed, 1 expected failure |
| Ruff over `guppylang`, `guppylang-internals`, and `tests` | Passed |
| Ruff formatting of all changed Python files | Passed |
| Mypy over both packages | Passed, 183 source files checked |
| Git whitespace check | Passed |

With a development environment installed, the relevant commands are:

```sh
uv run pytest -q tests/test_uncomputation.py tests/integration/test_uncomputation.py
uv run pytest -q tests/integration/test_modifier.py tests/integration/test_modifier_py312.py tests/integration/test_custom_modifier_callgraph.py tests/metadata/test_modifier_metadata.py tests/error/test_modifier_errors.py
uv run ruff check guppylang guppylang-internals tests
uv run mypy guppylang guppylang-internals
```

In this workspace the existing sibling Guppy virtual environment was reused with
`PYTHONPATH=guppylang/src:guppylang-internals/src`, so imports resolved to this fork.
The emulator requires a local results socket; its execution tests ran with the
necessary sandbox permission. No packages or files in the reference repository
were modified.

## Work still needed for the complete modular algorithm

1. **Garbage-aware function interfaces.** Allocating callees need signatures that
   distinguish conserved inputs, consumed inputs, produced outputs, and garbage
   bins. The current adapter only inverts balanced inout calls.
2. **Uncomputation erasure.** The stored computation/inverse pairs are a starting
   point, but no pass currently replaces paired cleanup with bin disposal or
   threads those bins through calls. Pair metadata alone provides no such effect.
3. **Separate adjoint synthesis.** Classical computation must remain forward while
   quantum consumption and production are reversed, including unbalanced
   signatures. This is where a second implementation branch should diverge from
   the default-dagger baseline.
4. **Recursive variant generation.** Forward, adjoint, and garbage-producing
   variants require mutually recursive declarations and a demand-driven worklist.
   The conservative summary cache currently rejects recursive producers.
5. **Control flow and composite quantum data.** Branch conditions, joins, loops,
   arrays, and structured values need explicit provenance and shape handling.
   A basic-block input is intentionally treated as having no local producer.
6. **Optimization and diagnostics.** Equivalence-based dup/undup elimination,
   cascade cleanup, reusable inverse wrappers, and source-level dependency paths
   would reduce overhead and improve errors.

These are architectural extensions, not enabled by an option in this branch.
The current tests establish the implemented subset; they do not claim to validate
the paper's full recursive or asymptotic interprocedural guarantees.
