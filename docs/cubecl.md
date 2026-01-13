# CubeCL Technical Specification

## 1. Execution Model
CubeCL abstracts hardware-specific compute hierarchies into a unified **Cube-Unit** model.

### 1.1 Hierarchy
- **Working Unit (Thread):** The smallest unit of execution.
- **Cube (Workgroup/Block):** A 3D collection of units sharing **Shared Memory** and synchronization primitives.
- **Hyper-cube (Grid):** The total dispatch space of cubes.
- **Plane (Warp/Subgroup):** A hardware-level optimization group (usually 32 or 64 units) that executes in SIMD lockstep. Access via `PLANE_DIM`.

### 1.2 Topology Constants
| Constant | Description | Mapping (CUDA/WGSL) |
| :--- | :--- | :--- |
| `UNIT_POS` | Local unit index in cube (x, y, z) | `threadIdx` / `local_invocation_id` |
| `CUBE_POS` | Cube index in hyper-cube (x, y, z) | `blockIdx` / `workgroup_id` |
| `CUBE_DIM` | Dimensions of the cube | `blockDim` / `workgroup_size` |
| `ABSOLUTE_POS` | Global flattened index of the unit | Global ID |

## 2. Memory Taxonomy
- **`Array<T>`:** Global memory (Buffer). Passed as kernel arguments.
- **`SharedMemory<T, SIZE>`:** Cube-local memory. Explicitly synchronized via `sync_units()`.
- **`LocalMemory<T, SIZE>`:** Unit-local registers (Stack).
- **`Line<T>`:** Vectorized type. Abstracted SIMD container (e.g., `Line<f32>` with factor 4 maps to `float4`).

## 3. The `#[cube]` Macro Architecture
The procedural macro performs a **two-stage expansion**:

1.  **Parsing:** The `syn` crate parses the Rust function into an AST.
2.  **Expansion (`__expand`):** Generates a companion Rust function that, when called, populates a `CubeContext`. This context builds the **CubeCL IR**.
3.  **JIT Compilation:** The IR is passed to a backend-specific compiler (e.g., `WgslCompiler`, `CudaCompiler`) which generates the final shader source.

## 4. Comptime System
The `comptime` system allows meta-programming by executing Rust code during the IR generation phase.

- **`#[comptime]` Attribute:** Marks a function argument as a compile-time constant.
- **`comptime!` Macro:** Forces the enclosed block to be evaluated by the Rust compiler during expansion, allowing for:
    - Kernel specialization based on input shapes.
    - Unrolling loops.
    - Branch pruning (e.g., removing bounds checks if dimensions are multiples of cube size).

## 5. Vectorization Logic
CubeCL employs **Automatic Vectorization** through the `Line<T>` trait.
- **Semantics:** Operations on `Line<T>` are semantically identical to `T` but trigger hardware SIMD instructions.
- **Broadcasting:** Scalars automatically broadcast to lines.
- **Specialization:** Kernels can be specialized for different vectorization factors (1, 2, 4, 8, 16) at launch time without code changes.

## 6. Type System & Traits
- **`CubeType`:** The base trait for any type that can exist within the IR.
- **`Float`, `Int`, `UInt`, `Numeric`:** Trait bounds for generic kernels.
- **`Launch`:** Trait implemented for functions that can be dispatched to a GPU.

## 7. Synchronization & Atomics
- **`sync_units()`:** A memory barrier ensuring all units in a cube reach the same point and shared memory writes are visible.
- **Atomic Operations:** Supported on `UInt` and `Int` via `AtomicAdd`, `AtomicMax`, etc., targeting global or shared memory.

## 8. Launch Protocol
1.  **`ComputeClient<R: Runtime>`:** Handle to the GPU command queue and memory manager.
2.  **`KernelDefinition`:** The compiled IR and metadata.
3.  **`CubeCount` & `CubeDim`:** Runtime dispatch configuration.
4.  **`TensorArg`:** Binding between a `ComputeBuffer` and the kernel `Array<T>`.

## 9. Low-Level Control (CPA)
**Cube Pseudo-Assembly (CPA)** allows for direct IR manipulation when the high-level Rust frontend is insufficient. It is used primarily for implementing intrinsics and highly optimized library functions like `matmul`.

```rust/dev/null/cpa_example.rs#L1-10
// Pseudo-code representation of CPA usage
cpa!(context, [
    instruction1,
    instruction2,
]);
