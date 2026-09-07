# Part IV - CUDA and Accelerator Programming

CUDA optimization is the discipline of translating an algorithm into a schedule over threads, instructions, memory levels, and asynchronous work. The correct starting point is never a favorite tile size or instruction. It is a resource model: what must move, what must be computed, which dependencies are unavoidable, and which hardware resource becomes limiting first.

This part progresses from that model to production kernels. It covers execution, memory, host orchestration, GEMM, reductions, scans, histograms, softmax, normalization, attention, serving-specific kernels, profiling, and correctness. Each chapter closes with design exercises and worked solutions that connect the primitive to production workloads.

:::callout decision|The CUDA optimization loop
Establish a correct reference. Measure the real workload. Build a bytes-and-FLOPs model. Identify one limiting resource. Change the schedule. Recheck correctness. Reprofile. Stop when the remaining gap is below the value of additional complexity.
:::

## GPU Execution, Memory, and Resource Accounting

LEAD: Threads execute in warps, warps are scheduled on streaming multiprocessors, and data moves through a hierarchy whose capacity grows as latency and sharing scope increase. Performance follows from how well the schedule uses those resources.

:::diagram memory_hierarchy|Registers and shared memory provide explicit locality; caches and HBM provide progressively larger scope. Each level has a capacity, bandwidth, allocation unit, and synchronization cost.

### The execution hierarchy

A kernel launch creates a grid. The grid contains thread blocks. A block is assigned to one streaming multiprocessor, or SM, and normally remains there until it finishes. Threads in a block can cooperate through shared memory and block-scoped barriers. Threads from unrelated blocks cannot assume concurrent residency or execution order.

Threads are issued in warps, commonly 32 lanes. A warp scheduler selects an eligible warp whose next instruction has ready operands. The GPU hides latency by switching among ready warps and by exploiting independent instructions within each warp.

Newer architectures add optional execution levels such as thread-block clusters and distributed shared-memory access. These features can coordinate nearby blocks, but they do not invalidate the basic reasoning method: define the participating group, storage scope, synchronization contract, and resource cost.

### SIMT, divergence, and predication

CUDA uses a single-instruction, multiple-thread execution model. Lanes in a warp may have different registers and addresses while issuing the same instruction. If lanes take different control paths, the warp executes the necessary paths under different masks.

Divergence cost depends on three facts:

- how many lanes are active on each path;
- how much work each path contains;
- whether the compiler converts a short branch into predicated instructions.

Branch divergence is not the same as memory divergence. A uniform branch can still issue scattered loads. A divergent branch can be inexpensive if only a few predicated instructions differ.

```cuda
// The boundary condition diverges only in the final block.
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
  y[i] = alpha * x[i] + y[i];
}
```

Do not remove a cheap bounds check by launching a separate cleanup kernel unless measurement shows the tail dominates. Extra launches and code variants can cost more than limited edge divergence.

### Warp scheduling and latency hiding

A warp becomes ineligible when it waits for data, a dependency, a barrier, or an execution resource. Latency hiding requires other eligible warps or independent instructions. This produces two forms of parallelism:

- **thread-level parallelism:** many resident warps can run while another waits;
- **instruction-level parallelism:** one thread or warp has several independent operations in flight.

A memory-bound gather may need many resident warps because each warp has dependent long-latency loads. A tensor-core mainloop can run well at lower occupancy if it has an effective asynchronous pipeline and enough independent matrix operations.

### Memory spaces and their contracts

**Registers** are thread-private and compiler allocated. Register pressure can reduce resident blocks; excess live values spill to local memory, which is physically backed by device memory and cached.

**Shared memory** is block-scoped, explicitly indexed, and suited to tiles, cross-thread exchange, reductions, and staging. Its capacity is reserved per block and therefore affects residency.

**L1 and L2 caches** provide hardware-managed reuse. L2 is shared across SMs and often explains why a second benchmark pass appears faster. Cache state must be considered when comparing cold and warm behavior.

**Global memory or HBM** is device-wide. It offers high bandwidth but much higher latency than on-chip storage. Coalescing and reuse determine useful bandwidth.

**Constant memory** is read-only from device code and can efficiently broadcast when a warp reads the same address. Divergent constant addresses serialize through the cache path.

**Pinned host memory** supports higher-bandwidth and asynchronous host-device transfers, but it is a scarce host resource. **Unified memory** can simplify address management, but page migration and fault behavior must be understood for performance-critical paths.

### Coalescing with a transaction example

For many modern CUDA devices, global accesses by a warp are serviced by the 32-byte sectors needed to cover the requested addresses. Consider 32 lanes loading adjacent `float` values from a 32-byte-aligned base:

`address(lane) = base + 4 * lane`

The warp requests 128 contiguous bytes, so four 32-byte sectors contain all values. Shift the base by one float and the same logical 128 bytes can span five sectors. Access one float every 32 floats and the warp may request one sector per lane while using only 4 bytes from each.

Useful bandwidth is:

`requested_bytes / transferred_bytes * measured_bus_bandwidth`

This ratio explains why a kernel can report high device-memory throughput while delivering little useful data.

:::callout insight|Explain coalescing with addresses
Say which address each lane requests, which aligned sectors cover those addresses, and how many bytes are useful. "Contiguous is good" is an observation; the transaction count is the explanation.
:::

### Alignment and vectorized access

Vector loads such as `float4` reduce instruction count when pointers are aligned and each thread owns adjacent data. They do not automatically fix a bad warp-level access pattern. A warp of threads performing aligned vector loads should still cover compact contiguous sectors.

Use scalar tail handling or a predicated final vector. Never reinterpret an unaligned pointer as a wider vector type solely to obtain fewer instructions. Validate compiler output because source-level vector syntax may be split when alignment is not provable.

### Shared-memory banks

Shared memory is divided into banks. A common mapping for four-byte words is conceptually:

`bank = word_address mod bank_count`

If lanes access different addresses in the same bank, the access may be serialized into several transactions. If lanes read the same address, hardware can broadcast.

A transpose tile often adds one padding column:

```cuda
__shared__ float tile[32][33];

tile[threadIdx.y][threadIdx.x] = input[row * ld + col];
__syncthreads();

// Column access now advances by 33 words rather than 32.
float value = tile[threadIdx.x][threadIdx.y];
```

Without padding, a column stride equal to the bank count maps many lanes to one bank. Padding rotates successive rows through different banks.

### Quantitative occupancy

Theoretical occupancy is active warps divided by the architectural warp limit per SM. Resident blocks are bounded by the minimum of several ceilings:

`blocks_threads = floor(max_threads_per_sm / threads_per_block)`

`blocks_registers = floor(registers_per_sm / registers_per_block)`

`blocks_shared = floor(shared_bytes_per_sm / shared_bytes_per_block)`

`resident_blocks = min(blocks_threads, blocks_registers, blocks_shared, max_blocks_per_sm)`

Register allocation occurs in architecture-specific granularity, so a small increase in registers per thread can cross a step and remove an entire resident block. Dynamic shared memory has a similar cliff.

#### Worked example

Suppose an illustrative SM supports 2,048 threads, 64 resident warps, 65,536 registers, and 100 KiB of shared memory. A block has 256 threads, uses 64 registers per thread, and reserves 32 KiB shared memory.

- thread ceiling: `floor(2048 / 256) = 8` blocks;
- register ceiling: `floor(65536 / (256 * 64)) = 4` blocks;
- shared-memory ceiling: `floor(100 KiB / 32 KiB) = 3` blocks.

The shared-memory limit allows 3 blocks, or 24 warps, which is 37.5 percent theoretical occupancy. That may be enough if the kernel pipelines memory well. Reducing shared memory to admit a fourth block helps only if the added warps hide a real stall and the redesign does not add traffic.

### Synchronization and memory ordering

`__syncthreads()` is a block barrier and establishes the shared-memory visibility needed by participating threads. It must be reached by all non-exited threads in the block along a convergent control path.

Warp shuffles exchange register values without shared memory. They require an accurate active mask. Passing a full mask when the final warp has inactive lanes can read undefined values or violate the primitive's participation contract.

Atomics make an update indivisible at a declared scope, but atomicity is not a general barrier. Memory fences order memory visibility but do not force other threads to arrive. Correct concurrent code states both the ordering requirement and the rendezvous requirement.

### Resource-led optimization table

| Symptom | Likely resource | Evidence | Candidate response |
| --- | --- | --- | --- |
| Many long-scoreboard stalls | Dependent memory latency | Scattered loads, low eligible warps | Coalesce, prefetch, increase independent work |
| High sectors per requested byte | Memory transactions | Poor global-load efficiency | Change layout or lane-to-data mapping |
| Low residency with spills | Registers | Local loads/stores, allocation report | Shorten live ranges, reduce tile, split kernel |
| Barrier-heavy timeline | Synchronization | Warp stalls at barriers, imbalance | Repartition work, warp-level exchange |
| High issue utilization, low tensor utilization | Instruction mix | Scalar work dominates | Reduce address/control overhead, use MMA path |

### Design Exercises

1. Explain coalescing using lane addresses and transactions.
2. When is low occupancy acceptable?
3. Why does padding a shared-memory transpose tile help?
4. What is the difference between a barrier, an atomic, and a memory fence?
5. A kernel reaches 80 percent of HBM bandwidth. What optimizations remain plausible?

### Worked Solutions

#### 1. Coalescing

Assume each of 32 lanes loads one adjacent four-byte word from an aligned base. The addresses cover 128 contiguous bytes, which can be serviced by four aligned 32-byte sectors on many modern devices. A four-byte base offset can make the range cross a fifth sector. A large stride can require nearly one sector per lane. The goal is to minimize transferred sectors per useful byte, not merely to make each thread's local access look sequential.

#### 2. Low occupancy

Low occupancy is acceptable when existing warps already hide the limiting latency or when instruction-level parallelism and asynchronous pipelines keep execution units busy. Large register tiles, tensor-core fragments, or multi-stage shared-memory pipelines often trade occupancy for reuse. It is unacceptable when the profile shows few eligible warps and exposed memory or dependency stalls that another resident block could cover.

#### 3. Padding a transpose tile

A square tile with row length equal to the shared-memory bank count makes column access advance by a whole bank cycle, so many lanes select the same bank. Adding one element changes the stride to a value relatively prime to the bank count, spreading lanes across banks. The cost is a small shared-memory increase; the benefit is fewer serialized bank transactions.

#### 4. Barrier, atomic, and fence

A barrier makes a group wait until all participants arrive and usually supplies memory visibility for a defined scope. An atomic serializes one read-modify-write operation on a location but does not make unrelated threads rendezvous. A fence orders visibility of memory operations at a scope but does not wait for consumers. Many producer-consumer algorithms require both an ordering mechanism and a separately communicated readiness condition.

#### 5. At 80 percent of HBM bandwidth

First determine whether 80 percent is bus traffic or useful requested bytes. If useful bandwidth is genuinely near the sustainable ceiling, large gains require fewer bytes: fuse passes, improve cache or shared-memory reuse, compress data, change layout, or change the algorithm. Smaller gains may come from reducing uncoalesced sectors, fixing partition camping, overlapping tails, or removing launches. Optimizing arithmetic alone is unlikely to matter unless it overlaps the memory path poorly.

## Host-Device Orchestration, Streams, and CUDA Graphs

LEAD: A fast kernel can sit inside a slow application. Transfers, allocations, launch submission, synchronization, and shape management determine whether kernel speed reaches the user.

### The asynchronous execution contract

CUDA calls usually enqueue work into streams. Operations in one stream execute in order. Independent operations in different streams may overlap if dependencies, resources, and hardware engines allow it. "May overlap" is important: concurrency is permitted, not guaranteed.

An event records progress in a stream. Another stream can wait on that event without forcing the host to synchronize the whole device. This builds a dependency graph while preserving unrelated concurrency.

```cuda
cudaEvent_t ready;
cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);

cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, copy_stream);
cudaEventRecord(ready, copy_stream);
cudaStreamWaitEvent(compute_stream, ready, 0);
kernel<<<grid, block, 0, compute_stream>>>(d_x, d_y, n);
```

Avoid `cudaDeviceSynchronize()` in a request path unless a full-device boundary is truly required. It converts independent queues into a global critical path and can hide the operation that actually created an error.

### Overlapping transfer and compute

Host-device overlap normally requires pinned host memory, asynchronous copy APIs, distinct streams, hardware support, and independent buffers. Double buffering partitions input into chunks:

1. copy chunk `i + 1` to device;
2. compute chunk `i`;
3. copy result `i - 1` to host.

Chunk size balances transfer efficiency, pipeline fill/drain, and working-set memory. Tiny chunks pay submission overhead; huge chunks leave little overlap.

Pinned memory should be pooled. Pinning and unpinning are heavyweight, and excessive pinned memory harms the operating system by reducing pageable capacity.

### Default-stream hazards

Legacy default-stream semantics can synchronize with work in other streams. Per-thread default streams and nonblocking streams provide different behavior. Production code should choose a convention explicitly rather than rely on compilation defaults that vary across components.

Libraries may enqueue work on a caller-provided stream or an internal stream. Document stream ownership and lifetime. A tensor cannot be freed, reused, or modified until all streams that access it have completed their dependencies.

### Stream priorities and fairness

Stream priority is a scheduling hint, not a deadline guarantee. A high-priority kernel cannot preempt every already-running block. Long-running persistent kernels and huge blocks can therefore delay latency-sensitive work even when placed in a higher-priority stream.

Design cooperative yielding or partition capacity when strict service isolation matters. Measure interference using the production mixture, not isolated streams.

### Allocation on the critical path

Synchronous allocation can introduce global coordination. Stream-ordered allocation with `cudaMallocAsync` and `cudaFreeAsync` ties lifetime to stream progress and lets the runtime reuse a memory pool.

Pool policy includes release threshold, device and process scope, cross-stream dependencies, fragmentation, and warmup. A fast allocator does not excuse unbounded temporary memory. Track high-water mark and reserve behavior.

### CUDA Graphs

A CUDA Graph represents a repeated dependency graph of kernels, copies, and supported memory operations. Instantiating the graph performs setup once; launching the executable graph reduces repeated CPU submission overhead.

Graphs are valuable when:

- the sequence of operations repeats;
- kernel work is small enough that launch overhead matters;
- addresses and shapes are stable or can be updated within supported limits;
- the application can manage graph variants and lifetimes.

Graphs are less useful when the workload changes structure every iteration, graph updates are frequent and expensive, or one long kernel dominates latency.

```cuda
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
stage1<<<g1, b1, 0, stream>>>(a, tmp);
stage2<<<g2, b2, 0, stream>>>(tmp, out);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
for (int step = 0; step < steps; ++step) {
  cudaGraphLaunch(graph_exec, stream);
}
```

Graph executables and captured resources have ownership constraints. Reusing one executable graph concurrently may be restricted. Graph memory nodes and stream-ordered allocations have GPU-ordered lifetimes that must be respected.

### Shape buckets and graph variants

LLM engines often maintain graph variants for batch and token-count buckets. Padding to a captured shape trades extra device work for stable launch overhead and addresses. Too many buckets create compilation, memory, and test burden.

Choose buckets from traffic distribution and kernel sensitivity. Keep a general eager fallback. Track hit rate, padded work, graph memory, and update failures.

### Multi-GPU copies and peer access

Device-to-device transfers can use peer access over the available interconnect. The relevant path may be PCIe, NVLink, or another fabric, with topology-dependent bandwidth and routing. Batch small transfers, preserve alignment, and overlap communication only when independent compute exists.

For collectives, use a collective library rather than hand-rolled peer copies unless the communication pattern is genuinely unusual. The application-level question is still exposed communication on the critical path, not theoretical link bandwidth.

### End-to-end timing

CPU wall-clock timing around an asynchronous launch measures submission, not completion. CUDA events measure elapsed device time between points in a stream. End-to-end latency should use the product boundary and include queueing, copies, and synchronization.

```cuda
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

Warm up context creation, allocations, JIT compilation, graph upload, and caches according to the claim. Report both cold and warm behavior when users encounter both.

### Design Exercises

1. What conditions are required to overlap a host-device copy with compute?
2. Why can `cudaDeviceSynchronize()` destroy concurrency?
3. When do CUDA Graphs help, and when do they not?
4. How would you choose graph shape buckets for an inference engine?
5. Design a double-buffered input pipeline and name its failure modes.

### Worked Solutions

#### 1. Copy-compute overlap

The copy must be asynchronous, host memory should normally be pinned, source and destination buffers must not conflict with compute, the work must be in streams without an ordering dependency, and hardware must support the required concurrent engines. Overlap can still be limited by shared copy engines, memory bandwidth contention, or a chunk too small to amortize submission.

#### 2. Device synchronization

`cudaDeviceSynchronize()` waits for all previously submitted work on the device, including independent streams. It inserts a host-visible global boundary, prevents pipelining across iterations, and can turn recoverable queue slack into latency. Prefer events or stream synchronization scoped to the dependency actually required.

#### 3. CUDA Graphs

Graphs help repeated fine-grained workloads where CPU launch and runtime setup are visible. They do not accelerate the instructions inside a dominant kernel. They lose value when topology, shapes, addresses, or operation sequences change so frequently that capture, update, or variant management exceeds the saved launch cost.

#### 4. Graph buckets

Measure the joint distribution of batch size, token count, and model path. Choose a small set that captures most traffic with tolerable padding, while capping executable-graph memory and compile time. Route unusual shapes to an eager fallback. Optimize hit rate and saved latency against padded FLOPs and variant ownership, not bucket count alone.

#### 5. Double buffering

Allocate two or more host and device buffers. While compute consumes buffer A, copy the next input into B and return the prior output from another buffer. Connect stages with events. Failure modes include overwriting a buffer before its consumer completes, using pageable memory, chunks too small or large, hidden default-stream synchronization, copy and compute contending for memory bandwidth, and missing backpressure when the producer outruns the GPU.

## Hierarchical Matrix Multiplication

LEAD: GEMM performance comes from hierarchical reuse. Each fetched A and B value should participate in many multiply-accumulates before leaving on-chip storage.

:::diagram gemm_tiling|A and B tiles are cooperatively staged; warps consume subtiles; threads or tensor-core fragments accumulate C in registers before an epilogue writes it efficiently.

### From the equation to the schedule

For `C[M,N] = A[M,K] B[K,N]`, the mathematical work is approximately `2 M N K` FLOPs. The implementation chooses a loop ordering and a hierarchy of tiles:

- grid tiles partition M and N across thread blocks;
- a block loops over K tiles;
- warps own subtiles within the block tile;
- lanes own scalar accumulators or matrix fragments;
- an epilogue maps accumulator layout to coalesced output stores.

The kernel must balance parallel output tiles against reuse along K.

### A correct scalar baseline

```cuda
__global__ void gemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}
```

This is a valuable oracle and a poor high-performance schedule. Neighboring threads reread A rows and B columns instead of explicitly sharing them.

### Cooperative shared-memory tiling

```cuda
template<int TILE>
__global__ void gemm_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    int ak = k0 + threadIdx.x;
    int bk = k0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && ak < K) ? A[row * K + ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bk < K && col < N) ? B[bk * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) C[row * N + col] = acc;
}
```

Out-of-range loads become zero, all participating threads reach both barriers, and the output store is guarded. An early return before a later barrier would be incorrect for an edge block.

### Arithmetic intensity from tile reuse

Ignoring the final store, a square block tile of width `T` loads roughly `2 T^2` values per K tile and performs roughly `2 T^3` FLOPs. Its shared-tile arithmetic intensity is proportional to:

`2 T^3 FLOPs / (2 T^2 values * bytes_per_value) = T / bytes_per_value`

Larger T raises reuse, but resource consumption grows. The estimate also assumes each input tile is loaded once and does not include cache effects, epilogue traffic, or edge waste.

### Register tiling and microkernels

One thread can compute an `r_m x r_n` output patch. It loads a small vector from A and B, then updates multiple independent accumulators. Register tiling increases reuse from shared memory and exposes instruction-level parallelism.

The cost is `r_m * r_n` accumulator registers plus address and staging state. Large microtiles can lower occupancy and cause spills. A good microkernel balances accumulator reuse against live-state pressure.

### Tensor-core execution

Tensor cores execute matrix multiply-accumulate on architecture-defined fragments. A high-performance tensor-core kernel must:

- use supported input, accumulator, and output types;
- satisfy alignment and layout constraints;
- stage data in a layout that avoids shared-memory conflicts;
- map lanes to matrix fragments correctly;
- pipeline copies and MMA instructions;
- define numerical behavior for reduced precision and accumulation;
- execute an efficient epilogue.

Mixed precision changes the error model. FP16 or BF16 inputs with FP32 accumulation usually reduce accumulation error compared with low-precision accumulation, but input rounding and operation order still differ from an FP32 reference. TF32, FP8, INT8, and block-scaled formats each require a stated quality contract.

### Multi-stage pipelines and asynchronous copies

Let stage `s` hold the tile being consumed and stage `s + 1` the tile being filled. A software pipeline repeats:

1. issue an asynchronous copy for a future A/B tile;
2. wait until the current stage is ready;
3. execute MMA on the current stage;
4. advance the circular stage index.

More stages can cover memory latency but consume more shared memory. At some point the extra stage reduces resident blocks or adds address and synchronization overhead without improving tensor-core utilization.

Modern architectures may provide bulk-copy engines and warp-specialized schedules in which producer warps move data while consumer warps issue MMA. The transferable idea is division of labor with explicit pipeline state, not one architecture-specific mnemonic.

### Layouts and the epilogue

The accumulator layout that suits MMA instructions may not produce coalesced global stores. The epilogue remaps fragments, often through shared memory, and applies operations such as:

`D = activation(alpha * accumulator + beta * C + bias)`

Fusing bias, scaling, activation, quantization, or residual operations can eliminate a full read and write. An overgrown epilogue can increase registers, inhibit vector stores, or create too many specialized variants.

### Split-K, Stream-K, and persistent scheduling

When M and N expose too few output tiles but K is large, split K across blocks. Partial C tiles are combined with atomics or a workspace reduction. The extra reduction is worthwhile when it unlocks otherwise idle SMs.

Stream-K-like scheduling distributes K work more evenly across a fixed set of work units to reduce wave quantization and tail imbalance. Persistent kernels keep blocks resident and pull tiles from a global or hierarchical queue. These strategies improve utilization for awkward shapes but add coordination and deterministic-order questions.

### Batched and grouped GEMM

**Batched GEMM** executes many matrices with compatible shapes and strides. **Grouped GEMM** accepts varying shapes and offsets. Both appear in attention heads, adapters, and mixture-of-experts layers.

Small matrices may be launch- and scheduling-bound. Grouping several problems into one persistent kernel increases available work but requires load balancing. Sorting by shape can improve efficiency while adding preprocessing and reordering cost.

### Library, template framework, or custom kernel

Use cuBLAS or another vendor library for standard dense GEMM. Use a template framework such as CUTLASS when you need controlled layouts, data types, schedules, or epilogues without writing every hardware primitive. Write a custom kernel when the operator is unusual enough that the library boundary creates material traffic or launch cost.

The decision is based on end-to-end value, maintainability, architecture coverage, and test burden. Beating a library on one shape is not the same as owning a production GEMM.

### Design Exercises

1. Why does tiling increase arithmetic intensity?
2. Fix the out-of-bounds and barrier bug in a naive shared-memory kernel.
3. What tradeoff determines tile size and pipeline depth?
4. When does split-K help, and what new cost does it create?
5. How do tensor cores change implementation and error analysis?
6. Why does a high-performance GEMM need an epilogue?

### Worked Solutions

#### 1. Tiling and arithmetic intensity

A tile is loaded once into shared memory and reused by many output accumulators. For a square width-T tile, input traffic grows like `T^2` while multiply-accumulate work grows like `T^3`, so operations per loaded byte grow with T. The exact gain depends on cache, dtype, edge tiles, and whether the compiler actually preserves the intended reuse.

#### 2. Bounds and barriers

Do not return threads that will be needed at a later `__syncthreads()`. Every thread loads either a valid element or zero into each shared tile, then all threads reach the barrier. After compute, all threads reach the second barrier before the tile is overwritten. Only the final global store is predicated on a valid output coordinate.

#### 3. Tile size and pipeline depth

Larger tiles and more stages increase reuse and overlap. They also consume shared memory, registers, threads, and instruction budget, which can reduce residency or spill. The optimum is the smallest resource footprint that keeps copy and compute pipelines busy for representative shapes. Use the occupancy calculation and achieved-stall profile, then benchmark nearby configurations.

#### 4. Split-K

Split-K helps when the M-by-N tile grid is too small to occupy the device and K is large enough to partition. It creates multiple partial results for each C tile, requiring atomic accumulation or a separate reduction workspace. That adds traffic, synchronization, possible nondeterminism, and initialization cost. It loses when ordinary output parallelism is already sufficient.

#### 5. Tensor cores and error

Implementation changes from scalar FMA loops to architecture-defined matrix fragments, with stricter dtype, alignment, and layout requirements. The kernel must pipeline data into the MMA path and map fragments to an output epilogue. Error analysis must specify input conversion, accumulation dtype, rounding, operation order, saturation or scaling for integer and FP8 paths, and tolerance versus a high-precision reference.

#### 6. The epilogue

MMA accumulator fragments are arranged for compute, not necessarily coalesced output. The epilogue rearranges them for efficient stores and applies `alpha`, `beta`, bias, activation, residual, or quantization while the values are still on chip. Without it, output stores may be inefficient and downstream elementwise kernels would reread and rewrite the entire matrix.

## Reductions, Prefix Scans, and Histograms

LEAD: Collective primitives expose the essential GPU issues: participation masks, associative structure, synchronization scope, contention, and the boundary between local and device-wide coordination.

### Reduction as a tree

A reduction replaces a linear dependency with a tree. Within a warp, shuffle instructions exchange register values. Across warps, one common pattern writes a partial per warp to shared memory and lets the first warp finish.

```cuda
__device__ float warp_sum(float x, unsigned mask) {
  int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, x, offset);
    int source_lane = lane + offset;
    if (source_lane < 32 && (mask & (1u << source_lane))) x += other;
  }
  return x;
}

__device__ float block_sum(float x) {
  __shared__ float partial[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  unsigned mask = __activemask();

  x = warp_sum(x, mask);
  if (lane == 0) partial[warp] = x;
  __syncthreads();

  int warp_count = (blockDim.x + 31) >> 5;
  float value = (threadIdx.x < warp_count) ? partial[lane] : 0.0f;
  if (warp == 0) {
    unsigned first_warp_mask = __ballot_sync(0xffffffff, lane < warp_count);
    value = warp_sum(value, first_warp_mask);
  }
  return value;  // valid in lane 0 of warp 0
}
```

The ownership contract matters: only one lane owns the final value. A caller that lets every thread store `value` is incorrect.

### Device-wide reduction

Blocks cannot generally synchronize with arbitrary other blocks in one ordinary launch. Common strategies are:

- first kernel writes one partial per block; second kernel reduces partials;
- blocks atomically accumulate into one or several outputs;
- a persistent/cooperative kernel performs a grid-wide protocol on supported launches;
- use a production primitive such as CUB.

Atomics are attractive when there are few blocks or contention is low. A two-pass reduction has extra launch and workspace cost but scales predictably.

### Numerical reduction

Floating-point addition is not associative. A parallel tree changes result order and can vary across launch shape. Accumulating FP16 or BF16 values in FP32 improves stability. Pairwise trees often have better error than one long linear sum. Compensated summation improves accuracy but adds instructions and state.

Determinism may require a fixed partition and reduction order, which can reduce performance. State whether reproducibility means bitwise identity, stable tolerance, or stable model-level metrics.

### Prefix scan

An exclusive scan of `[a, b, c, d]` produces `[identity, a, a+b, a+b+c]`. An inclusive scan includes the current element. Scan powers stream compaction, radix sort, cumulative offsets, token packing, and sparse indexing.

A work-efficient block scan has an upsweep that builds partial sums and a downsweep that propagates prefixes. Warp-shuffle scans reduce shared-memory traffic within warps; shared memory carries warp totals.

```cuda
__device__ int warp_inclusive_scan(int x, unsigned mask) {
  int lane = threadIdx.x & 31;
  for (int offset = 1; offset < 32; offset <<= 1) {
    int y = __shfl_up_sync(mask, x, offset);
    if (lane >= offset) x += y;
  }
  return x;
}
```

For a device-wide scan, each block scans a tile, block totals are scanned, and the resulting offsets are added to each tile. A library handles edge cases, recursion, tuning, and temporary storage.

### Stream compaction

Compaction keeps elements whose predicate is true:

1. compute a zero-or-one flag;
2. exclusive-scan flags into output positions;
3. scatter selected elements;
4. derive output count from the last prefix and flag.

This avoids one atomic per selected item. For very sparse or small workloads, atomics can be simpler and competitive. Compare contention and scan overhead.

### Histograms

A naive histogram performs one global atomic per sample. If many samples choose the same bin, contention serializes updates. Common improvements:

- privatize bins per warp or block in shared memory;
- aggregate repeated values within a warp before one atomic;
- use a sort-and-run-length strategy for highly concentrated inputs;
- partition bins or data across blocks, then merge partial histograms.

Shared-memory atomics are not free, and a large bin count may not fit. The input distribution controls the best method. Benchmark uniform, skewed, and adversarial distributions.

### Segmented reductions

A segmented reduction combines values within variable-length groups. Segments may be represented by offsets, keys, or boundary flags. Short segments waste blocks if assigned one per block; long segments need multi-block cooperation.

Sort or bucket by size, combine tiny segments per block, and split very long segments. The overhead of preprocessing must be amortized across repeated operations or large workloads.

### Gather, scatter, and irregular access

Gather reads `out[i] = input[index[i]]`; scatter writes to indexed destinations. Address locality, duplicate indices, and output conflicts dominate. Sort or reorder indices to improve locality when semantics allow. For scatter with duplicates, define whether updates are last-writer, atomic sum, max, or invalid.

Caching helps repeated gather indices, but random accesses can remain latency-bound. Vectorizing random accesses does not make them coalesced.

### Use production primitives deliberately

CUB provides warp-, block-, and device-wide reductions, scans, histograms, and selection. Use it unless a fused operator, unusual data type, fixed small shape, or special semantics justify custom code.

The value of implementing a primitive is understanding its invariants. The production decision still favors a maintained library when the boundary fits.

### Design Exercises

1. Write and explain a warp-plus-block sum reduction.
2. When would you choose atomics over a two-pass device reduction?
3. Explain a device-wide exclusive scan.
4. Design stream compaction and derive the output count.
5. How would you optimize a histogram with severe hot-bin contention?
6. What does deterministic reduction cost?

### Worked Solutions

#### 1. Warp-plus-block reduction

Each warp reduces its lanes with shuffles under the correct active mask. Lane zero writes a warp partial to shared memory. A block barrier ensures all partials are visible. The first warp loads those partials, filling inactive entries with the identity, and reduces again. Lane zero of the first warp owns the block result. The answer must state mask construction, identity values, barrier placement, and result ownership.

#### 2. Atomics versus two passes

Use atomics when the number of blocks is modest, contention is low, the hardware supports the type efficiently, or the output has several independent bins. Use a two-pass reduction when many blocks target one location, deterministic order matters, or atomic serialization dominates. The two-pass method costs workspace, another launch, and another read/write of partials.

#### 3. Device-wide scan

Each block scans a tile and writes its total. Recursively scan block totals to produce a prefix offset for each tile. Launch a uniform-add phase that adds the tile offset to every local prefix. Production single-pass variants exist, but this three-stage decomposition is easy to reason about and test. The operator must be associative and the identity must be correct.

#### 4. Stream compaction

Map the predicate to flags. Exclusive-scan the flags; each true element writes to `output[prefix[i]]`. The final count is `prefix[n-1] + flag[n-1]` for nonempty input. Handle `n = 0`, avoid duplicate writers, and choose stable or unstable semantics explicitly.

#### 5. Hot-bin histogram

Reduce the number of contending atomics. Privatize bins per warp or block, aggregate equal values inside a warp, then merge partial histograms. If the bin range is large or the distribution is extremely concentrated, sort or partition keys and reduce runs. Measure shared-memory footprint, merge traffic, and skewed cases; a uniform-only benchmark hides the problem.

#### 6. Deterministic reduction

Determinism fixes partitioning and operation order, restricts atomic races, and may require extra passes or workspace. This can reduce load balancing, eliminate faster nondeterministic atomics, and lower parallelism. The cost is justified only when bitwise reproducibility is a real requirement; many applications need a numerical tolerance rather than identical bits.

## Softmax, Normalization, Top-K, and Sampling

LEAD: These kernels are small relative to GEMM in FLOPs and large in system importance. They are often bandwidth- or launch-bound, numerically sensitive, and excellent fusion candidates.

### Stable softmax

For a row `x`, softmax is:

`p_i = exp(x_i - m) / sum_j exp(x_j - m)`, where `m = max_j x_j`

Subtracting the row maximum prevents overflow. A basic row kernel has three logical phases:

1. reduce to the row maximum;
2. reduce the shifted exponentials to the denominator;
3. normalize and write outputs.

```cuda
float local_max = -INFINITY;
for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  local_max = max(local_max, x[row * cols + i]);
}
float row_max = block_max(local_max);

float local_sum = 0.0f;
for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  local_sum += expf(x[row * cols + i] - row_max);
}
float row_sum = block_sum(local_sum);

for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  y[row * cols + i] = expf(x[row * cols + i] - row_max) / row_sum;
}
```

The code may recompute exponentials in the final loop. Keeping every value in registers can be faster for narrow rows but creates spills for wide rows. Recompute spends special-function arithmetic to avoid HBM or local-memory traffic.

### Online softmax derivation

Suppose a processed prefix has maximum `m_a` and exponential sum `l_a = sum exp(x - m_a)`. A new tile has maximum `m_b` and local sum `l_b = sum exp(x - m_b)`. The combined maximum is:

`m = max(m_a, m_b)`

Rescale each partial into the new reference frame:

`l = exp(m_a - m) l_a + exp(m_b - m) l_b`

This state `(m, l)` is associative up to floating-point order. Adding an output accumulator produces the recurrence used in tiled attention.

### Masking and edge cases

A masked element contributes negative infinity before the maximum. A fully masked row is a policy decision: produce zeros, NaNs, or a declared fallback. Silently evaluating `exp(-inf - -inf)` produces invalid arithmetic.

Causal masks can avoid loading disallowed tiles. Arbitrary masks may add branches or extra memory traffic. Packed ragged sequences require per-row lengths and careful prevention of cross-sequence reads.

### Softmax backward

Given output `p` and upstream gradient `g`, the gradient with respect to logits is:

`dx_i = p_i * (g_i - sum_j g_j p_j)`

This needs one dot-product reduction followed by an elementwise pass. Fusing the reduction and output when the row fits on chip avoids an intermediate. Accumulate the dot product in higher precision for low-precision inputs.

### LayerNorm

LayerNorm over a row computes:

`mean = sum x_i / N`

`variance = sum (x_i - mean)^2 / N`

`y_i = gamma_i * (x_i - mean) / sqrt(variance + epsilon) + beta_i`

A one-pass Welford state `(count, mean, M2)` is numerically stable and mergeable across lanes. A two-pass mean-then-variance kernel can be competitive when rows are moderate and input rereads hit cache.

LayerNorm backward reduces both `sum(dy times gamma)` and `sum(dy times gamma times x_hat)` for each row, then applies the closed-form gradient. Gradients for `gamma` and `beta` reduce across rows and may require a second dimension of parallelism.

### RMSNorm

RMSNorm omits mean subtraction:

`inv_rms = 1 / sqrt(mean(x_i^2) + epsilon)`

`y_i = gamma_i x_i inv_rms`

It needs one reduction and is simpler to fuse. Accumulate squares in FP32 when inputs are FP16 or BF16. Extremely large values can still overflow before accumulation if conversion or scaling is wrong.

### Residual and normalization fusion

A common transformer sequence is:

`residual = x + branch`

`normalized = RMSNorm(residual)`

Fusing residual addition, optional bias, normalization, and perhaps quantization can eliminate one or more full-tensor round trips. The kernel must still expose the updated residual if a later branch needs it.

Fusion tradeoffs include:

- saved HBM bytes and launches;
- extra live values and register pressure;
- more output tensors;
- dropout or RNG semantics;
- backward-save requirements;
- variant count across dtype and hidden size.

### Top-k selection

Full sorting is unnecessary for small k. A common hierarchy is:

1. each thread scans a strided subset and keeps local candidates;
2. lanes merge candidates within a warp;
3. warps merge through shared memory;
4. the block emits k values and indices.

For very small k, register insertion networks or compare-exchange sequences work well. For larger k, radix selection, heap-like structures, or partial sorting may win. The best method depends on vocabulary size, k, dtype, and batch.

### Distributed vocabulary top-k

If logits are sharded across `P` ranks, each rank computes its local top-k with global token IDs. The global top-k must lie in the union of `P * k` candidates, so ranks can all-gather candidates and select again.

Communication is small relative to gathering the full vocabulary. For top-p, local top-k may not include enough mass; use a larger candidate bound, histogram/bucket thresholding, or an exact distributed algorithm with additional rounds.

### Sampling pipeline

A sampling kernel may apply temperature, repetition or presence penalties, forbidden-token masks, top-k, top-p, normalization, and RNG. Fusing the pipeline reduces launches but complicates exact semantics and variant coverage.

Counter-based RNG maps request, sequence position, and sample index to deterministic random bits without mutable per-thread generator state. Reordering requests must not change a user's random stream unless the product contract permits it.

### Design Exercises

1. Why is stable softmax logically a maximum reduction and a sum reduction?
2. Derive the online-softmax merge rule.
3. When is recomputing `exp` cheaper than storing it?
4. Compare LayerNorm and RMSNorm kernel structure.
5. How would you implement distributed vocabulary top-k?
6. What can go wrong when fusing sampling operations?

### Worked Solutions

#### 1. Two reductions

The maximum establishes a safe numerical reference so shifted exponentials cannot overflow. The denominator is then a sum over those shifted exponentials. The second reduction depends on the first result, so they are logically sequential even if each reduction is parallel. Omitting the max risks overflow; using a local max without global row combination gives the wrong distribution.

#### 2. Online merge

For partial states `(m_a, l_a)` and `(m_b, l_b)`, choose `m = max(m_a, m_b)`. Because `l_a` was measured relative to `m_a`, convert it by multiplying with `exp(m_a - m)`; do the analogous conversion for `l_b`; then add. This maintains `l = sum exp(x - m)` for the union. The same rescaling applies to a weighted output accumulator.

#### 3. Recompute versus store

Recompute wins when storing exponentials would spill registers or create a global/local-memory round trip whose latency and bandwidth cost exceeds another `exp`. It is especially attractive on compute-rich GPUs for moderate rows. Storing wins when values already fit in registers, `exp` throughput is the bottleneck, or the values are needed by several later operations.

#### 4. LayerNorm versus RMSNorm

LayerNorm needs mean and variance, usually Welford or two reductions, then scale and bias. RMSNorm needs only mean square, inverse RMS, and scale. Both are often bandwidth-bound and benefit from residual fusion. LayerNorm backward has two rowwise statistics; RMSNorm backward is simpler. Both should accumulate statistics in higher precision for low-precision inputs.

#### 5. Distributed top-k

Each rank selects local top-k pairs `(score, global_token_id)`. All-gather the `P * k` pairs and perform a second top-k. This is exact because an item outside a rank's local top-k has at least k better items on that rank and therefore cannot enter the global top-k. Define tie-breaking and NaN handling consistently.

#### 6. Fused sampling risks

Fusion can change operation order, numerical rounding, tie behavior, RNG indexing, and top-p truncation semantics. It can also create high register pressure and too many variants. Validate outputs statistically and, where exact reproducibility is promised, bitwise against the reference for every policy combination. Preserve request-specific RNG under batching and reordering.

## FlashAttention and IO-Aware Exact Attention

LEAD: FlashAttention is not an approximate attention mechanism. It is an exact tiled schedule that avoids materializing the quadratic score and probability matrices in HBM.

:::diagram attention|Attention combines QK scores, masking and softmax, then a weighted V reduction. The performance question is which intermediates must cross HBM.

### The naive schedule

For one head, standard attention is:

`S = Q K^T * scale`

`P = softmax(S + mask)`

`O = P V`

A naive implementation writes `S` to HBM, reads it for softmax, writes `P`, reads `P` for the V product, and writes `O`. `S` and `P` each contain `sequence^2` elements.

Even if the GEMMs are efficient, moving quadratic intermediates can dominate memory and prevent long contexts from fitting.

### Tiled forward algorithm

Partition Q into row tiles and K/V into column tiles. For each Q tile:

1. load Q and initialize per-row maximum `m`, sum `l`, and output accumulator `o`;
2. load a K/V tile;
3. compute local scores with tensor-core or SIMT matrix multiply;
4. apply scale, causal or explicit mask, and local row maxima;
5. rescale old state and add local exponentials;
6. update the output accumulator with local probabilities times V;
7. continue across K/V tiles;
8. normalize and store O plus compact row statistics.

Q can remain on chip while K/V tiles stream through, or the work can be repartitioned based on sequence length and hardware resources.

### Deriving the output recurrence

Suppose existing state uses reference max `m_a`, denominator `l_a`, and unnormalized output `o_a = sum exp(s_i - m_a) v_i`. A new tile has state `(m_b, l_b, o_b)`.

Set:

`m = max(m_a, m_b)`

`alpha = exp(m_a - m)`

`beta = exp(m_b - m)`

Then:

`l = alpha l_a + beta l_b`

`o = alpha o_a + beta o_b`

The normalized output is `o / l`. Every term is now expressed relative to the same maximum.

### Causal and ragged workloads

For causal attention, a Q tile does not need future K/V tiles. Diagonal tiles require element-level masking; tiles entirely above the causal diagonal can be skipped. This creates different work per Q tile, so static assignment can leave tail imbalance.

Ragged batches add per-sequence lengths and offsets. Pack sequences to avoid padding, but ensure no tile crosses sequence identity. Sort or bucket by length only if reordering cost and output restoration are controlled.

### Work partitioning

The original IO schedule is only the first step. Efficient implementations reduce non-matmul scalar work, increase parallelism across sequence tiles, and assign warp roles to minimize shared-memory exchange.

At short sequence or small batch, there may be too few heads and Q tiles to occupy the GPU. Splitting one head across more blocks increases parallelism but requires combining partial softmax/output states. The merge uses the same max-and-sum recurrence.

### Backward equations

Let upstream gradient be `dO`. The conceptual backward is:

`dV = P^T dO`

`dP = dO V^T`

`dS = P * (dP - rowsum(dP * P))`

`dQ = dS K * scale`

`dK = dS^T Q * scale`

The kernel recomputes score tiles and P from Q, K, and saved row statistics rather than reading a stored quadratic P. This adds arithmetic and saves quadratic storage and HBM traffic.

Backward partitioning is difficult because K and V receive contributions from many Q tiles. Strategies assign ownership to avoid atomics, use separate kernels, or accumulate partials with controlled reductions. The fastest plan depends on head dimension, sequence, and batch.

### What must be saved

At minimum, backward needs Q, K, V or access to recompute them, output O or equivalent state for some formulations, per-row log-sum-exp or max/sum statistics, masks or lengths, scale, and dropout RNG information if dropout is used.

Saving less increases recomputation. Saving more increases activation memory. The chosen boundary belongs in the model's activation-checkpointing plan.

### Dropout and reproducibility

Attention dropout must regenerate the same mask in backward. Counter-based RNG can derive mask bits from batch, head, row, column, and a seed/offset. Tiling and work partition changes must not silently change the logical random mask when reproducibility is promised.

### Training versus decode attention

FlashAttention-style training and prefill process many queries and exploit matrix-multiply reuse. Decode has one new query per sequence and reads a long KV cache. It is usually memory-bandwidth and latency sensitive, with a very different optimal schedule.

Calling every fused attention kernel "FlashAttention" hides this distinction. State whether the workload is training, prefill, or decode, and model the corresponding shapes.

### Fusion boundaries

Useful attention fusion may include Q/K scaling, RoPE, masking, bias, softmax, dropout, or output transformations. Fusion loses when it inflates live state, reduces occupancy, complicates recomputation, or creates a variant explosion.

### Design Exercises

1. Why can FlashAttention perform more arithmetic yet run faster?
2. Derive the running max, sum, and output update.
3. What must be saved for backward?
4. How does causal masking affect work partitioning?
5. When does attention fusion reduce performance?
6. Why is a decode-attention kernel different from a training kernel?

### Worked Solutions

#### 1. More arithmetic, less time

FlashAttention recomputes or performs extra scalar work but removes reads and writes of quadratic score/probability matrices. Modern accelerators have far more arithmetic throughput than HBM bandwidth relative to this workload, so spending additional FLOPs to reduce HBM traffic lowers wall time and activation memory.

#### 2. Running state

Merge old and new tile maxima into `m = max(m_a, m_b)`. Rescale the old denominator and output by `exp(m_a - m)` and the new tile by `exp(m_b - m)`. Add the rescaled sums and weighted outputs. This preserves the exact softmax numerator and denominator relative to one stable maximum.

#### 3. Backward state

Save or make recomputable Q, K, and V; save output or the formulation's equivalent; save per-row log-sum-exp or max/sum statistics; and preserve mask, lengths, scale, and dropout RNG mapping. Full attention probabilities need not be saved because score tiles can be recomputed.

#### 4. Causal work partitioning

Q tiles near the beginning attend to fewer K/V tiles than later Q tiles, and tiles above the causal diagonal are skipped. This creates triangular work and tail imbalance. Assign tiles dynamically, process longer rows first, or split long rows across blocks and merge partial states. Diagonal tiles still need fine-grained masking.

#### 5. Fusion losses

Fusion can increase registers and shared memory, reduce resident blocks, prevent use of optimized GEMM/Tensor Core paths, force incompatible shapes into one schedule, or multiply compile variants. It is beneficial only when saved launches and intermediate bytes exceed these costs for real workloads.

#### 6. Decode versus training

Training/prefill has many query rows and large tiles, enabling compute reuse and tensor-core efficiency. Decode has one new query per active sequence, streams a long and often paged KV cache, and is commonly bandwidth- and latency-bound. Its schedule emphasizes KV layout, vectorized loads, batch mapping, and low launch overhead rather than quadratic-score avoidance.

## CUDA Kernels for LLM Inference

LEAD: LLM serving kernels operate on dynamic batches, ragged state, low precision, and strict token cadence. Layout and scheduler contracts are part of kernel design.

### Embedding lookup and output projection

Embedding lookup is a gather. Adjacent output dimensions for one token are contiguous; token IDs across requests are irregular. Assign lanes across the embedding dimension so one token's vector loads coalesce. Repeated token IDs may benefit from cache, but do not rely on it for the worst case.

The output projection is a GEMM against the vocabulary matrix. For small decode batch it can be bandwidth-bound. Tensor parallelism shards vocabulary or hidden dimension, after which top-k and sampling need distributed coordination.

### RoPE

Rotary position embedding rotates pairs of query and key components. For pair `(x_0, x_1)` and angle `theta`:

`y_0 = x_0 cos(theta) - x_1 sin(theta)`

`y_1 = x_0 sin(theta) + x_1 cos(theta)`

The kernel can fuse RoPE into Q/K projection output handling or KV-cache write. Precomputed sine/cosine tables trade memory reads for transcendental arithmetic. Generate angles on the fly when table bandwidth or very long positions make it worthwhile.

Layout must define whether paired dimensions are adjacent or split into halves. A mismatch can produce numerically plausible but incorrect results.

```cuda
float2 pair = load_pair(q, token, head, pair_id);
float c = cos_table[position * pairs + pair_id];
float s = sin_table[position * pairs + pair_id];
float2 rotated = make_float2(
    pair.x * c - pair.y * s,
    pair.x * s + pair.y * c);
store_pair(q_out, token, head, pair_id, rotated);
```

### KV-cache append

Each decode step writes one K and V vector per layer and sequence. The write is small and frequent. Fuse layout conversion, RoPE for K, and quantization when doing so reduces full-vector traffic without blocking the main attention path.

The address depends on logical sequence, token position, layer, KV head, page mapping, and dtype. Keep that indexing contract centralized. A one-off mismatch between append and read kernels corrupts attention silently.

### Paged KV attention

A block table maps logical token blocks to physical pages. Decode attention traverses pages, loads K/V vectors, computes query-key scores, performs online softmax, and accumulates values.

Key design decisions:

- tokens per page;
- K/V head and dimension layout inside a page;
- vector width and alignment;
- how sequences map to blocks or warps;
- how GQA query heads share one KV head;
- how page-table lookups are cached or prefetched;
- where quantization scales reside;
- how partial results merge for long context.

Smaller pages reduce fragmentation and copy cost but increase page-table entries and discontinuities. Larger pages improve contiguous access but waste more tail capacity.

### Split-K decode attention

One sequence with long context may not expose enough parallel work if one block owns the entire KV range. Partition context across blocks. Each block emits partial `(m, l, o)` state; a second phase merges states with the online-softmax recurrence.

This is analogous to split-K GEMM: more parallelism in exchange for a partial-result reduction. It helps long contexts and small batches, then loses when batch already supplies enough blocks.

### GQA and MQA mapping

With grouped-query attention, several query heads read one KV head. Map those query heads close enough to reuse K/V cache lines or shared tiles. Excessive sharing can increase live output accumulators. The kernel chooses between reloading K/V for simpler ownership and retaining K/V while processing multiple query heads.

### KV-cache quantization

Quantized K/V reduces capacity and bandwidth. The kernel reads packed values plus scales, dequantizes into compute registers, and accumulates in higher precision. Scale granularity may be per tensor, head, channel, page, or token group.

Finer scales improve fidelity and increase metadata traffic. Dynamic token scales add append-time reduction. K and V can have different sensitivity and therefore different formats.

### Weight-only quantized GEMM

For low-batch decode, weight bandwidth dominates many linear layers. A weight-only kernel loads packed weights, dequantizes them, and multiplies by higher-precision activations. The schedule must overlap unpack/dequant with MMA and load scales efficiently.

Compression is useful only if the hardware path processes packed data efficiently. A format can reduce stored bytes and still run slower because unpacking, irregular scales, or unsupported tensor-core instructions dominate.

### MoE routing and grouped GEMM

MoE routing computes top experts, counts tokens per expert, scans counts into offsets, scatters tokens into expert-contiguous buffers, executes grouped GEMM, then scatters outputs back with routing weights.

The pipeline combines top-k, histogram, scan, gather/scatter, and GEMM. Fusing every phase is rarely ideal. The important interfaces are compact routing metadata and layouts that let grouped GEMM consume contiguous expert batches.

Hot experts create imbalance. Capacity limits, token dropping, or expert replication change semantics and belong to the model contract, not only the kernel.

### Sampling and token cadence

The decode iteration often ends with logits processing and sampling. At small batch, several tiny kernels can add visible launch latency. Graph capture and carefully scoped fusion help. Preserve request cancellation, dynamic sampling policies, and RNG mapping.

### Design Exercises

1. How would you fuse RoPE with KV-cache append?
2. Explain the page-size tradeoff in paged KV attention.
3. When does split-K decode attention help?
4. How should a GQA kernel exploit K/V sharing?
5. Why can KV quantization increase capacity but reduce speed?
6. Walk through an MoE routing kernel pipeline.

### Worked Solutions

#### 1. RoPE plus append

The Q/K projection produces K in a temporary layout. Instead of writing K, rereading it for RoPE, then writing the cache, assign lanes across K pairs, apply the rotation in registers, convert to the cache layout, optionally quantize, and store directly to the physical page. Preserve a separate Q output path, correct position indexing, and the model's pair layout. Fusion loses if it delays a library GEMM epilogue or inflates registers excessively.

#### 2. Page size

Small pages reduce internal fragmentation and make branch/copy operations cheaper, but increase block-table size, address translation, and discontinuous loads. Large pages improve contiguous vector access and reduce metadata but waste more memory in partially filled tails and make copy-on-write heavier. Choose from context distribution, kernel transaction efficiency, allocator pressure, and sharing behavior.

#### 3. Split-K decode

It helps when batch and head count provide too few blocks while context is long enough to partition. Each block scans a KV range and emits partial softmax/output state; a second phase merges. It costs partial-state memory, another synchronization or launch, and extra arithmetic. It loses at high batch or short context.

#### 4. GQA reuse

Map query heads that share one KV head into the same block or nearby warps so a loaded K/V tile serves several queries. Balance that reuse against more output accumulators and registers. Another design processes fewer query heads per block and reloads K/V, trading bandwidth for occupancy. Benchmark by head dimension, group size, context, and batch.

#### 5. Quantized KV slower

Capacity grows because stored state is smaller, but latency can worsen if scale loads, address calculations, unpacking, type conversion, or uncoalesced packed layout exceed the bandwidth saved. The kernel may also lose vector width or tensor-core eligibility. Measure total bytes and instruction throughput on the exact decode shapes.

#### 6. MoE pipeline

Compute router logits and top experts; form token-expert pairs; count pairs per expert; exclusive-scan counts into expert offsets; scatter token rows into expert-contiguous buffers; execute grouped GEMMs; multiply by routing weights; scatter-add outputs back to token order. Handle capacity, duplicate expert choices, load imbalance, quantization, and multi-GPU all-to-all explicitly.

## Kernel Engineering, Profiling, and Correctness

LEAD: Kernel work is complete only when the optimization is reproducible across representative shapes, numerically valid, integrated into the application, and understandable from evidence.

:::diagram roofline|A roofline places achieved work against arithmetic intensity. Hierarchical rooflines can reveal whether HBM, L2, L1, or compute is the active ceiling.

### Build a trustworthy benchmark

Specify:

- device, architecture, clocks or power mode, toolkit, driver, and compiler flags;
- input shapes and their production frequency;
- dtype, layout, alignment, sparsity, and value distribution;
- warmup, iteration count, synchronization boundary, and cache policy;
- reference implementation and numerical tolerance;
- whether allocation, transfer, launch, and framework dispatch are included.

Random inputs can hide value-dependent behavior such as histogram contention, sparsity, overflow, or early exit. Include adversarial and production-derived distributions.

### Cold, warm, and steady-state claims

Cold latency may include CUDA context initialization, module load, JIT compilation, graph instantiation, allocation, and cache miss. Warm latency may assume all of these are complete. Both are legitimate if labeled.

For interactive systems, report p50 and tail distributions over the real request mix. Median kernel time alone does not predict queueing or graph-cache miss.

### Roofline analysis

Arithmetic intensity is useful operations divided by bytes moved across a chosen boundary. The simple roofline bound is:

`attainable_work_rate <= min(compute_peak, bandwidth * arithmetic_intensity)`

Choose the correct work units and peak. Tensor-core FLOPs, scalar FP32 FLOPs, integer operations, and special functions have different ceilings. Choose the correct bytes: requested HBM bytes, measured HBM traffic, or cache-level traffic.

A hierarchical roofline adds L1 and L2 ceilings. A kernel may sit below the HBM roof because it is actually limited by L1 bandwidth, shared-memory conflicts, instruction issue, or dependency latency.

### From profiler sections to a hypothesis

A disciplined Nsight Compute investigation asks:

1. Is the kernel compute-, bandwidth-, latency-, or synchronization-limited?
2. Is achieved occupancy different from theoretical occupancy?
3. Are there enough eligible warps per scheduler?
4. Which stall reasons are large, and which instructions create them?
5. Are global and shared transactions efficient?
6. Are tensor-core, scalar, load/store, and special-function pipes balanced?
7. Is the tail caused by grid size or per-block imbalance?

Do not optimize a counter in isolation. A higher cache hit rate can hurt if obtaining it adds instructions or reduces occupancy. A lower stall percentage can mean the denominator changed.

### Read generated code

Compiler reports show registers, shared memory, and spills. SASS or disassembly confirms vectorization, tensor-core instructions, local-memory traffic, and unexpected conversions. Source code expresses intent; generated instructions determine the schedule.

Inspect generated code after large performance changes, unexpected register cliffs, or architecture/compiler upgrades. Avoid hand-tuning assembly before proving the compiler path is the bottleneck.

### Correctness layers

1. **Semantic reference:** compare with a simple CPU or high-level implementation.
2. **Shape coverage:** zero, singleton, odd, prime, aligned, maximum, ragged, and empty slices.
3. **Value coverage:** zeros, infinities where valid, NaNs, extreme magnitude, repeated keys, adversarial masks.
4. **Concurrency:** run with multiple streams, cancellation, reused buffers, and sanitizer tools.
5. **Numerical analysis:** absolute/relative error, ULP where useful, distributional error, and model-level quality.
6. **Gradient checks:** finite difference on tiny cases plus reference backward.
7. **Determinism:** define and test the promised level.

### Tolerances

Use a combined test such as:

`abs(actual - reference) <= atol + rtol * abs(reference)`

Choose tolerances from dtype, operation count, accumulation, and downstream sensitivity. One global tolerance can hide large error near zero or reject harmless scale-proportional error.

For softmax and probabilities, check normalization and KL or output behavior in addition to elementwise error. For top-k, ties require a declared ordering. For quantization, test task quality and long-context behavior.

### Race and memory checking

Run compute-sanitizer modes appropriate to memory access, race, initialization, and synchronization. Debug variants may reduce optimization or alter timing, so sanitizer success complements rather than replaces stress tests.

Add canaries around buffers and test noncontiguous strides. Many kernels are correct for dense tensors and silently wrong when framework views carry a different stride.

### Autotuning without overfitting

Tune over a declared shape distribution and cache the result by device, toolkit, dtype, layout, and semantic options. Separate a training set of shapes from a validation set to catch over-specialization.

Constrain the search space with resource models. Reject configurations that exceed register/shared limits or produce poor transaction geometry before benchmarking. Retain a safe general fallback.

### Integration and regression

A faster kernel can make the application slower through conversion, layout changes, synchronization, graph misses, or loss of fusion elsewhere. Benchmark the operator boundary and the end-to-end model.

Maintain performance thresholds by shape family, not one absolute number. Hardware load and clocks vary; use robust baselines, confidence bands, and dedicated performance workers where possible.

### Code review checklist

| Area | Review questions |
| --- | --- |
| Ownership | Which block, warp, and lane owns each output and partial? |
| Memory | Are addresses in bounds, aligned, coalesced, and correctly scoped? |
| Synchronization | Does every participant reach barriers? Are masks accurate? |
| Resources | What limits resident blocks? Are there spills or bank conflicts? |
| Numerics | What rounds, accumulates, rescales, or becomes nondeterministic? |
| Shapes | Which cases use specialized paths and which use fallback? |
| Integration | Which streams, graphs, allocations, and layouts are assumed? |
| Evidence | Which measurement proves the optimization helps end to end? |

### Design Exercises

1. Design a benchmark that distinguishes launch-bound from bandwidth-bound behavior.
2. A profiler reports low occupancy. What do you do next?
3. How do you interpret a high long-scoreboard stall percentage?
4. What belongs in a correctness suite for a fused attention kernel?
5. How do you choose numerical tolerances?
6. When should an autotuner create a specialized kernel variant?

### Worked Solutions

#### 1. Launch-bound versus bandwidth-bound

Sweep problem size and iteration count. Measure device event time for the kernel, an empty or minimal launch baseline, achieved memory bytes, and end-to-end submission. A launch-bound kernel has a relatively flat time floor and improves materially under graph capture or fusion. A bandwidth-bound kernel scales with bytes and approaches sustainable memory throughput; graph capture changes little once the workload is large.

#### 2. Low occupancy

Calculate the resource limiter: threads, registers, shared memory, or block limit. Then inspect achieved eligible warps and stall reasons. If execution units are busy and latency is hidden, do nothing. If exposed stalls correlate with too few warps, test a smaller tile, shorter live ranges, fewer stages, or launch bounds, watching for lost reuse and spills. Optimize performance, not the occupancy percentage.

#### 3. Long-scoreboard stalls

They commonly indicate waiting on long-latency memory dependencies, but the counter alone is not a diagnosis. Locate the load instructions, inspect coalescing and cache behavior, measure eligible warps, and check dependency chains. Possible fixes are improved locality, prefetch, independent loads, more resident warps, or a different data layout. A correctly HBM-bound kernel may legitimately retain these stalls.

#### 4. Fused-attention correctness

Compare forward and backward with a high-precision reference across causal/noncausal, ragged, boundary, GQA/MQA, masked, and extreme-logit cases. Test fully masked-row policy, dropout mask reproducibility, noncontiguous layouts, gradients, multiple streams, and sanitizer results. Verify no cross-sequence access and test tolerance by dtype and head dimension.

#### 5. Numerical tolerances

Start from input and accumulation dtypes, reduction length, operation conditioning, and expected rounding order. Use combined absolute and relative error, add invariant checks such as softmax sums, and examine worst-case plus distributional error. Tighten or loosen only with a documented numerical model and downstream quality evidence, not to make a failing test pass.

#### 6. Specialization

Create a variant when a frequent shape or semantic mode has a materially different optimal schedule and the end-to-end gain exceeds compile, binary, memory, dispatch, and test costs. Bucket nearby shapes when padding is cheaper than another variant. Require a fallback and monitor traffic hit rate so dead variants can be removed.

### Further Study and Primary References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - execution, memory, synchronization, streams, graphs, and architecture features.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - transfer, coalescing, bandwidth, occupancy, and optimization methodology.
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/) - profiler sections, metrics, rooflines, and analysis workflow.
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) - hierarchical GEMM mapping, mainloop, and epilogue design.
- [CUB Documentation](https://nvidia.github.io/cccl/unstable/cub/index.html) - maintained warp-, block-, and device-wide collectives.
- [FlashAttention paper](https://arxiv.org/abs/2205.14135) - IO-aware exact attention and tiling analysis.
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) - improved work partitioning and parallelism.

### Final CUDA Principle

> A kernel is a proof that an algorithm, a data layout, and a hardware schedule agree.

The proof has three parts: correctness for every supported shape, a resource model that predicts the bottleneck, and measurements that show the optimization survives integration. Missing any one produces a benchmark artifact rather than a production kernel.
