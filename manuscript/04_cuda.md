# Part IV - CUDA and Accelerator Programming

CUDA optimization is the discipline of translating an algorithm into a schedule over threads, instructions, memory levels, and asynchronous work. The correct starting point is never a favorite tile size or instruction. It is a resource model: what must move, what must be computed, which dependencies are unavoidable, and which hardware resource becomes limiting first.

This part is self-contained. It does not require Part III open beside it. Wherever serving concepts appear - KV cache, paging, grouped-query attention, mixture-of-experts routing - they are restated here at the level a kernel engineer needs. Part III remains the place for fleet scheduling and product SLOs; Part IV develops the device schedule, first in CUDA and then across other accelerator ecosystems.

### Running example used throughout

Every major section returns to one illustrative workload so estimates stay comparable. The hidden and head dimensions match Part III's running model; the 2,048-token prefill and 4,096-token decode contexts below are separate kernel benchmark shapes, not a change to its request-level workload. Treat the hardware numbers as a teaching machine class, not a product claim.

| Symbol | Meaning | Value in this part |
| --- | --- | --- |
| `D` | model hidden width | 4096 |
| `H` | query heads | 32 |
| `H_kv` | key/value heads (GQA) | 8 |
| `d` | head dimension (`D / H`) | 128 |
| Prefill | one prompt | `B = 1`, `S_q = 2048` |
| Decode | concurrent sequences | `B = 8`, each with context `S = 4096` |
| Device | illustrative datacenter GPU | ~3 TB/s HBM, high tensor-core peak |

Two regimes matter:

- **Prefill / training-like attention:** many query rows attend over keys and values. Arithmetic intensity can be high when tiles reuse well.
- **Decode:** each sequence adds one new query token and rereads a long KV cache. Bytes per useful FLOP rise sharply; launch and memory latency matter as much as peak math.

:::diagram prefill_decode_kernels|The same attention equations appear in both regimes. Prefill wants large tiled GEMM reuse. Decode wants efficient paged KV traffic and low per-token overhead.

### Prerequisites and reader contract

You need comfort with matrix multiplication, reductions, and reading short CUDA C++ kernels. You do not need prior mastery of a particular GPU generation. Architecture names change; the method does not: define ownership, storage scope, synchronization, bytes, FLOPs, and the measurement that would falsify the claim.

This part progresses from that model to production kernels. It covers execution, memory, host orchestration, GEMM, reductions, scans, histograms, softmax, normalization, attention, serving-specific kernels, profiling, and correctness. Each chapter closes with design exercises and worked solutions that connect the primitive to production workloads.

:::callout decision|The CUDA optimization loop
Establish a correct reference. Measure the real workload. Build a bytes-and-FLOPs model. Identify one limiting resource. Change the schedule. Recheck correctness. Reprofile. Stop when the remaining gap is below the value of additional complexity.
:::

:::pagebreak

### Map of this part

| Section | Role in the story |
| --- | --- |
| Execution and memory | Build the resource ledger the rest of the part uses |
| Host orchestration | Show when kernel speed fails to reach the user |
| Hierarchical GEMM | Construct the reusable engine behind linear layers |
| Reductions, scans, histograms | Obtain the collective primitives softmax and routing need |
| Softmax, norm, top-k | Compose those primitives into transformer epilogues |
| FlashAttention | Fuse score, softmax, and value reduction without quadratic HBM |
| Blackwell pipelines | Track asynchronous buffer ownership, tensor memory, and attention dependencies |
| LLM inference kernels | Specialize the schedule for decode, paging, and quantization |
| Profiling and correctness | Close the loop on the running example with evidence |
| Beyond CUDA and NVIDIA | Separate portable model semantics from target-specific execution |

## GPU Execution, Memory, and Resource Accounting

LEAD: Threads execute in warps, warps are scheduled on streaming multiprocessors, and data moves through a hierarchy whose capacity grows as latency and sharing scope increase. Performance follows from how well the schedule uses those resources.

Before optimizing any kernel in the running example, you need a model of how the GPU executes work and where data can live. The rest of this part repeatedly asks: which level holds the tile, which group synchronizes, and which resource saturates first?

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

Example status: CUDA excerpt; not compiled or benchmarked.

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

Apply this to the running example. A memory-bound decode gather over paged KV may need many resident warps because each warp has dependent long-latency loads. A prefill tensor-core mainloop can run well at lower occupancy if it has an effective asynchronous pipeline and enough independent matrix operations.

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

In the running example, a well-laid-out KV vector of width `d = 128` in FP16 is 256 bytes per token per head. A warp that loads those values as contiguous pairs can cover them with a small number of sectors. The same logical read through a badly packed page table, with lanes jumping across pages, can multiply transferred sectors without increasing useful bytes.

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

Example status: CUDA excerpt; not compiled or benchmarked.

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

This cliff appears later when FlashAttention or decode kernels grow pipeline stages or accumulator tiles: more on-chip state can raise reuse and still lower achieved bandwidth if residency collapses.

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

:::callout pitfall|Occupancy is not the objective
Raising theoretical occupancy while destroying reuse, vector width, or tensor-core issue often slows the kernel. Use occupancy as a residency constraint, then optimize the limiting stall or bandwidth metric on the real shape.
:::

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

The previous section gave a resource model inside one kernel. The running example does not live there. A decode step is a short sequence of kernels, copies, and sampling work submitted from the host or a captured graph. This section is the shell around that sequence: how work is ordered, overlapped, and kept off the global critical path.

### The asynchronous execution contract

CUDA calls usually enqueue work into streams. Operations in one stream execute in order. Independent operations in different streams may overlap if dependencies, resources, and hardware engines allow it. "May overlap" is important: concurrency is permitted, not guaranteed.

An event records progress in a stream. Another stream can wait on that event without forcing the host to synchronize the whole device. This builds a dependency graph while preserving unrelated concurrency.

Example status: CUDA excerpt; not compiled or benchmarked.

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

In the decode regime of the running example, each step may launch several short kernels: RoPE or cache append, attention, MLP GEMMs, normalization, logits, and sampling. If measured CPU dispatch and launch gaps are comparable to the short GPU work at small batch, submission becomes visible. Graph capture amortizes that overhead for stable shape buckets; there is no universal per-launch latency to assume across runtimes and hardware.

Example status: CUDA excerpt; not compiled or benchmarked.

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

For the running example, a decode graph family might cover `B in {1, 2, 4, 8}` with a small set of context-length buckets. Padding eight sequences to the next captured length is acceptable only when the saved launch latency exceeds the wasted KV and GEMM work.

:::callout decision|Capture only after the dependency graph is stable
Do not graph-capture a prototype that still allocates, synchronizes globally, or changes topology every step. First make the eager path stream-ordered and leak-free; then capture the steady-state shape.
:::

### Multi-GPU copies and peer access

Device-to-device transfers can use peer access over the available interconnect. The relevant path may be PCIe, NVLink, or another fabric, with topology-dependent bandwidth and routing. Batch small transfers, preserve alignment, and overlap communication only when independent compute exists.

For collectives, use a collective library rather than hand-rolled peer copies unless the communication pattern is genuinely unusual. The application-level question is still exposed communication on the critical path, not theoretical link bandwidth.

### End-to-end timing

CPU wall-clock timing around an asynchronous launch measures submission, not completion. CUDA events measure elapsed device time between points in a stream. End-to-end latency should use the product boundary and include queueing, copies, and synchronization.

Example status: CUDA excerpt; not compiled or benchmarked.

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

With the machine model and host shell in place, the next building block is the matrix multiply that dominates both prefill and the MLP path. In the running example, a linear layer maps activations of width `D = 4096` through a weight matrix. Prefill exposes many token rows at once; decode may expose only `B = 8` rows. The math is the same GEMM. The efficient schedule is not.

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

Example status: CUDA excerpt; not compiled or benchmarked.

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

Example status: CUDA excerpt; not compiled or benchmarked.

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

Launch this kernel with a two-dimensional block of exactly `(TILE, TILE)` threads and a grid covering M and N; `TILE * TILE` must fit the device's block limit. Out-of-range loads become zero, all participating threads reach both barriers, and the output store is guarded. An early return before a later barrier would leave shared-tile entries uninitialized for an edge block.

### Arithmetic intensity from tile reuse

Ignoring the final store, a square block tile of width `T` loads roughly `2 T^2` values per K tile and performs roughly `2 T^3` FLOPs. Its shared-tile arithmetic intensity is proportional to:

`2 T^3 FLOPs / (2 T^2 values * bytes_per_value) = T / bytes_per_value`

Larger T raises reuse, but resource consumption grows. The estimate also assumes each input tile is loaded once and does not include cache effects, epilogue traffic, or edge waste.

#### Running-example intensity sketch

Consider the decode projection that maps `M = B = 8` rows through `K = N = D = 4096` in FP16 weights.

- FLOPs are approximately `2 * 8 * 4096 * 4096 = 2.7e8`.
- Weight traffic alone is approximately `4096 * 4096 * 2 = 34 MB` if weights are read once.
- Arithmetic intensity against weights is roughly `2.7e8 / 3.4e7 = 8` FLOP/byte before activation and output traffic.

On a device whose HBM roof is near 3 TB/s, even perfect weight streaming yields only tens of microseconds of memory time, while tensor-core peak would finish the math much faster if data were free. Decode linear layers are therefore often weight-bandwidth bound at small batch. Prefill with `M = 2048` raises intensity and can move toward the compute roof for the same weights.

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

In the running example's decode GEMM with `M = 8`, ordinary output tiling may leave most of the GPU idle. Splitting K creates more independently schedulable work, at the cost of partial-result traffic. Persistence alone cannot create parallelism when too few output tiles exist; it must be paired with a work decomposition that exposes enough tasks.

Stream-K-like scheduling distributes K work more evenly across a fixed set of work units to reduce wave quantization and tail imbalance. Persistent kernels keep blocks resident and pull tiles from a global or hierarchical queue. These strategies improve utilization for awkward shapes but add coordination and deterministic-order questions.

### Batched and grouped GEMM

**Batched GEMM** executes many matrices with compatible shapes and strides. **Grouped GEMM** accepts varying shapes and offsets. Both appear in attention heads, adapters, and mixture-of-experts layers.

Small matrices may be launch- and scheduling-bound. Grouping several problems into one persistent kernel increases available work but requires load balancing. Sorting by shape can improve efficiency while adding preprocessing and reordering cost.

### Library, template framework, or custom kernel

Use cuBLAS or another vendor library for standard dense GEMM. Use a template framework such as CUTLASS when you need controlled layouts, data types, schedules, or epilogues without writing every hardware primitive. Write a custom kernel when the operator is unusual enough that the library boundary creates material traffic or launch cost.

The decision is based on end-to-end value, maintainability, architecture coverage, and test burden. Beating a library on one shape is not the same as owning a production GEMM.

:::callout pitfall|Winning one shape is not owning GEMM
A custom kernel that beats a library on `M = 8` decode can lose on prefill, on another dtype, or after an architecture upgrade. Measure the production shape mix and the maintenance boundary before replacing a library path.
:::

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

GEMM builds the large linear maps. Softmax, normalization, sampling, and expert routing still need collective operations: maxima, sums, prefix offsets, and bin counts. This section develops those primitives once, with ownership and masks stated carefully, so later kernels can compose them instead of inventing synchronization ad hoc.

### Reduction as a tree

A reduction replaces a linear dependency with a tree. Within a warp, shuffle instructions exchange register values. Across warps, one common pattern writes a partial per warp to shared memory and lets the first warp finish.

Example status: CUDA excerpt; not compiled or benchmarked.

```cuda
// Contract: all 32 lanes participate; invalid data contributes zero.
__device__ float warp_sum(float x) {
  int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffffu, x, offset);
    if (lane + offset < 32) x += other;
  }
  return x;
}

__device__ float block_sum(float x) {
  __shared__ float partial[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) partial[warp] = x;
  __syncthreads();

  int warp_count = blockDim.x >> 5;
  float value = (threadIdx.x < warp_count) ? partial[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  __syncthreads();  // protect shared partials before a subsequent call
  return value;  // valid in lane 0 of warp 0
}
```

This excerpt requires a one-dimensional block with 32 to 1,024 threads in a multiple of 32, and every block thread must call it. Threads outside the logical input contribute zero rather than exit. This explicit full-warp contract avoids treating an arbitrary sparse mask as a valid shuffle reduction tree. Only thread zero owns the final value. A caller that lets every thread store `value` is incorrect; a caller that needs the sum everywhere must broadcast it through shared memory and a block barrier.

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

Example status: CUDA excerpt; not compiled or benchmarked.

```cuda
// Contract: all 32 lanes participate; valid items form a lane prefix.
__device__ int warp_inclusive_scan(int x) {
  int lane = threadIdx.x & 31;
  for (int offset = 1; offset < 32; offset <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, offset);
    if (lane >= offset) x += y;
  }
  return x;
}
```

Unused data lanes contribute zero but still execute the scan. This is not an arbitrary-mask scan: a sparse participation pattern requires a different lane mapping or a collective with that contract. For a device-wide scan, each block scans a tile, block totals are scanned, and the resulting offsets are added to each tile. A library handles edge cases, recursion, tuning, and temporary storage.

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

These primitives reappear later when mixture-of-experts routing counts tokens per expert and scans those counts into expert-buffer offsets.

### Segmented reductions

A segmented reduction combines values within variable-length groups. Segments may be represented by offsets, keys, or boundary flags. Short segments waste blocks if assigned one per block; long segments need multi-block cooperation.

Sort or bucket by size, combine tiny segments per block, and split very long segments. The overhead of preprocessing must be amortized across repeated operations or large workloads.

### Gather, scatter, and irregular access

Gather reads `out[i] = input[index[i]]`; scatter writes to indexed destinations. Address locality, duplicate indices, and output conflicts dominate. Sort or reorder indices to improve locality when semantics allow. For scatter with duplicates, define whether updates are last-writer, atomic sum, max, or invalid.

Caching helps repeated gather indices, but random accesses can remain latency-bound. Vectorizing random accesses does not make them coalesced.

### Use production primitives deliberately

CUB provides warp-, block-, and device-wide reductions, scans, histograms, and selection. Use it unless a fused operator, unusual data type, fixed small shape, or special semantics justify custom code.

The value of implementing a primitive is understanding its invariants. The production decision still favors a maintained library when the boundary fits.

:::callout insight|State ownership before writing the loop
Before coding a reduction or scan, name which lane owns the result, which mask participates, what identity fills inactive lanes, and which barrier makes partials visible. Most collective bugs are ownership bugs dressed as arithmetic bugs.
:::

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

The collectives from the previous section become transformer epilogues here. Softmax is two reductions plus a normalize pass. LayerNorm and RMSNorm are statistics plus an affine map. Top-k and sampling sit at the end of a decode step. In the running example, these kernels rarely dominate FLOPs, but they dominate launch count, numerical edge cases, and fusion decisions around attention and logits.

### Stable softmax

For a row `x`, softmax is:

`p_i = exp(x_i - m) / sum_j exp(x_j - m)`, where `m = max_j x_j`

Subtracting the row maximum prevents overflow. A basic row kernel has three logical phases:

1. reduce to the row maximum;
2. reduce the shifted exponentials to the denominator;
3. normalize and write outputs.

Example status: CUDA excerpt; not compiled or benchmarked.

```cuda
__shared__ float row_stats[2];
float local_max = -INFINITY;
for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  local_max = max(local_max, x[row * cols + i]);
}
float max_at_thread_zero = block_max(local_max);
if (threadIdx.x == 0) row_stats[0] = max_at_thread_zero;
__syncthreads();
float row_max = row_stats[0];

float local_sum = 0.0f;
for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  local_sum += expf(x[row * cols + i] - row_max);
}
float sum_at_thread_zero = block_sum(local_sum);
if (threadIdx.x == 0) row_stats[1] = sum_at_thread_zero;
__syncthreads();
float row_sum = row_stats[1];

for (int i = threadIdx.x; i < cols; i += blockDim.x) {
  y[row * cols + i] = expf(x[row * cols + i] - row_max) / row_sum;
}
```

Here one block owns one row, `cols > 0`, inputs are finite, and the full-warp launch contract from `block_sum` applies. `block_max` is the analogous reduction with maximum and negative infinity as the identity; it also returns its result only in thread zero. Both row statistics are explicitly broadcast before other threads consume them. Masked rows need the policy below before this arithmetic runs.

The code recomputes exponentials in the final loop. Keeping every value in registers can be faster for narrow rows but creates spills for wide rows. Recompute spends special-function arithmetic to avoid HBM or local-memory traffic.

For attention in the running example, a softmax row length equals the attended context. Prefill rows can be length 2048; decode rows can be length 4096. Materializing those scores in HBM is the problem FlashAttention later removes. The reduction structure remains the same.

### Online softmax derivation

Suppose a processed prefix has maximum `m_a` and exponential sum `l_a = sum exp(x - m_a)`. A new tile has maximum `m_b` and local sum `l_b = sum exp(x - m_b)`. The combined maximum is:

`m = max(m_a, m_b)`

Rescale each partial into the new reference frame:

`l = exp(m_a - m) l_a + exp(m_b - m) l_b`

This state `(m, l)` is associative up to floating-point order. Adding an output accumulator produces the recurrence used in tiled attention.

Represent an empty or fully masked tile by zero mass and handle it as a neutral state before evaluating exponentials. Merging two empty states by blindly subtracting their negative-infinity maxima produces NaNs. The same guard is needed for attention's weighted-output state.

:::diagram online_softmax|Two tiles keep compact state instead of the full score vector. Merging rescales both sides into one maximum, then adds denominators and weighted outputs.

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

In the running example, normalization rows have length `D = 4096`. That fits a block-strided reduction comfortably. The dominant cost is usually reading and writing the activation tensor, which is why residual-plus-norm fusion matters.

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

For one residual stream of shape `[tokens, D]` in FP16, a separate residual add performs two tensor reads and one intermediate write. RMSNorm must then read that intermediate to compute statistics, read it again to apply the scale unless the row remains on chip, and write the output. The unfused lower bound is therefore five full activation-tensor transfers and the common two-pass case is six, before counting gamma. If the residual is not needed later, fusion can reduce this to two input reads and one normalized-output write. If a later layer needs the updated residual, its write remains: fusion saves the norm reread, or two rereads for the two-pass case. At prefill with `tokens = 2048` and `D = 4096`, each FP16 tensor is 16 MiB, so one eliminated transfer saves 16 MiB. At decode with `tokens = 8`, the absolute byte saving is small, but avoiding a launch can still matter. These are logical transfer counts; measured HBM savings also depend on cache reuse.

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

:::callout pitfall|Fusion that changes sampling semantics
A fused logits path that reorders penalties, truncates top-p differently, or reindexes RNG under batching can pass unit tests on toy shapes and still alter user-visible text. Preserve the declared probability contract, then fuse.
:::

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

LEAD: FlashAttention computes dense attention through a tiled memory schedule that avoids writing the quadratic score and probability matrices to HBM.

Online softmax made the merge rule available. FlashAttention is that rule applied to attention tiles so the running example's prefill path never writes `S` or `P` to HBM. Exact here means the same dense-attention operation without a sparsity or low-rank approximation; floating-point order and kernel precision can still change numerical results.

:::diagram attention|Attention combines QK scores, masking and softmax, then a weighted V reduction. The performance question is which intermediates must cross HBM.

### The naive schedule

For one head, standard attention is:

`S = Q K^T * scale`

`P = softmax(S + mask)`

`O = P V`

A naive implementation writes `S` to HBM, reads it for softmax, writes `P`, reads `P` for the V product, and writes `O`. `S` and `P` each contain `sequence^2` elements.

Even if the GEMMs are efficient, moving quadratic intermediates can dominate memory and prevent long contexts from fitting.

#### Running-example HBM contrast

For one head during prefill with `S_q = 2048` and FP16 scores:

- `S` or `P` alone is `2048^2 * 2 = 8 MiB` per head;
- across `H = 32` heads, one complete score or probability tensor is 256 MiB;
- merely storing both tensors consumes 512 MiB, and writing then rereading each produces at least 1 GiB of quadratic HBM traffic before counting Q, K, V, or output traffic;
- FlashAttention keeps per-row state `(m, l)` and a query-tile output accumulator on chip, and writes an output of size `S_q * d`. Saved attention intermediates are linear in sequence length. Total HBM traffic is not necessarily linear: different query tiles may reread K/V, with the amount governed by tile size, on-chip capacity, and cache reuse.

The arithmetic may increase because tiles are rescaled and scores are recomputed in backward. The wall-clock win comes from staying under the HBM roof.

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

For the running example's decode step, each of `B = 8` sequences contributes one query row of length `d = 128` per head and must read `S = 4096` cached keys and values. With GQA, `H = 32` query heads share `H_kv = 8` KV heads, so each KV stream is reused by four query heads. The kernel problem is no longer "avoid `S^2` materialization"; it is "stream paged KV efficiently and keep online softmax state compact."

:::callout decision|Name the attention regime before naming the kernel
If the workload is prefill or training, argue IO-aware tiling and tensor-core occupancy. If it is decode, argue KV layout, paging, vectorized loads, and launch overhead. Reusing the wrong regime's plan is a common false optimization.
:::

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

## Blackwell Pipelines and Modern Attention Kernels

LEAD: A compound GPU kernel must keep several asynchronous engines supplied while preserving every intermediate's lifetime and numerical meaning. Its speed depends on how those engines cooperate, as well as their individual throughput.

### Separate the hardware resources

The earlier tiled GEMM explains reuse. A Blackwell data-center implementation adds another question: which engine owns each stage, and how does another engine know that it has completed? The architecture-specific discussion here concerns the SM100 family, not a promise that every product carrying the Blackwell name supports the same instructions.

| Resource | What belongs there | What it does not imply |
| --- | --- | --- |
| Global memory / HBM | Persistent weights, activations, KV, outputs | A value is not on chip merely because its address is known |
| Shared memory | Cooperatively staged operand tiles | A normal thread barrier is not every asynchronous engine's completion event |
| Registers | Thread-local indices and scalar/vector work | More live values can reduce occupancy or spill |
| Tensor memory, TMEM | Architecture-defined matrix accumulators and supported intermediates | It is not a general replacement for shared memory or the KV cache |
| TMA and tensor-core engines | Asynchronous transfers and matrix operations | Issuing an operation does not make its result ready |

CUTLASS documents SM100 GEMM organization and supported instruction/layout combinations. PTX specifies the actual `tcgen05` operations, memory ordering, and completion mechanisms. A framework may call these matrix operations UMMA; use the architecture and compiler documentation when choosing legal operand layouts, synchronization, and CTA grouping. See [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) and the [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

### Derive a two-buffer ownership protocol

:::diagram double_buffer|A dependency-valid schedule for the example below: loads take 3 microseconds and consumption takes 5. Numbered tiles alternate buffers. Arrows connect transfer completion to eligible consumption; slot reuse also waits for the prior consumer's final read.

Start with a portable dependency model before writing architecture-specific instructions. Buffer zero carries even tiles; buffer one carries odd tiles. Each reuse has a generation number so completion of an old tile cannot satisfy a wait for a new tile. The valid lifetime is `FREE → FILLING → READY → IN_USE → FREE`.

| Event | Producer permission | Consumer permission |
| --- | --- | --- |
| Buffer free for generation g | Begin filling the entire required tile | None |
| Transfer complete for g | Do not overwrite the tile | Read the tile or issue its matrix operation |
| Matrix operation issued | Still cannot overwrite an asynchronously consumed operand | Wait before reading unfinished results |
| Last consumer complete | Reuse for generation g+1 | No further access to generation g |

Example status: Explanatory pseudocode; this describes dependencies, not a CUDA API implementation.

```text
producer, for each tile j:
    slot = j % 2
    generation = j // 2
    wait_until_free(slot, generation)
    start_transfer(slot, tile=j, generation=generation)
    publish_ready_after_transfer_completion(slot, generation)

consumer, for each tile j:
    slot = j % 2
    generation = j // 2
    wait_until_ready(slot, generation)
    operation = issue_matrix_work(slot)
    wait_for_last_read_of_slot(operation)
    release_slot(slot, generation + 1)
wait_for_all_result_writes_before_epilogue()
```

The real implementation may use phase bits rather than unbounded counters, and may prove an operand is released before the full output operation finishes. That optimization requires the documented lifetime rule; it cannot be inferred from host submission order. A missing completion wait can pass tests when the producer is slow and fail once transfer overlap improves. A partial final tile must also obey the transfer's expected byte count or padding protocol; a wait for bytes that will never arrive can hang forever.

Use a simple original timing model. Loading one tile takes 3 microseconds and consuming it takes 5. Four serial tiles take `4 × (3 + 5) = 32` microseconds. With independent engines, sufficient buffering, and negligible synchronization cost, the overlapped schedule takes `3 + 4 × 5 = 23` microseconds. It does not take 12: the consumer remains the bottleneck. If both operations saturate the same memory path, the independence assumption fails. Extra buffers then consume capacity without achieving the predicted overlap.

### What FlashAttention-4 changes

The [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451), released in March 2026, targets Blackwell's imbalance between matrix throughput, non-matrix work, and data movement. Its forward pipeline overlaps matrix work and softmax; its backward design uses tensor memory and cooperative CTA execution to reduce traffic and reduction overhead. It also supports a deterministic execution mode. These are changes to the schedule and numerical implementation of attention, not permission to omit attended keys.

The [authors' implementation explanation](https://tridao.me/blog/2026/flash4/) describes alternating query tiles, a separate correction stage, distributing exponential work between special-function and FMA units, conditional rescaling, and tensor-memory reuse. The important learning step is to connect each optimization to a resource: distributing exponentials relieves a specialized unit; moving an intermediate reduces shared-memory traffic; changing tile ownership changes reduction and synchronization cost. None promises the same speedup for short decode, every head dimension, or a whole service.

Conditional rescaling is easiest to understand independently of a particular kernel. Softmax can accumulate numerator and denominator relative to **any** shared reference `r`, provided intermediate exponentials remain representable:

:::equation o = Σ_{j} exp(s_{j} - r) v_{j} / Σ_{j} exp(s_{j} - r)|The common factor exp(-r) cancels; choosing a stable reference is a numerical requirement.

With scores `[0, 1]`, reference zero gives unnormalized weights `[1, e]`. Reference one gives `[1/e, 1]`. Both normalize to the same probabilities. Therefore a small increase in the observed maximum does not mathematically require immediate rescaling if the old reference is retained consistently. Changing the reference for new terms while forgetting to rescale old terms is wrong. A large positive score jump can overflow, so a real implementation needs a safe threshold, consistent numerator/denominator bookkeeping, and a final normalization. An approximate exponential adds a separate numerical error that this identity does not remove.

### Low-bit attention is a numerical experiment

Weight-only quantization, low-bit `QK` multiplication, and low-bit probability/value multiplication are separate choices. Softmax probabilities can contain many small values, and errors in the probability/value product affect a weighted sum, not just a single matrix element. Scale selection, accumulation, backward recomputation, and layer-wise drift all matter.

A September 3, 2026 [hardware-aware FP4 FlashAttention-4 preprint](https://arxiv.org/abs/2609.04105) illustrates the boundary: it reports benefits in selected kernels, while its evaluated distributed training keeps an FP8 probability/value path after tested MXFP4 alternatives diverged. This is early, workload-specific evidence, not a conclusion that FP4 attention is either universally safe or universally unusable.

An acceptance ladder is: compare against an FP32 reference on adversarial rows; check full-layer outputs and gradients; run a short matched training trajectory; then evaluate convergence and deployment metrics. Include all-masked rows, large score differences, long reductions, and non-power-of-two dimensions. A maximum error from random Gaussian inputs is not a substitute for these cases.

### Choose the implementation boundary

Use a library when its supported operation matches the contract. Use a kernel language or template system when fusion, a layout, or a workload specialization creates a measurable opportunity. Use hand-written low-level instructions only when their extra control addresses an observed limit and the team can maintain synchronization and architecture-specific tests. [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) and Triton change how schedules are expressed; they do not remove tile lifetimes, register pressure, or numerical obligations.

Compare three baselines explicitly: a readable mathematical reference for semantics, a production implementation for performance, and the current deployed end-to-end path for value. A scalar CUDA loop is an excellent correctness starting point and usually an inappropriate headline performance denominator.

### Exercises and worked answers

1. **When can a staging buffer be reused?** After its last asynchronous reader completes, not merely after the instruction is issued or producer threads synchronize.
2. **What happens if loading becomes 7 microseconds and consumption remains 5?** The ideal four-tile pipeline takes `4 × 7 + 5 = 33` microseconds; transfer now controls steady state.
3. **May a softmax kernel skip every rescale?** Only under a proven representability bound with consistent reference bookkeeping. Arbitrary scores defeat that bound.
4. **Why can faster matrix instructions fail to improve attention?** Exponentials, layout conversion, shared-memory traffic, synchronization, or insufficient parallel work can control the critical path.
5. **Does a forward FP4 benchmark justify training deployment?** No. Backward behavior, accumulated numerical drift, convergence, and matched quality remain untested by that result.

## CUDA Kernels for LLM Inference

LEAD: LLM serving kernels operate on dynamic batches, ragged state, low precision, and strict token cadence. Layout and scheduler contracts are part of kernel design.

Prefill attention and GEMM covered the compute-rich path. This section specializes the schedule for decode and related serving kernels. The serving vocabulary is restated here so the section stands alone: what KV cache is, why it is paged, how GQA shares keys and values, and which contracts a kernel must honor.

### Serving contracts restated for kernel work

For the running full-history GQA model, decode retains each layer's keys and values for tokens processed so far, including the prompt. That working set is the **KV cache**. Its lifetime follows requests, not training batches. The element payload is `2 * tokens * layers * KV_heads * head_dimension * bytes_per_element`, summed over active sequences; the factor two counts K and V, and physical allocation adds overhead.

Other architectures need a different state ledger. Part I's hybrid example carries recurrent matrices and convolution history alongside attention KV; a port, prefix restore, or handoff must preserve them at the same accepted token boundary.

A **paged KV cache** stores tokens in fixed-size physical pages and maintains a block table from logical token blocks to physical pages. Kernels must translate `(sequence, position)` into a page and offset, then load K/V vectors. The page size trades fragmentation against contiguous access and metadata overhead.

**Grouped-query attention (GQA)** and **multi-query attention (MQA)** reduce KV heads relative to query heads. In the running example, `H = 32` and `H_kv = 8`, so four query heads read one KV head. The kernel must map those query heads to reuse loaded K/V without silently mixing sequences or heads.

**Token cadence** is the product constraint that decode steps are short and frequent. A kernel that wins a microbenchmark but adds host synchronization, allocator traffic, or an extra full-tensor pass can miss inter-token latency even when its FLOP rate looks healthy.

### Embedding lookup and output projection

Embedding lookup is a gather. Adjacent output dimensions for one token are contiguous; token IDs across requests are irregular. Assign lanes across the embedding dimension so one token's vector loads coalesce. Repeated token IDs may benefit from cache, but do not rely on it for the worst case.

The output projection is a GEMM against the vocabulary matrix. For small decode batch it can be bandwidth-bound. Tensor parallelism shards vocabulary or hidden dimension, after which top-k and sampling need distributed coordination.

### RoPE

Rotary position embedding rotates pairs of query and key components. For pair `(x_0, x_1)` and angle `theta`:

`y_0 = x_0 cos(theta) - x_1 sin(theta)`

`y_1 = x_0 sin(theta) + x_1 cos(theta)`

The kernel can fuse RoPE into Q/K projection output handling or KV-cache write. Precomputed sine/cosine tables trade memory reads for transcendental arithmetic. Generate angles on the fly when table bandwidth or very long positions make it worthwhile.

Layout must define whether paired dimensions are adjacent or split into halves. A mismatch can produce numerically plausible but incorrect results.

Example status: CUDA excerpt; not compiled or benchmarked.

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

For the running example, one decode step appends, per layer, `B * H_kv` vectors of length `d`. In FP16 that is `8 * 8 * 128 * 2 = 16 KiB` of K plus the same for V - tiny compared with reading the full context, but correctness-critical because every later attention step depends on the written layout.

### Paged KV attention

A block table maps logical token blocks to physical pages. Decode attention traverses pages, loads K/V vectors, computes query-key scores, performs online softmax, and accumulates values. This is the same `(m, l, o)` recurrence derived earlier; only the K/V addressing and parallelism differ from FlashAttention prefill.

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

#### Bandwidth sketch for the running example

For one layer of a decode attention step, an ideal schedule that reuses each KV head across its query heads reads roughly:

`B * H_kv * S * d * bytes_per_element * 2 (K and V)`

With `B = 8`, `H_kv = 8`, `S = 4096`, `d = 128`, FP16:

`8 * 8 * 4096 * 128 * 2 * 2 = 134,217,728` bytes = 128 MiB

At an illustrative 3 TB/s HBM ceiling, streaming 128 MiB takes about 0.045 ms before scoring arithmetic, softmax, and writes. This is a per-layer lower bound assuming the KV reads reach HBM. Across Part III's 32 layers, the logical KV traffic is 4 GiB (about 4.3 GB), matching that part's decode ledger. Repeated loads across GQA query groups can raise traffic, while cache hits can reduce it. Useful bandwidth falls if page jumps destroy coalescing. This is why layout and paging dominate decode attention more than peak tensor-core throughput.

### Split-K decode attention

One sequence with long context may not expose enough parallel work if one block owns the entire KV range. Partition context across blocks. Each block emits partial `(m, l, o)` state; a second phase merges states with the online-softmax recurrence.

This is analogous to split-K GEMM: more parallelism in exchange for a partial-result reduction. It helps long contexts and small batches, then loses when batch already supplies enough blocks.

### GQA and MQA mapping

With grouped-query attention, several query heads read one KV head. Map those query heads close enough to reuse K/V cache lines or shared tiles. Excessive sharing can increase live output accumulators. The kernel chooses between reloading K/V for simpler ownership and retaining K/V while processing multiple query heads.

### KV-cache quantization

Quantized K/V reduces stored bytes and potential read traffic, leaving room for more context or requests in a fixed memory budget. The kernel reads packed values plus scales, dequantizes into compute registers, and accumulates in higher precision. Scale granularity may be per tensor, head, channel, page, or token group.

Finer scales improve fidelity and increase metadata traffic. Dynamic token scales add append-time reduction. K and V can have different sensitivity and therefore different formats.

If the running example quantizes KV to 4 bits with modest scale metadata, theoretical KV read volume drops by roughly 4x before considering dequant instructions and scale loads. The optimization wins only when the unpack path still feeds the score pipeline faster than the FP16 baseline.

### Weight-only quantized GEMM

For low-batch decode, weight bandwidth dominates many linear layers. A weight-only kernel loads packed weights, dequantizes them, and multiplies by higher-precision activations. The schedule must overlap unpack/dequant with MMA and load scales efficiently.

Compression is useful only if the hardware path processes packed data efficiently. A format can reduce stored bytes and still run slower because unpacking, irregular scales, or unsupported tensor-core instructions dominate.

### MoE routing and grouped GEMM

Mixture-of-experts layers activate a subset of experts per token. Routing computes top experts, counts tokens per expert, scans counts into offsets, scatters tokens into expert-contiguous buffers, executes grouped GEMM, then scatters outputs back with routing weights.

The pipeline combines top-k, histogram, scan, gather/scatter, and GEMM - the primitives from earlier sections. Fusing every phase is rarely ideal. The important interfaces are compact routing metadata and layouts that let grouped GEMM consume contiguous expert batches.

Hot experts create imbalance. Capacity limits, token dropping, or expert replication change semantics and belong to the model contract, not only the kernel.

### Sampling and token cadence

The decode iteration often ends with logits processing and sampling. At small batch, several tiny kernels can add visible launch latency. Graph capture and carefully scoped fusion help. Preserve request cancellation, dynamic sampling policies, and RNG mapping.

:::callout insight|Decode kernels inherit the cache contract
Append, attention, and quantization kernels share one address and dtype contract for KV pages. Document that contract once. Most silent attention corruptions are layout mismatches, not softmax algebra mistakes.
:::

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

The preceding sections built a schedule for the running example's layer and decode step. This final section is how you defend that schedule: benchmarks that match the claim, profiler hypotheses tied to resources, and correctness layers that catch the shapes serving actually produces.

:::diagram roofline|This schematic illustrates a bandwidth slope and compute ceiling, not measured prefill or decode performance. A measured hierarchical roofline can distinguish HBM, cache, and compute limits under a specified workload.

### Build a trustworthy benchmark

Specify:

- device, architecture, clocks or power mode, toolkit, driver, and compiler flags;
- input shapes and their production frequency;
- dtype, layout, alignment, sparsity, and value distribution;
- warmup, iteration count, synchronization boundary, and cache policy;
- reference implementation and numerical tolerance;
- whether allocation, transfer, launch, and framework dispatch are included.

For this part's running example, a minimum matrix includes prefill `[B=1, S=2048]` attention and GEMM shapes, decode `[B=8, S=4096]` paged attention, and the corresponding linear layers at `D=4096`. Random inputs can hide value-dependent behavior such as histogram contention, sparsity, overflow, or early exit. Include adversarial and production-derived distributions.

### Cold, warm, and steady-state claims

Cold latency may include CUDA context initialization, module load, JIT compilation, graph instantiation, allocation, and cache miss. Warm latency may assume all of these are complete. Both are legitimate if labeled.

For interactive systems, report p50 and tail distributions over the real request mix. Median kernel time alone does not predict queueing or graph-cache miss.

### Roofline analysis

Arithmetic intensity is useful operations divided by bytes moved across a chosen boundary. The simple roofline bound is:

`attainable_work_rate <= min(compute_peak, bandwidth * arithmetic_intensity)`

Choose the correct work units and peak. Tensor-core FLOPs, scalar FP32 FLOPs, integer operations, and special functions have different ceilings. Choose the correct bytes: requested HBM bytes, measured HBM traffic, or cache-level traffic.

A hierarchical roofline adds L1 and L2 ceilings. A kernel may sit below the HBM roof because it is actually limited by L1 bandwidth, shared-memory conflicts, instruction issue, or dependency latency.

Place the running example on that roof deliberately: decode linear layers and paged attention belong near the bandwidth side; large prefill GEMMs and well-tiled FlashAttention can approach the compute side. If a decode kernel claims compute-bound behavior, demand the intensity calculation.

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

Serving kernels add cases that dense unit tests miss: partially filled final pages, GQA head mapping, sequence packing boundaries, and graph replay with reused device addresses.

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

:::callout decision|Close on the claim you opened with
If the chapter's running example was decode at `B=8`, `S=4096`, the acceptance evidence must include that shape, not only a large square GEMM where every kernel looks good. Optimize the workload you promised to serve.
:::

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

Keep three artifacts with the optimized kernel: a semantic reference and edge-case tests, a profile explaining the resource limit, and an engine replay showing the integration effect. These establish different things; passing tests is not a proof over every possible input. For the running example, check that prefill still avoids quadratic score storage and that fused decode preserves KV indexing, masking, and cancellation behavior.

These artifacts also provide the starting point for a port. The next chapter asks which parts of this schedule survive a change of compiler or accelerator, and which must be rebuilt.

## Accelerator Ecosystems Beyond CUDA and NVIDIA

LEAD: Moving a model is easier than moving its performance. The weights and attention equations may survive unchanged while kernels, memory layouts, compilation, collectives, and serving behavior all need new evidence.

The documentation assistant now needs a second deployment option. Perhaps NVIDIA capacity is scarce, the team already operates an AMD cluster, or a cloud accelerator offers a promising deployment path. The task is not to translate every CUDA kernel immediately. It is to find the smallest supported stack that can serve the same model correctly, then determine whether its latency, operating cost, and maintenance burden justify adoption.

This chapter assumes the prefill/decode and memory models developed earlier in this part. It separates three often-confused choices: the accelerator, the software stack that operates it, and the language used to write a kernel. The ecosystem notes were checked against primary documentation on September 13, 2026. They are entry points for an evaluation, not a permanent hardware or model support matrix.

### First separate the layers

CUDA names more than a kernel language: an application may depend on its runtime, libraries, compiler, graph execution, and development tools. Replacing CUDA C++ with Triton can change how kernels are written while leaving the application on NVIDIA hardware. Moving to AMD changes the underlying stack as well. Moving to TPU or Trainium changes the device execution model enough that a literal translation of warp-level code is usually the wrong starting point.

:::diagram accelerator_portability|Keep the model contract above the porting boundary. Below it, choose a supported implementation path and revalidate kernels, state layout, and distributed execution. These are representative paths, not an exhaustive compatibility matrix; Pallas also has GPU backends.

There are three separate acceptance tests. **Source portability** asks how much code can be reused. **Semantic portability** asks whether it still implements the required computation and state transitions. **Performance portability** asks whether the target meets the workload's service objective. A successful import or compilation establishes none of the latter two by itself.

We will follow that boundary in three stages: port an operator, place the model's state and compiled variants, then judge a deployment against the assistant's workload. Triton supplies the worked tiled kernel; HIP exposes block synchronization; TPU and NKI illustrate different tile schedules. At each stop, ask who owns the row, where working values live, how tails remain correct, and what evidence establishes performance.

The [SYCL standard](https://www.khronos.org/sycl/) offers another heterogeneous C++ programming model, including implementations for Intel GPUs. Its libraries and serving integrations still depend on the selected implementation; this chapter concentrates on the four paths in the diagram.

### Carry one operation across the boundary

Use RMSNorm as a small porting exercise before moving the whole assistant. For an input row `x` of actual width `N`, learned weights `w`, and positive epsilon, the operation is `y[i] = x[i] * rsqrt(sum(x[j]^2) / N + epsilon) * w[i]`. Unlike LayerNorm, it does not subtract a mean. The contract is short; the implementation decisions are not. Who owns a row? Where does its partial sum live? Which participants synchronize? What happens to a padded lane?

For the examples here, input rows and weights are separate read-only tensors; the output is newly allocated. Convert loaded values to FP32 before squaring and accumulating, then convert the final result to the output storage dtype.

:::diagram rmsnorm_port|Padding changes the tile, not the mathematical row width.

For `x = [3, 4, 0]`, unit weights, and epsilon `1e-6`, the squared sum is 25. Using the actual width `N=3` in the mean-square term produces approximately `[1.0392304, 1.3856406, 0]`. Mistakenly using the padded width four produces approximately `[1.1999999, 1.5999999, 0]`. Both look plausible. A width-4096-only test would never expose this bug. The repository's `examples/accelerator_portability.py` provides a higher-precision CPU oracle and rejection tests for out-of-range arithmetic.

Keep two references when validating a real kernel: higher-precision arithmetic for the mathematics, and a framework expression with the chosen accumulation and output dtypes. Set tolerances for a bounded activation range: finite FP32 inputs can still overflow when squared, and a very small positive epsilon can round to zero. Reduction trees and reciprocal-square-root implementations need not agree bitwise, but an overly loose tolerance can conceal a systematic error. Follow operator tests with model-quality checks; a training port also needs backward and gradient tests.

### Triton: change the kernel language, not necessarily the GPU

Triton lets a programmer express operations over tiles of tensor elements instead of spelling out every thread's scalar work. Its upstream project lists NVIDIA and AMD GPU support. It is distinct from the similarly named NVIDIA Triton Inference Server. A Triton kernel targeting NVIDIA still needs the relevant NVIDIA execution stack; selecting an AMD backend does not port surrounding CUDA libraries or extensions. See the [Triton project and compatibility notes](https://github.com/triton-lang/triton).

Consider RMSNorm over a row of width 4096. A tiled implementation loads the row, accumulates its squared values, computes the normalization factor, applies the learned scale, and stores the result. The useful algorithm survives a backend change. The best tile layout, number of participating warps, and reduction schedule may not. Test awkward widths and noncontiguous inputs as well as the common case; padding lanes must not contribute to the reduction.

For a first port, use the framework's supported operator. Write or retune a Triton kernel only when profiling identifies a meaningful gap. A portable source file with two backend-specific tuning configurations can be a better maintenance choice than either two unrelated implementations or one universally mediocre configuration.

Here is the body of the accompanying forward-only implementation. A program handles one row; `BLOCK` is the next power of two at least as large as `N`. Input strides are measured in elements. Output rows are contiguous, so their stride is `N`, not the input row stride. Promoting indices before stride multiplication avoids an accidental 32-bit offset limitation.

Example status: Source-only Triton kernel excerpt; no device run recorded.

```python
@triton.jit
def rmsnorm_rows(X, W, Y, sx0, sx1, sw,
                 N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    valid = col < N
    x = tl.load(X + row*sx0 + col*sx1,
                mask=valid, other=0).to(tl.float32)
    w = tl.load(W + col*sw, mask=valid, other=0).to(tl.float32)
    square_sum = tl.sum(x*x, axis=0)
    y = x * tl.rsqrt(square_sum/N + EPS) * w
    tl.store(Y + row*N + col, y, mask=valid)
```

The host wrapper is part of the operation. This inference-only example accepts matched input types, positive strides, width at most 8192, and epsilon in `1e-12..1e-2`. It allocates contiguous output on the input device and skips empty launches. Its four-warp configuration needs target-specific tuning. The complete wrapper and optional target tests are in `examples/accelerators/triton_rmsnorm.py`; the bounded-value contract avoids a synchronizing scan on every call.

Test a strided view, not just a contiguous matrix: if storage has shape `(M, 2*N+3)` and the view is `storage[:, 1:1+2*N:2]`, the row stride is `2*N+3`, the column stride is 2, and the new output's row stride is `N`. Sweep unit widths, either side of powers of two, and the implementation limit. Moving from 4096 to 4097 doubles this kernel's logical tile size; that can change register pressure and occupancy. Correct indexing and coalesced access are separate goals.

The [Triton softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) is a useful reference for masked power-of-two tiles and strides. The [LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) explains row reductions and launch choices; its operation includes centering and is not this RMSNorm kernel. Read them alongside the original example rather than assuming a tutorial's benchmark transfers to this workload.

### AMD GPUs: ROCm, HIP, and the code that translation misses

AMD Instinct GPUs are an alternative hardware family for datacenter training and inference. ROCm supplies the surrounding software; HIP offers a CUDA-like C++ interface, and HIPIFY tools help translate supported CUDA constructs. The hard cases include architecture-specific instructions, inline PTX, library dependencies, and assumptions about execution-group width. Treat AMD's [CUDA-to-HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) as a migration inventory, not a claim of drop-in binary compatibility.

The reduction kernels earlier in this part explain why names alone are insufficient. A hard-coded 32-lane mask, a warp shuffle, or a shared-memory layout may encode an NVIDIA-specific assumption. Audit the target's wavefront behavior and compiler semantics; do not simply replace every 32 with 64. Rewrite ownership and synchronization first, then retune register use, local storage, and matrix instructions. CUDA graphs and asynchronous pipelines likewise require validation through the chosen backend rather than a name-for-name substitution.

For multi-device work, RCCL supplies AMD-oriented collective operations, including reductions, gathers, and all-to-all communication. API familiarity does not imply the same collective schedule or latency on a different topology. Measure representative payloads on the actual node and network. The [RCCL overview](https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html) describes its communication scope.

Start with a supported framework and serving-engine build, then enumerate custom extensions, attention backends, quantization formats, and MoE kernels. Pin the GPU, operating system, driver, ROCm release, framework, and engine together. The [ROCm compatibility matrix](https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html) is the starting point; an AMD product name alone is not a support guarantee.

Suppose the assistant needs separate prefill and decode pools. Local attention support is only the first gate. In the official vLLM [`ROCM_ATTN` implementation](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/) inspected on September 13, 2026, `supports_kv_connector()` returns false: that backend's KV layout is incompatible with the blocks-first connector layout. This is a restriction of the inspected backend, not every ROCm deployment. Record the installed build, actual attention dispatch, head geometry, cache layout, and required connector together. If the connector gate fails, keep those phases on one compatible path or select a separately supported backend.

#### A HIP reduction makes ownership explicit

A simple HIP baseline uses exactly one 256-thread block per nonempty row, valid FP32 pointers, and positive element strides. Each thread accumulates columns `tid, tid+256, ...`; shared memory combines the partial sums, independently of warp/wave width. The excerpt is the reduction portion: a complete kernel also loads weights and stores normalized values in a second loop.

Example status: Source-only HIP kernel excerpt; no compile or device run recorded.

```cpp
const int64_t row = static_cast<int64_t>(blockIdx.x);
const unsigned tid = threadIdx.x;
__shared__ float sums[256];
float local = 0.0f;
for (int64_t c = tid; c < n; c += 256) {
    float v = x[row * sx0 + c * sx1];
    local += v * v;
}
sums[tid] = local;
__syncthreads();
for (unsigned step = 128; step != 0; step >>= 1) {
    if (tid < step) sums[tid] += sums[tid + step];
    __syncthreads();
}
float inv_rms = rsqrtf(sums[0] / static_cast<float>(n) + eps);
```

For a three-element row, 253 threads contribute zero, but all 256 still reach every block barrier. Adding `if (tid >= n) return;` would break that synchronization contract. This baseline rereads inputs for the output pass, unlike the source-level single-load Triton version; similar arithmetic is not necessarily similar memory traffic. A supported block-reduction primitive may be a better production choice after profiling.

The launch boundary matters too. Check allocation/copy calls, check the launch error, and synchronize at a test boundary to detect asynchronous execution errors. Do not free or reuse input, output, or transfer buffers until their consumers finish. A producer on another stream needs an explicit dependency; residing on the same device does not order it. See the [HIP asynchronous execution guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html). Do not put a global synchronization after every production operation merely because it simplifies the test.

The launch below assumes checked buffers, positive dimensions, and a live stream; `HIP_CHECK` stops the test on an error.

Example status: Source-only HIP launch excerpt; no compile or device run recorded.

```cpp
hipLaunchKernelGGL(rmsnorm_f32, dim3(rows), dim3(256), 0, stream,
                  d_x, d_w, d_y, n, sx0, sx1, sw, eps);
HIP_CHECK(hipGetLastError());
HIP_CHECK(hipStreamSynchronize(stream));  // correctness-test boundary
```

If another stream produces `d_x`, record an event after that production and make this stream wait on it before the launch. Synchronizing the consumer after launch does not retroactively establish the missing producer dependency. The optional `examples/accelerators/hip_rmsnorm.cpp` companion supplies the kernel, output pass, allocation, and test boundary; its compile/run status is recorded separately from CPU tests.

Some familiar names intentionally remain: ROCm PyTorch uses the `torch.cuda` interface and the `cuda` device type. Replacing them with an invented `hip` device string is not the port. Conversely, low-level masks need an actual audit: HIP documents unsigned 64-bit warp masks even for 32-lane targets. Consult [PyTorch HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html) and the release-matched [HIP language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html), rather than making a global textual substitution.

#### Follow the time before collecting counters

Start with an unprofiled correctness run and warm timings, then capture host calls, dispatches, and copies. On current ROCm, the ROCprofiler-SDK workflow uses `rocprofv3`; older `rocprof`/`rocprofv2` tooling is marked deprecated. The [SDK quick reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html) gives current trace and counter commands.

Example status: Source-only rocprofv3 recipe; not run here.

```sh
rocprofv3 --version
rocprofv3 --hip-trace --kernel-trace \
  --memory-copy-trace -- ./rmsnorm_bench
rocprofv3-avail list --pmc
```

First ask whether time disappears into host gaps, transfers, or the kernel. Only then select supported counters for that kernel. Record dimensions, strides, dtype, architecture, compiler/runtime versions, warm-up, repetition policy, and clock/power conditions. Profiled timings can include instrumentation overhead; keep the unprofiled baseline.

### Google TPU: compile the graph and design the tile schedule

A TPU is not a GPU with different branding. Its TensorCores combine matrix-multiply, vector, and scalar resources; the precise organization varies by generation. The matrix units favor well-filled matrix work, but attention also needs reductions, masking, and memory movement. Those operations remain part of the latency budget. Google's [TPU architecture guide](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) describes the hardware model.

XLA compiles framework computations for the target, handling transformations such as fusion and buffer planning. Pallas provides explicit custom-kernel control within JAX when automatic compilation does not produce the desired schedule. Pallas supports TPU and GPU work but exposes hardware-specific APIs; common notation does not mean an optimized GPU kernel can be carried over unchanged. See [XLA architecture](https://openxla.org/xla/architecture) and the [Pallas guide](https://docs.jax.dev/en/latest/pallas/).

TPUs can serve autoregressive models as well as train them. Google's current inference documentation uses the `tpu-inference` plugin for vLLM, with JAX and PyTorch model paths. Start from that integration's supported model and feature set, not an assumption that every CUDA engine flag is meaningful on TPU. See [Run inference on Cloud TPU](https://docs.cloud.google.com/tpu/docs/tpu-inference).

For the assistant, prefill presents many query rows and opportunities for matrix reuse. Decode exposes weight and KV reads, small operations, scheduling, and collective latency. The accelerator's matrix peak alone cannot predict either request-level TTFT or token cadence. Ragged lengths, paged KV, and dynamic arrivals require a serving implementation that manages those states efficiently; they do not make TPU serving impossible. Compilation buckets, attention layouts, and host/device boundaries become concrete review items.

#### Choose a Pallas window and count its live values

Begin with a dtype-matched JAX RMSNorm reference, then choose a window for the custom kernel. Suppose the input is `(128, 4096)`, weights are presented as `(1, 4096)`, and each program handles eight complete rows. The input and output windows advance along rows; every program sees the same weight window.

Example status: Source-only Pallas window specification; kernel body and launch omitted.

```python
from jax.experimental import pallas as pl

rows, width, rows_per_program = 128, 4096, 8
grid = (rows // rows_per_program,)
x_window = pl.BlockSpec((8, width), lambda p: (p, 0))
w_window = pl.BlockSpec((1, width), lambda p: (0, 0))
y_window = pl.BlockSpec((8, width), lambda p: (p, 0))
```

Under blocked indexing, program `p=3` selects input rows 24 through 31: the index map returns block coordinates, which are multiplied by the block shape. The 16 programs cover all 128 rows. These windows satisfy the documented two-dimensional TPU block-size rules; that is only one part of a valid kernel. The [Grids and BlockSpecs guide](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html) also describes full-dimension exceptions and out-of-bounds padding. Such padding has unspecified values; explicitly mask it when a reduction could read it.

The FP32 input window alone contains `8 * 4096 * 4 = 128 KiB`. An equally sized output adds 128 KiB of logical values, and a single FP32 weight row adds 16 KiB. Temporaries, layout padding, and extra pipeline buffers are additional costs; live ranges determine how much must coexist. This byte ledger narrows the design, but the target compiler and memory report decide whether the schedule fits VMEM.

For a custom TPU kernel, the next question is how HBM rows and weights become on-chip working tiles. TPU Pallas exposes vector memory (VMEM), scalar memory (SMEM), and synchronization mechanisms for staged work. A `Ref` is a mutable memory view; its name alone does not identify its memory space. The [TPU pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) shows how block windows and pipelines arrange transfers. The [matrix-multiplication guide](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) is a concrete example of a grid, an accumulating dimension, and reuse.

Eight complete rows keep each normalization local to a program. Splitting one row across two programs changes the dependency: their partial squared sums must be combined before either fragment can normalize. The implementation must then retain or reload the row values and use a supported multi-stage schedule. The global denominator alone cannot repair a fragment-local numerator.

:::diagram rmsnorm_fragments|Fragmenting a row creates a full-row reduction dependency. The arrows describe required information, not a universally available cross-program barrier.

Pallas offers `pallas_call(..., interpret=True)` for CPU-accessible semantic debugging. It lowers the grid/body into JAX execution; it does not run TPU instructions or establish VMEM fit, legal TPU layouts, DMA overlap, or speed. A separate TPU-specific interpreter models more of those mechanisms, but remains a simulator. Read the [Pallas call contract](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html) before interpreting a passing test as hardware support. Pallas is experimental; pin JAX and jaxlib together and consult its [changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html) when adapting older examples.

### AWS accelerators: Neuron is the stack, not the chip

Trainium and Inferentia are AWS accelerator families. Inferentia targets inference; Trainium supports training and inference. Neuron is their software stack, including compiler, runtime, framework integrations, libraries, and profiling tools. Native PyTorch through TorchNeuron is a closed-beta path in the checked SDK 2.32 documentation; PyTorch NeuronX and JAX are distinct integrations. Confirm access as well as feature support. See the [Neuron SDK overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) and [native PyTorch access note](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/frameworks/torch/pytorch-native-overview.html).

For a new serving evaluation, begin with the vLLM Neuron plugin's supported models and targets. The checked [announcement](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html) describes a beta path for Trn2 and Trn3 that no longer depends on NxD Inference; [NxD entered maintenance mode](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nxd-inference.html) with SDK 2.32. The [migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html) separates a supported model move from custom-modeling work. Check the assistant's attention, quantization, sampling, and cache features against that path, including target generation: support in the umbrella SDK does not establish plugin support for older Inferentia hardware.

#### NKI: follow the tile between memory spaces

NKI, the Neuron Kernel Interface, exposes tiled custom computation and explicit data movement. HBM holds large tensors, SBUF holds on-chip working tiles, and PSUM provides distinct accumulation storage for applicable matrix results. Partition dimensions, tile constraints, and lifetimes determine the schedule. The [memory hierarchy guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/memory-hierarchy-overview.html) is the conceptual starting point.

The checked SDK 2.32 pairs with NKI 0.6 and uses the `nki`, `nki.language`, and `nki.isa` namespaces. Use the [migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html) when adapting older examples, especially for returned-buffer and runtime-loop rules.

Use this schedule to read the maintained RMSNorm implementation. It is intentionally pseudocode: supported tile shapes, reduction operations, and dtype conversions must come from the selected SDK and target.

Example status: Explanatory pseudocode; NKI tile schedule.

```text
For a bounded tile containing complete rows:
    load input and weight tiles from HBM into SBUF
    reduce squared values across each actual row width
    compute the inverse RMS and scale the working tile
    store the output into shared HBM
If a row spans tiles, combine its partial sums before normalizing it.
```

SBUF is software-managed working storage, not merely a hardware cache. PSUM is distinct accumulation storage used for matrix results; it should not be drawn as an obligatory stop for every scalar reduction. Returning an output tensor also has a buffer contract: modern NKI uses shared HBM for returned tensors. Read the [memory hierarchy guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/memory-hierarchy-overview.html) with the SDK 2.32 [`nki.language.rms_norm` reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.rms_norm.html). That API's documented targets and input contract are narrower than a universal framework operator; an old tutorial with a similar name may use a different namespace or SDK.

Make the operator arguments concrete. For an FP32 `(128, 512)` input tile already in SBUF, dimension zero identifies the 128 partitions and dimension one contains each row's 512 values. Present the learned scale as a same-shaped FP32 weight tile repeating the row weights. This explicit layout is easy to read; reusing or broadcasting a smaller weight tile is a separate implementation choice.

Example status: Source-only NKI 0.6 API fragment; no simulator or device run recorded.

```python
import nki.language as nl

y_sbuf = nl.rms_norm(x_sbuf, w_sbuf, axis=1, n=512,
                    epsilon=1e-6, compute_dtype=nl.float32,
                    dtype=nl.float32)
```

Here `axis=1` selects the reduction dimension and `n=512` specifies its actual width. Passing `n=4096` for one 512-element fragment of a wider row would still omit the other fragments' squared values. The experimental API's [official source reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_modules/nki/language.html#rms_norm) documents the tile operation; the surrounding implementation must supply loads, a legal target schedule, and the shared-HBM output store.

The [NKI CPU simulator](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html), invoked through `nki.simulate(kernel)`, is a functional/debug aid. Sequential host simulation cannot establish instruction scheduling, engine overlap, latency, or bandwidth. After it passes, compile for the named accelerator generation, compare device outputs, and inspect a device trace with [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/how-to-profile-workload.html). These are separate gates. A simulator result should never appear in a chart labeled Trainium performance.

| Porting question | Triton / HIP on GPU | Pallas on TPU | NKI on Neuron |
| --- | --- | --- | --- |
| Who owns a row? | Program tile or block | Grid program and block window | Program/tile schedule |
| Where does working state live? | Compiler-managed registers; explicit shared storage in HIP | VMEM/SMEM selected by the TPU schedule | SBUF; PSUM for applicable accumulation |
| What must a tail preserve? | Masked loads, actual divisor, valid stores | Legal bounded windows and full reduction semantics | Legal tile/partition shapes and full reduction semantics |
| What establishes speed? | Warm device execution and engine replay | TPU execution/profile and engine replay | Neuron execution/profile and engine replay |

This is a comparison of questions to answer, not equivalence between memory spaces or a hardware support matrix. The common mathematical reference stays above the implementation boundary.

### Port the running model before comparing devices

Keep the assistant's checkpoint, tokenizer, GQA geometry, and quality evaluation fixed. With 32 layers, eight KV heads, head dimension 128, and two-byte cached elements, one token's K and V require `2 * 32 * 8 * 128 * 2 = 131,072` bytes across the model. Eight sequences at context 4096 therefore occupy 4 GiB of logical KV state. This is independent of vendor; physical allocation can be larger because of padding, replication, alignment, and reserved pools.

The earlier model's roughly 14 GB of weights and 4.3 GB of logical KV reads give a useful first decode ledger under the stated full-read assumptions. Replace the illustrative GPU bandwidth with the target's measured sustainable bandwidth; do not reuse the earlier 3 TB/s number as a TPU, AMD, or Trainium specification. Likewise, aggregate memory across devices is not freely pooled memory. Weight shards and KV shards must fit where the executable places them.

Before applying the 4 GiB ledger, ask which precision changed: the weight artifact's packing and scales, the compute dtype, or KV storage and its scale layout. In the checked [vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html#kv-cache-fp8-quantization), FP8 KV is dequantized to BF16 for attention; weight loading is a separate path with its own checkpoint and target restrictions.

Changing weights alone leaves this assistant's two-byte KV payload at 4 GiB. A supported one-byte KV format stores 2 GiB of elements before scales, padding, replication, and reservations. Measure the full allocation and latency. Treat a requantized checkpoint as a new artifact, preserve the original baseline, and rerun the quality gate.

#### More ranks can replicate state

Assume uniform full-history GQA, no pipeline partition, and tensor parallelism that partitions whole KV heads when possible. With eight KV heads and TP=4, each rank owns two heads: the 4 GiB logical cache becomes 1 GiB per rank. At TP=16, suppose the implementation instead replicates each KV head across two ranks. Each rank holds 512 MiB, but aggregate physical KV becomes 8 GiB. Lower per-rank memory and greater aggregate memory are compatible outcomes.

Now combine TP=4 with an abstract independent context axis, CP=2: the groups partition heads and token positions independently. Each of eight ranks stores two heads for half the history, giving 512 MiB per rank and 4 GiB aggregate. This capacity model is neither an engine flag nor a schedule. Part V distinguishes phase-specific prefill PCP from decode DCP. Uneven context partitions add padding; pipeline partitions, sliding windows, latent caches, and other replication policies require different arithmetic.

:::diagram kv_port_placement|The same 4 GiB logical cache has different physical layouts. Counts describe ranks, not interchangeable devices. The example isolates KV state; weights, activations, workspace, and reserved pools are additional allocations.

Example status: Runnable Python; self-contained KV arithmetic for the three stated layouts.

```python
layers, batch, context = 32, 8, 4096
kv_heads, head_dim, element_bytes = 8, 128, 2
logical_kv = (2 * layers * batch * context
              * kv_heads * head_dim * element_bytes)
results = []
for tp, cp in [(4, 1), (16, 1), (4, 2)]:
    # Whole heads partition; beyond 8 TP ranks, each head is replicated.
    heads_per_rank = max(1, kv_heads // tp)
    tokens_per_rank = (context + cp - 1) // cp
    local_kv = (2 * layers * batch * tokens_per_rank
                * heads_per_rank * head_dim * element_bytes)
    modeled_total = local_kv * tp * cp
    results.append((local_kv // 2**20, modeled_total // 2**20))
assert logical_kv == 4 * 2**30
one_byte_kv = logical_kv // element_bytes
assert one_byte_kv == 2 * 2**30  # excludes scale metadata
assert results == [(1024, 4096), (512, 8192), (512, 4096)]
```

Each result is `(MiB per rank, aggregate modeled KV MiB)`. The general helper in `examples/accelerator_portability.py` validates integer and head-divisibility assumptions; the printed calculation intentionally evaluates only the three named configurations.

The preceding CP example adds ranks; with prefill context parallelism disabled, [vLLM DCP](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) reuses them within TP. In the [Neuron SDK 2.32 decode mapping](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/design/parallelism/dcp.html), TP=16 and DCP=2 form pairs of ranks that would otherwise replicate a KV head. Each rank owns interleaved blocks covering half that head's history. For the assistant's 32 query heads and eight KV heads, the modeled payload becomes `512 / 2 = 256 MiB` per rank and `16 * 256 MiB = 4 GiB` aggregate. There are 16 ranks, not 32.

This mapping requires TP greater than the KV-head count, DCP no greater than TP/KV-heads, and DCP to divide both TP and the query-to-KV-head ratio; attention data parallelism is excluded. Our geometry passes these checks. Prefill groups and whole-model support need their own validation. Record group membership and keep the independent-CP helper separate from this nested-DCP calculation.

For admission, use the most constrained rank's peak live allocation, not the average across the cluster. A useful per-rank ledger is `weights + live KV + peak workspace/activations + runtime reservations + safety margin`. If one rank cannot fit its largest supported prefill chunk while retaining admitted decode state, spare bytes elsewhere do not rescue that executable. Reducing TP can remove collectives but increase per-rank weight traffic; adding CP reduces local history while introducing partial-attention communication. Measure both phases after each placement change.

#### Treat compiled variants as deployment artifacts

Compilation adds another budget. Suppose a serving implementation buckets prompt lengths 2048 and 4096, and pads a 2300-token prompt to 4096. A fully padded linear operation processes about `4096 / 2300 = 1.78` times as many positions. A dense quadratic operation would process about `1.78^2 = 3.17` times as many position pairs. These are padding ratios, not measured latency multipliers: masked tile skipping, ragged kernels, chunked prefill, and other operators change the result. Smaller buckets reduce waste but add executable variants, compilation work, and warm-up cost.

Capture cold compilation separately from warm execution. Then test a rolling deployment where new workers must become ready while old workers still serve traffic. A fast warm benchmark is insufficient if compiling or loading the required variants prevents timely recovery.

:::diagram compile_lifecycle|A warm request uses a compatible compiled variant; a miss needs an explicit policy. Prewarming and persistent artifacts reduce repeated work but do not make an unsupported shape legal. The cache key is a compatibility identity, not merely a prompt length.

Concrete tensor shapes, dtypes, static configuration, and target/compiler choices can affect specialization. The exact key belongs to the integration. For JAX, [JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html) and the [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) explain the distinction. Shape-polymorphic export can reuse tracing/lowering work without promising a single device executable for every concrete shape. For NKI, the SDK's specialization rules similarly distinguish tensor metadata from runtime values. A bounded valid length inside allocated storage is not an unbounded dynamic allocation.

Use a small cost calculation to decide which variants deserve prewarming. Suppose one variant costs 60 seconds to compile, a baseline call takes 125 milliseconds, and the candidate takes 62.5 milliseconds. At a fixed serial workload, it repays compilation after `60 / (0.125 - 0.0625) = 960` calls. These deliberately simple numbers are illustrative, not a vendor measurement. A rare shape used twenty times per worker may never repay its own compile cost. Sharing a trusted compatible artifact changes that calculation; so do overlap, throughput-oriented batching, and worker lifetime.

The CPU helper tests the padding and repayment arithmetic. To plan deployment, also count which configurations need separate artifacts. Assume a hypothetical integration specializes on two storage-length buckets, two dtypes, and two supported static layouts: that creates eight possible combinations. Its runtime valid-length value only changes a mask within the selected storage bucket. The actual integration defines the key; these choices are an example.

| Call | Storage length / valid length | Dtype / static layout | Artifact decision in this example |
| --- | --- | --- | --- |
| A | 4096 / 2300 | BF16 / layout 1 | Select the matching variant |
| B | 4096 / 3000 | BF16 / layout 1 | Reuse A's variant; change the runtime mask |
| C | 4096 / 2300 | FP16 / layout 1 | Different dtype variant |
| D | 4096 / 2300 | BF16 / layout 2 | Different static-layout variant |

Same prompt length therefore need not mean the same executable. A prewarm manifest records the required variants with model revision, target/compiler compatibility, and trusted artifact identity. Economics and readiness can also disagree: a rare required variant may deserve prewarming to meet cold-start SLOs even when that worker never repays its compilation cost through faster calls.

The serving decision is not “always add more buckets.” Choose a bounded set from the workload, warm the important variants before declaring readiness, and specify the miss policy: queue within a deadline, use a supported fallback, or reject. Do not assume an eager fallback exists on every integration. Count misses and compile time separately from prompt execution. A shared executable cache is also a trust boundary: accept artifacts only from trusted writers, with version/target compatibility and integrity checks.

### Training, prefill, and decode need different acceptance evidence

| Phase | What must work | What to measure on the target |
| --- | --- | --- |
| Training | Backward kernels, optimizer state, precision, checkpoint restore | Time to agreed quality; step tails; communication and recovery |
| Prefill | Prompt masks, ragged lengths, KV writes, chunk scheduling | TTFT by length; useful tokens/s; padding and compile misses |
| Decode | Paged state, sampling, cancellation, cache reuse | Token-gap tails; admitted concurrency; KV traffic and collectives |

Compare serving candidates on the assistant's request distribution and p99 targets. Check logits against a trusted reference with justified tolerances, then run task-level quality evaluation. Different reduction orders can change generated tokens without implying a bug; convincing-looking output can also hide one.

Replay eight active decodes while a long prompt arrives. Does the installed scheduler put prompt chunks and decode work in one batch, or alternate phase-specific batches? Which interval creates the worst visible token gap? The checked [vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html#segmented-prefill) describes segmented prefill and continuous batching with separate prefill and decode batches. Separate batches still permit request interleaving; they change how to interpret the trace.

For that replay, record the prefill and decode rank groups, KV ownership, and reductions, including which collectives cross hosts. Identical payload arithmetic can travel over very different communication paths. Judge the configuration by TTFT and token-gap tails under the existing load and quality gate. Part V's *Distributed Inference and Stateful Placement*, after its collective foundations, derives those decompositions; here the scheduler trace connects those choices to a user's wait.

For disaggregated serving, apply the backend, connector, and representation gates above even when both pools use the same vendor. Then budget repacking and transport. A shared checkpoint is a useful starting point, not a shared cache ABI.

Name the handoff unit before calculating its cost: one request, a microbatch, or a pool migration. The earlier 4 GiB represents the whole batch of eight sequences with 4096 committed tokens each. One 2000-token request instead has 250 MiB of logical KV under the same full-history, uncompressed geometry. Transfer bytes also depend on physical representation and which shards move.

Suppose that whole-batch 4 GiB handoff uses an illustrative effective link rate of 50 GiB/s. Its payload-only time is 80 milliseconds. If packing and unpacking each take 15 milliseconds and the operations are serialized, the handoff already costs 110 milliseconds before queueing, protocol overhead, and attention work. A separate 100-millisecond transfer budget fails even though the link-only estimate passed. Overlap requires a measured schedule and correct buffer lifetimes. This is planning arithmetic, not measured cross-vendor interoperability; the transfer budget is distinct from the assistant's visible token-gap SLO.

### Decide with a bounded experiment

Begin with one supported model and a replay containing short and long prompts, warm and cold prefixes, cancellation, and overload. Establish correctness, then profile the largest end-to-end gap. Only then decide whether it warrants a custom HIP, Triton, Pallas, or NKI kernel. Keep a known-good backend and versioned artifacts for rollback; do not attempt to recover an in-flight request by copying opaque device memory between ecosystems.

Compare cost per successful request that meets the SLO, including idle capacity, host resources, networking, compilation, and recovery. Add the engineering cost of maintaining another backend over an explicit planning horizon. For training, use time and cost to an agreed quality threshold, not merely examples per second. These denominators make a capacity-driven or cost-driven choice explainable without claiming that one vendor is universally faster.

For the assistant, retain the existing p99 first-token target of 1.5 seconds and p99 visible token-gap target of 100 milliseconds. Report them by prompt/output-length cohort at the same arrival process, including rejected and failed requests in the accounting. Do not claim a win by quietly admitting less traffic or shortening outputs. Report the final quality gate, sustained offered load, accepted concurrency, cold-worker readiness, peak per-rank memory, and SLO-qualified cost together. The best kernel result is a diagnostic; the deployment result is the decision.

| Evidence stage | Smallest useful artifact | What it still does not prove |
| --- | --- | --- |
| CPU semantics | Oracle tests; simulator or interpreter results where available | Device compilation, hardware numerics, or speed |
| Device operator | Pinned versions, shape/dtype sweep, numerical comparison | Model integration and request behavior |
| Integrated engine | Same model/features, quality gate, warm replay | Cold recovery, overload, sustained operating cost |
| Deployment trial | Arrival replay, SLO tails, failure/recovery, total cost | Universal superiority on other workloads |

The repository validates CPU planning and oracle tests. Accelerator listings remain source-only unless explicitly marked device-tested; no accelerator compilation or benchmark is recorded for this chapter. The companion examples carry target-validation checklists, and `docs/accelerator-acceptance-template.md` supplies a baseline/candidate report with unmeasured fields left explicit.

### Design Exercises

1. A custom RMSNorm passes a CPU interpreter or simulator at widths 4096 and 4097. What must happen before reporting a device speedup?
2. A TPU or Neuron serving port has fast warm decode but poor TTFT after scaling out. Once warm, a long prompt also interrupts eight active decodes. Which traces separate the two problems?
3. Why does the independent TP=4, CP=2 example use eight ranks, while the nested TP=16, DCP=2 example uses sixteen? Could a larger-memory target safely reduce TP?
4. A 2000-token request uses this chapter's cache geometry. A proposed cross-vendor handoff has a 25 GiB/s effective link, serialized pack/unpack costs of 2 ms each, and a 15 ms transfer budget. Is that enough evidence to approve it?
5. A three-element row uses a four-element Triton tile or a 256-thread HIP block. A developer divides by four in the first kernel and adds `if (tid >= n) return;` to the second. Diagnose both changes.

### Worked Solutions

#### 1. A passing interpreter is the first gate

Compile for the named device and pinned stack; check supported operations, layouts, and working-memory fit. Compare device outputs across the declared dtypes and boundary shapes, then warm and synchronize the timed region. Separate compilation, copies, dispatch, and kernel work. Finally measure the operator inside the engine under the same workload and quality gate. The CPU result establishes neither device legality nor the speed of an asynchronous device execution.

#### 2. Cold-start attribution

Trace compilation, artifact loading, allocation, warm-up, queue time, and prefill separately. Replay the same shapes on already-warm workers and count new executable variants. A cold-only regression points toward readiness and artifact distribution. For the warm interruption, inspect padded work and the scheduler's mixed or separate batches; align phase boundaries and collective intervals with visible token gaps. Require both the autoscaling test and the eight-decode replay before attributing the problem to a slow attention kernel.

#### 3. Memory is necessary, not sufficient

Independent CP adds a second rank dimension: `4 * 2 = 8`. Nested DCP partitions history among the existing sixteen TP ranks, giving 256 MiB/rank for the stated geometry. Passing DCP=2 as the helper's independent CP argument would double-count ranks. To reduce TP on a larger-memory target, recompute weights, KV, workspace, activations, and replication on the constraining rank. Confirm backend support, then compare TTFT, decode tails, and admitted concurrency under the same traffic: fewer ranks can remove collectives while reducing the bandwidth available to a request.

#### 4. A small margin and an unproven cache contract

The request's logical KV is `2000 * 131072 = 262144000` bytes, or 250 MiB. Payload transfer takes `250 / (25 * 1024)` seconds, about 9.766 ms; adding both conversions gives 13.766 ms. Only about 1.234 ms remains for the other transfer-path costs. This clears the illustrative arithmetic, not the deployment gate: establish actual wire bytes, compatible positions/head/dtype/quantization semantics, and latency under concurrent load. Test cancellation, partial transfer, ownership handoff, and failure. If that path is unsupported or misses the budget, keep homogeneous pools or route whole requests to each ecosystem.

#### 5. A tile is not the row, and inactive work still synchronizes

The Triton squared sum must be divided by the three actual elements. The zero-filled padding lane adds no squared value and should not change the normalization. In HIP, threads with no input elements still contribute zero and reach every block barrier. Returning them early violates the block-wide synchronization contract; keep them in the reduction or design a different supported collective.

The port ends with a supported deployment configuration and workload evidence, not a translated source tree. The next part adds peers to the device schedule: which rank owns a tensor, when another rank needs it, and what happens when that rank fails. Those questions survive every accelerator choice.
