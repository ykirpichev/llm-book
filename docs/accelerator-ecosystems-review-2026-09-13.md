# Accelerator ecosystems addition - September 13, 2026

## Scope and editorial decision

Added chapter 35, **Accelerator Ecosystems Beyond CUDA and NVIDIA**, at the end
of Part IV. The book now has 69 chapters. The chapter bridges single-device
kernel engineering and the distributed execution material that follows it.
Earlier chapter-by-chapter review documents describe their historical 68-chapter
snapshot; they are not reviews of this addition.

The new chapter keeps the documentation assistant and its model geometry as
the running example. It distinguishes source, semantic, and performance
portability, explains the software/hardware layers, and closes with a bounded
migration experiment and four exercises with worked solutions. It covers AMD
ROCm/HIP/RCCL, Triton, TPU/XLA/Pallas, AWS Neuron/NKI, and a short SYCL boundary.
It does not rank vendors or repeat marketing throughput comparisons.

## Primary-source checks

Sources are linked next to their claims in the manuscript. Checked September
13, 2026; moving documentation URLs are not reproducible deployment pins.

| Source | Claim or decision checked |
| --- | --- |
| Triton upstream repository | Kernel language/compiler; NVIDIA and AMD support; distinct from the inference server |
| AMD HIP porting guide | Translation boundaries, device-specific assumptions, library/assembly caveats |
| ROCm compatibility matrix | Hardware/OS/release qualification is necessary |
| RCCL overview | Multi-GPU and multi-node collective scope |
| Google TPU architecture guide | Matrix, vector, and scalar hardware resources |
| OpenXLA architecture; JAX Pallas guide | Graph compilation versus explicit custom-kernel control; backend-specific APIs |
| Google Cloud TPU inference guide | Current vLLM tpu-inference plugin and JAX/PyTorch paths |
| Neuron SDK overview | Hardware/SDK separation and framework integration choices |
| NKI introduction v2.31.0; programming model v2.26.0 | Tiled execution and HBM/SBUF/PSUM concepts; older version's exact constraints are not generalized |
| NxD release notes; Neuron update announcement | NxD Inference maintenance from SDK 2.32.0; newer vLLM Neuron beta without NxD dependency, targeting Trn2/Trn3 |
| vLLM Neuron migration guide | Supported-model deployment versus custom-model reimplementation |
| Khronos SYCL overview | Standard programming model is not a full serving stack |

## Technical and visual checks

- Recomputed 131,072 bytes per cached token and 4 GiB for eight 4096-token
  contexts. These are logical KV amounts before physical allocation overhead.
- Recomputed padding ratios: 4096/2300 is about 1.78; its square is about 3.17.
  These are work ratios, not measured latency claims.
- Training, prefill, and decode have separate acceptance evidence. Distributed
  placement and cross-vendor KV transfer are explicitly implementation-dependent.
- Added one native vector figure with a shared contract, four representative
  software paths, and four hardware targets. Its caption disclaims exhaustive
  coverage and notes Pallas GPU backends.
- Automated figure bounds, label overlap, connector/text, and manuscript
  diagram checks pass. A wording regression test checks the layer distinctions.
- Read the entire new chapter and inspected all seven rendered chapter pages.
  Also inspected the surrounding transitions and revised navigation/glossary.
- The first render left a nearly empty appendix page after the expanded glossary.
  Condensed the duplicate definitions into a compact cross-reference rather than
  changing the book's font size or shrinking the decision-index table.

## Validation limits

Final local checks: 87 CPU tests passed; PDF verification passed with 434 pages,
79 outline entries, 215 external-link annotations, and no flagged sparse pages.
Chapter 35 occupies pages 241-247. The final glossary and decision-index pages
were re-rendered and inspected after the pagination correction. The PDF skill's
render-and-inspect step caught the appendix issue that the structural verifier
did not flag.

This is a source-grounded editorial addition, not a device-validation report.
No CUDA, ROCm, TPU, or Neuron kernels were compiled or benchmarked. The CPU suite
and PDF checks cannot establish vendor support, model quality, or deployment
performance. This chapter received one editorial/technical pass here; no new
independent multi-model review is claimed.
