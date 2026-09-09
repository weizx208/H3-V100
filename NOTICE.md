# Notices and component provenance

The H3 V100 node is distributed as a GPL-3.0-only work; see LICENSE. Existing
third-party copyright and permissive license notices remain applicable to their
components. No additional noncommercial or no-reverse-engineering restriction
is imposed. Model weights and ComfyUI are separate dependencies.

## Mixed precision integration

The H3 mixed-precision patch is adapted from
[Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch).
Its GPL-3.0-only text is included in licenses/H3_MIXED_PRECISION_GPL-3.0.txt.
This version uses a workflow-scoped MODEL patch, separate audio protection,
bounded activations, and FP8 / INT8 weight handling. Avoid applying both patches
to the same model.

## FlashAttention and CUDA headers

The SM70 attention implementation derives from
[Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100)
and FlashAttention by Tri Dao and contributors. The BSD-3-Clause license and
upstream AUTHORS are in licenses/FLASH_ATTENTION_BSD-3-CLAUSE.txt and
licenses/FLASH_ATTENTION_AUTHORS.txt. Upstream copyright notices remain in
the corresponding source headers. Local modifications include corrected sparse
attention, bounded range outputs, direct strided rectangular attention,
LSE-free inference, routing and centroid preparation, and inference-only
operator registration.

CUTLASS/CuTe C++ headers used for these builds are from NVIDIA CUTLASS 4.2.0.
Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. The applicable
BSD-3-Clause license is included verbatim in licenses/CUTLASS_BSD-3-CLAUSE.txt
and in the native source trees. This package does not distribute CuTeDSL.

QK normalization/RoPE and scaled SwiGLU/store sources, their registrations, and
the build scripts are included in the corresponding source package. PyTorch,
the CUDA toolkit and the compiler are external build dependencies, not bundled
toolchains.

## Algorithm references

SOL sparse routing with compensation refers to
[NVIDIA Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) and
[Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine).
Runtime-adaptive cache reuse refers to
[EasyCache by Xin Zhou, Dingkang Liang and collaborators](https://github.com/H-EmbodVis/EasyCache).
Their repositories describe Apache-2.0 licensing; the original EasyCache
Apache-2.0 text is retained in licenses/EASYCACHE_APACHE-2.0.txt.

The shipped H3 integration adds separate audio/video state, FP32 audio cache,
video INT8 residual storage, frame-risk checks, per-sampling-run reset,
resource-aware fallback and two-device coordination. The SM70 implementation
and H3 integration are not the upstream Sol-Engine or EasyCache distributions;
these references acknowledge the methods, not a claim that the underlying
algorithms were invented by this project.

The release-native source has removed uninstalled content-trajectory guidance,
validation/debug entrypoints, and ideal/run experiments. The matching source
archive includes the actual sources used for the distributed libraries and
their dependent headers, with no private prompts, videos or research reports.
