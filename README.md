# H3 V100 Optimize 2.0.3

[简体中文](README_zh-CN.md) | English

A MiniMax H3 acceleration node for ComfyUI on NVIDIA V100. One node combines model optimization, Flash / SOL attention, EasyCache, and optional dual-GPU execution.

2.0.3 fixes a reproduced long-sequence numerical overflow. 2.0.2 builds on 2.0.0 with improved WDDM long-sequence dual-GPU scheduling, repeated-run and resource-error recovery, and corrected SOL preset controls. The node acts on its input MODEL branch and does not rewrite ComfyUI or other nodes; it supports compatible MiniMax H3 model structures only.

## Demo

[![Video demonstration 575](assets/demo-575.jpg)](assets/demo-575.mp4)

[Watch with audio](assets/demo-575.mp4): 5 seconds, 0.7 MP upscaled 2× to 2304×1280, two-stage sampling, dual GPU, SOL Speed + EasyCache Speed.

## Features

- **V100 inference optimization**, with precision safeguards for critical video and audio computations.
- **Adaptive memory management**, including resource-aware chunking, prepared-weight reuse, and resource handoff between text encoding, sampling, and subsequent runs.
- **Dual V100 support**, with automatic or explicit secondary-GPU selection.
- **Scaled FP8 E4M3 UNet support**, alongside INT8 ConvRot.
- **SOL presets**: Quality, Speed, Ultra, and Manual.
- **EasyCache**: Off, Quality, and Speed, with separate video and audio safeguards.
- **Improved two-stage sampling**, including latent upscaling and Sigma Refiner workflows.

Core compute and memory policies apply automatically; no separate memory-management node is required. See [release notes](RELEASE_NOTES.md) for the v2.0.3 fix.

## Requirements

| Item | Requirements and validation scope |
|---|---|
| GPU | NVIDIA Tesla V100 (SM70); validated on one 16 GB card and two 16 GB cards |
| OS and Python | Windows x64, Python 3.12 |
| PyTorch | 2.8.0+cu128 |
| ComfyUI | A version supporting MiniMax H3 and DynamicVRAM |
| UNet | INT8 ConvRot or scaled FP8 E4M3; this does not cover every similarly named quantization format |

No extra pip packages are required. Do not replace a working Torch installation just to install this node. Precompiled CUDA libraries target the environment above; Linux and other Python/PyTorch combinations require rebuilding and validation.

The text encoder and video/audio VAEs remain managed by ComfyUI and the workflow. The text encoder does not need to stay on CPU. Host RAM, other resident models, and available VRAM affect capacity.

## Launch options

Keep the working DynamicVRAM configuration:

```text
Remove --disable-dynamic-vram
Remove --lowvram
Do not enable --fast fp16_accumulation
```

Fully restart ComfyUI and reload the model after changing these options. The node reports an error if the required DynamicVRAM configuration is unavailable.

WDDM dual-GPU mode supports the native allocator (`--disable-cuda-malloc`) and `cudaMallocAsync` in the validated environment. Keep your working allocator setting; the flag does not disable DynamicVRAM. The async path keeps ComfyUI's allocation graph enabled and has passed consecutive full-video runs with picture/audio acceptance, plus an independent cancellation/recovery test. These tests do not guarantee every ComfyUI version or hardware combination.

If the node reports `is_dynamic()=False`, check for `DynamicVRAM support detected and enabled` in the startup log and verify that ComfyUI, comfy-aimdo, and the model loader actually provide a DynamicVRAM ModelPatcher. Do not bypass that guard: a legacy patcher producing a video does not validate this node's dynamic weight and memory path.

For a core-weight-format error, check the model loader's log. This node accepts native INT8 ConvRot (group size 256) or scaled FP8 E4M3 H3 core weights only. `gguf qtypes: ... Q4_K` identifies an unsupported GGUF model, not a missing scale file; fabricating a scale or bypassing the check will not make it compatible.

## Installation and upgrade

1. Close ComfyUI. Back up and move the old `custom_nodes/H3_V100` folder elsewhere.
2. Extract the runtime package into `custom_nodes`, keeping the complete `H3_V100` folder and its four CUDA libraries. Do not replace individual Python files or mix old and new libraries.
3. Restart ComfyUI and refresh the page. Existing v2.0.x workflows can keep their node; when upgrading from v1.4.1, add and configure a fresh H3 V100 Optimize node.
4. Check video and audio using an existing short-video workflow before increasing resolution or duration.

## Workflow connections

Connect: **model loader → LoRA, if used → H3 V100 Optimize → sampler**.

For two-stage sampling, keep the usual latent-upscaler and Sigma Refiner connections. Each stage handles its own sampling progress and cache; a short second stage may not use SOL or cache skipping because of normal safeguards.

Do not stack this node or patches modifying the same H3 computations on one model. To compare settings, branch the model before Optimize. The validated sampler for the 8-step Turbo LoRA is **Euler**.

## Controls

| Control | Default | Purpose |
|---|---|---|
| Backend | Flash | Flash for exact attention; SOL for sparse acceleration |
| SOL_Quality | Quality | Quality favors quality; Speed favors speed; Ultra is more aggressive; Manual allows adjustment |
| SOL_Tau | 1.0 | Shown for SOL in Manual mode; higher values are generally more aggressive, so check outputs |
| EasyCache | Off | Off disables caching; Quality is more conservative; Speed is more aggressive |
| Dual_GPU | Disabled | Enables a second V100 to participate |
| Dual_GPU_ID | auto | Shown when dual GPU is enabled; automatically selects or explicitly chooses a process-visible secondary GPU |

## Choosing acceleration settings

Use **Flash + EasyCache Off** to establish a quality baseline. Then try SOL Quality, followed by Speed or Ultra based on the output. EasyCache can be used alone or together with SOL.

SOL and EasyCache can change outputs; visible differences depend on the scene. If fast motion, repeated details, or audio change, lower the relevant preset or disable caching for comparison. Speedups depend on sequence length, step count, and resources; not every configuration will be faster.

Dual GPU and two-stage sampling are independent options. The secondary GPU handles eligible parallel work, so unequal utilization and VRAM use are normal. Execution falls back to one GPU when conditions are unsuitable; dual-GPU participation is not guaranteed at every step.

## Dual-GPU operation

Supports computation across two V100 GPUs without requiring NVLink or enabling SLI. The current implementation stages transfers through host memory; the cards' VRAM is not combined into a single pool. [NVIDIA multi-GPU documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

Short sequences or SOL/EasyCache workloads may gain less or run slower. Results depend on computation, transfers, and available resources; a 2× speedup is not promised. This long-sequence scheduling update was validated under WDDM; TCC/NVLink performance is outside the v2.0.2 claim.

## Validated examples

| Duration | Resolution setting | Sampling | GPU setting |
|---|---|---|---|
| 5 s | 0.4 / 0.8 MP | Single-stage | Single GPU |
| 5 s | 2.0 MP | Single-stage | Single GPU |
| 5 s | 0.7 MP, upscaled 2× | Two-stage | Dual GPU |
| 15 s | 0.4 MP, upscaled 1.6× | Two-stage | Single and dual GPU tested separately |

MP means megapixels; upscaling multiplies both width and height. For example, 0.4 MP upscaled 1.6× becomes approximately 1.0 MP, subject to dimension alignment. Plan two-stage sampling resources around the final resolution.

These cases passed video/audio checks, but do not guarantee capacity for every configuration. Their prompts and settings differ, so they do not establish a fixed speedup over v2.0.0.

## Recovery after resource errors

After a recoverable dual-attention execution resource failure, the node rebuilds the input and completes the current block on one GPU before allowing fresh dual admission later in the same sample. Each device pair, shape and route has at most two recovery opportunities per sample. Capacity checks still apply; single-GPU execution may also fail for lack of resources. The budget resets for the next sample. Fatal CUDA errors and native process aborts cannot be guaranteed recoverable within the process.

## Practical limits

Start with **0.4 MP, 5 seconds**, then increase one setting at a time. Development tests also passed at approximately **1.1–1.2 MP, 15 seconds**, but execution can become much slower near resource limits and may still run out of memory.

Additional user tests completed 7+2-step workflows at 2560x1472 in 27:16 and 2752x1536 in 33:19. For latent T=37, batch one and the current patch layout, the video-only estimates are 136,160 and 152,736 tokens; text, audio and other inputs are additional. The earlier 1.2 MP, 15-second case logged 135,718 total packed tokens. These are observed cases, not fixed capacity limits.

The same stress test at 3008x1664 aborted the process during second-stage single-GPU QKV allocation after dual admission was rejected, requiring a ComfyUI restart. This case did not pass. The two successful logs do not show the execution-OOM recovery branch being triggered, so they do not establish real-GPU OOM recovery validation.

Capacity depends on weight format, LoRA, reference inputs, resolution, duration, host RAM, and ComfyUI version. Dual-GPU allocation uses available resources; VRAM capacity is not simply the sum of both cards. Adaptive support allows for 16+32 GB and 32+32 GB configurations, but these combinations have not completed hardware validation.

## Source and credits

Normal installation needs only the runtime package, which includes precompiled CUDA libraries. Matching CUDA/C++ source, required headers, and build scripts are provided separately in the source package and do not need to be installed in `custom_nodes`. Distributed under GPL-3.0-only, with third-party licenses retained; see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).

Thanks to [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [MiniMax H3 V100 Patch](https://github.com/Icbears/minimax-h3-v100-patch), [FlashAttention V100](https://github.com/Icbears/flash-attention-v100), [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/), [EasyCache](https://github.com/H-EmbodVis/EasyCache), and [CUTLASS](https://github.com/NVIDIA/cutlass).
