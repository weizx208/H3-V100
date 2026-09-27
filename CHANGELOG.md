# Changelog

## v2.0.2

Changes since v2.0.0.

- After a recoverable dual-attention execution resource failure, rebuild the
  input and complete the block on one GPU before allowing fresh dual admission
  later in the same sample. Allow at most two such recovery opportunities per
  device pair, shape and route per sample; preserve all capacity checks.
- Propagate fatal CUDA errors found during cleanup after releasing the run
  lock. Preserve ordinary recovery, cancellation and the original fatal error.
- Record larger successful workflows and the observed process-abort boundary
  without promising a universal sequence limit or guaranteed OOM recovery.

- Improve WDDM long-sequence dual-GPU head balance using live capacity and
  reusable QKV workspace. Preserve peak-memory reserves, host-pool checks and
  safe single-GPU fallback; capacity fallback warnings are rate-limited.
- Support cudaMallocAsync without disabling ComfyUI's allocation graph:
  secondary allocations, execution and cleanup share one owning worker;
  cross-block FP32 residuals use the model-root lifetime.
- Improve repeated-run and exception cleanup, including retained EasyCache
  predictions before sampler graph teardown. Preserve SOL calibration routing.
- Fix preset SOL Tau values (Quality 1.0, Speed 2.0, Ultra 2.5); show the editable
  Tau control only in Manual, with legacy workflow migration retained.
- Remove development timing collection and verbose planning logs from the
  installed node; clarify DynamicVRAM and unsupported core-weight errors.
- Retain the single public node, v2.0.0 model formats, audio precision and four
  native CUDA libraries. Consecutive full videos passed picture/audio acceptance;
  independent tests also passed cancellation/recovery in the tested conditions.

## v2.0.0

- Added dual-V100 execution for Flash and SOL, with automatic or explicit secondary-GPU selection.
- Added scaled FP8 E4M3 support alongside INT8 ConvRot.
- Added SOL Quality / Speed / Ultra / Manual presets and integrated EasyCache Off / Quality / Speed.
- Improved two-stage sampling, audio/video safeguards, and memory coordination for longer sequences and repeated runs.
- Upgrade by replacing the complete old folder, restarting ComfyUI, and adding a fresh Optimize node.

## v1.4.1

- Fixed repeat-run AIMDO `hostbuf_read_file_slice` process aborts by releasing
  an inactive, node-managed H3 DynamicVRAM model before the next prompt loads
  its text encoder.
- Cleans completed prefetch queues at that phase boundary before detaching H3,
  preventing stale VBAR pins or stream state from crossing into the next run.
- Limits the new load guard to models explicitly marked by H3 V100 Optimize;
  unrelated DynamicVRAM models retain ComfyUI's normal residency behavior.

## v1.4.0

- Promoted the validated DynamicVRAM profile to stable.
- Made Dynamic VBAR ownership and scaled FP16 SwiGLU built-in behavior.
- Added MLP-invocation-local fc1/fc2 expanded-weight reuse, eliminating the
  large cold-start penalty caused by repeated preparation for every chunk.
- Added role/state-based release of inactive prior-stage CUDA Dynamic models.
- Updated the validated launcher profile: no `--disable-dynamic-vram`, no
  `--lowvram`, and no global
  `--fast fp16_accumulation` on V100.
