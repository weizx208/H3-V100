# H3 V100 v2.0.0 更新说明

相对上一公开版 **v1.4.1**，本版将更多加速功能整合到同一个主节点中。

- **新增双卡支持**：可自动选择或指定第二张 V100，Flash 和 SOL 均可配合双卡运行。
- **新增 FP8 适配**：支持 scaled FP8 E4M3 UNet，并保留 INT8 ConvRot 支持。
- **升级 SOL**：提供 Quality、Speed、Ultra 和 Manual 档位，便于选择质量与速度。
- **集成 EasyCache**：提供 Off、Quality、Speed，分别保护视频与音频。
- **完善双采支持**：改善两段采样、latent 放大和 Sigma Refiner 的配合。
- **优化显存与稳定性**：改善长序列、连续运行及阶段切换的资源协调，保留显存不足时的保护。

升级请完整替换旧文件夹，重启 ComfyUI，并重新添加 Optimize 节点。继续使用默认 DynamicVRAM；8 步 Turbo LoRA 使用已验证的 **Euler**。建议先用已有短视频工作流确认音画。

使用方法与运行示例见 [中文说明](README_zh-CN.md) / [English](README.md)。
