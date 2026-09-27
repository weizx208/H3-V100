# H3 V100 Optimize 2.0.2

简体中文 | [English](README.md)

面向 NVIDIA V100 的 MiniMax H3 ComfyUI 加速节点，将模型优化、Flash / SOL、EasyCache 和双卡支持集中在一个节点中。

2.0.2 在 2.0.0 的基础上改善 WDDM 长序列双卡调度、连续运行与资源错误恢复，并修正 SOL 档位控制。节点作用于传入的 MODEL 分支，不改写 ComfyUI 或其他节点的源文件；仅适用于受支持的 MiniMax H3 模型结构。

## 演示

[![575 视频演示](assets/demo-575.jpg)](assets/demo-575.mp4)

[观看带声音的视频](assets/demo-575.mp4)：5 秒，0.7 MP 放大 2 倍至 2304×1280，双采、双卡，SOL Speed + EasyCache Speed。

## 主要功能

- **基础推理加速**：针对 V100 优化 H3 计算，在关键画面与音频环节保留精度保护。
- **自适应显存管理**：根据可用资源调整分块，复用已准备的权重，并协调文本编码、采样及下一轮运行之间的资源释放。
- **双卡加速**：支持两张 V100，自动选择或手动指定副卡。
- **FP8 支持**：支持 scaled FP8 E4M3 UNet，保留 INT8 ConvRot 路线。
- **SOL 档位**：Quality、Speed、Ultra 和 Manual，按需要选择画质与速度。
- **EasyCache**：Off、Quality、Speed，加入音视频分别保护的缓存加速。
- **双采支持完善**：支持两段采样，可配合 latent 放大及 Sigma Refiner。

基础计算与显存策略自动生效，无需另外连接显存管理节点。相对 2.0.0 的完整改动见 [2.0.2 更新说明](RELEASE_NOTES.md)。

## 运行环境

| 项目 | 要求与验证范围 |
|---|---|
| 显卡 | NVIDIA Tesla V100（SM70）；已验证单张 16 GB 和双张 16 GB |
| 系统与 Python | Windows x64、Python 3.12 |
| PyTorch | 2.8.0+cu128 |
| ComfyUI | 包含 MiniMax H3 与 DynamicVRAM 支持的版本 |
| UNet | INT8 ConvRot 或 scaled FP8 E4M3；不代表支持所有同名量化格式 |

不需要额外安装 pip 包，也不要为安装本节点替换已经正常工作的 Torch。预编译 CUDA 库仅面向上述环境；Linux、其他 Python/PyTorch 组合需另行构建和验证。

文本编码器和视频/音频 VAE 仍由原工作流与 ComfyUI 管理，不要求文本编码器固定在 CPU。主机内存、其他常驻模型及显卡剩余显存都会影响可运行范围。

## 启动参数

使用正常工作的 DynamicVRAM 配置：

```text
移除 --disable-dynamic-vram
移除 --lowvram
不要启用 --fast fp16_accumulation
```

修改后完整重启 ComfyUI 并重新加载模型。未启用所需的 DynamicVRAM 时，节点会给出错误提示。

在已验证环境中，WDDM 双 GPU 模式支持原生分配器（`--disable-cuda-malloc`）及 `cudaMallocAsync`。沿用已正常工作的分配器设置即可；该参数不会关闭 DynamicVRAM。异步分配器路径保持 ComfyUI 模型分配图开启，已通过连续完整视频与音画验收，并完成独立的取消／恢复测试；不代表所有 ComfyUI 版本和硬件组合均已验证。

若节点提示 `is_dynamic()=False`，请检查启动日志是否出现 `DynamicVRAM support detected and enabled`，并检查 ComfyUI、comfy-aimdo 和模型加载器是否实际提供 DynamicVRAM ModelPatcher。不要绕过该校验：普通 ModelPatcher 即使能生成视频，也未经过本节点的动态权重／显存路径验证。

若提示核心权重格式不支持，先检查模型加载日志。本节点只接受原生 INT8 ConvRot（group size 256）或 scaled FP8 E4M3 H3 核心权重；`gguf qtypes: ... Q4_K` 表示 GGUF 模型，不是 scale 文件丢失，不能通过伪造 scale 或跳过校验来兼容。

## 安装与升级

1. 关闭 ComfyUI，将旧 `H3_V100` 文件夹备份到 `custom_nodes` 之外。
2. 将运行包解压到 `custom_nodes`，保留完整的 `H3_V100` 文件夹及其中四个 CUDA 库。不要只替换单个 Python 文件或混用新旧库。
3. 重启 ComfyUI 并刷新页面。从 2.0.x 升级可沿用原工作流；从 1.4.1 升级须重新添加并设置节点。
4. 用一个已有的短视频工作流确认画面和声音，再提高分辨率或时长。

## 工作流连接

连接方式：**模型加载 → LoRA（如有）→ H3 V100 Optimize → 采样器**。

双采时，latent 放大和 Sigma Refiner 继续按原工作流连接。两段分别处理自身的采样进度与缓存，短的第二段可能不启用 SOL 或缓存跳步，这是正常保护。

不要在同一模型上重复添加本节点，或叠加修改相同 H3 计算部分的加速补丁。比较不同设置时，应从 Optimize 之前分出模型分支。8 步 Turbo LoRA 的已验证采样器是 **Euler**。

## 节点参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| Backend | Flash | Flash 使用精确注意力；SOL 使用稀疏加速 |
| SOL_Quality | Quality | Quality 偏重质量；Speed 偏重速度；Ultra 更积极；Manual 手动调整 |
| SOL_Tau | 1.0 | 仅在 SOL 的 Manual 模式显示；值越大通常越积极，需检查画面 |
| EasyCache | Off | Off 关闭；Quality 较保守；Speed 更积极 |
| Dual_GPU | 关闭 | 启用第二张 V100 参与计算 |
| Dual_GPU_ID | auto | 双卡开启后显示；自动选择或指定当前进程可见的副卡 |

## 如何选择加速方式

需要建立画质基线时，使用 **Flash + EasyCache Off**。确认正常后，可切换 SOL Quality，再按画面表现尝试 Speed 或 Ultra；EasyCache 也可以单独开启或与 SOL 配合。

SOL 和 EasyCache 均可能改变输出，不能保证每个场景都没有可见差异。快速运动、重复细节或音频出现变化时，先降低相关档位，或关闭缓存做对照。加速受序列长度、步数和资源状态影响，并非所有配置都会更快。

双卡与双采是独立选项：单采也能用双卡，双采也能只用单卡。副卡只参与适合并行的计算，两张卡利用率和显存占用不必相同；条件不满足时会回到单卡，不承诺每一步均由双卡执行。

## 双卡运行

支持双 V100 协同计算，不依赖 NVLink，也无需开启 SLI。当前通过主机内存中转数据，两张卡的显存不会直接合并。[NVIDIA 多 GPU 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

短序列或已开启 SOL/EasyCache 时，双卡收益可能较小，甚至更慢；实际效果取决于计算量、数据传输和可用资源，不承诺两倍速度。本次长序列调度改动仅在 WDDM 下验收，TCC/NVLink 不属于 2.0.2 的性能承诺。

## 已通过的运行示例

| 时长 | 分辨率设置 | 采样 | GPU 设置 |
|---|---|---|---|
| 5 秒 | 0.4 / 0.8 MP | 单采 | 单卡 |
| 5 秒 | 2.0 MP | 单采 | 单卡 |
| 5 秒 | 0.7 MP 放大 2 倍 | 双采 | 双卡 |
| 15 秒 | 0.4 MP 放大 1.6 倍 | 双采 | 单卡、双卡分别通过 |

MP 指百万像素，放大倍数作用于宽和高。例如 0.4 MP 放大 1.6 倍，最终约为 1.0 MP，具体尺寸受像素对齐影响。双采的资源需求应按放大后的第二段计算。

上述为通过音画检查的具体案例，不是所有配置的容量保证。不同案例使用的提示词和设置不同，不用来推算相对 2.0.0 的固定加速倍数。

## 资源错误后的恢复

双卡 attention 执行遇到可恢复的资源错误时，先重建输入并用单卡完成当前 block，再允许同一采样段的后续调用重新尝试双卡准入。每个设备组合、形状和计算路线在一段采样内最多获得两次恢复机会；容量不足仍可走单卡，下一段重新计数。单卡也可能因资源不足而失败。致命 CUDA 错误及底层进程中止无法保证在当前进程恢复。

## 使用边界

首次使用可从 **0.4 MP、5 秒** 开始，再逐项增加。开发期间另有约 **1.1–1.2 MP、15 秒** 的通过记录，但接近资源边界时，运行时间可能明显增长，也仍有 OOM 的可能。

新增用户测试中，2560×1472 和 2752×1536 的 7+2 步双采分别完整运行 27:16 和 33:19。按 T=37、单样本及当前 patch 划分估算，视频部分分别为 136,160 和 152,736 tokens，尚未计入文本、音频等输入。此前 1.2 MP、15 秒案例记录的总序列为 135,718 tokens；这些数值不是固定容量上限。

同次压力测试的 3008×1664 档位在第二段开始时，双卡准入拒绝后转单卡 QKV 分配，发生底层进程中止，需要重启 ComfyUI。它不是成功案例。成功两档的日志未显示执行期 OOM 恢复分支被触发，因此不能据此宣称真实 GPU OOM 恢复已通过验证。

实际上限取决于权重格式、LoRA、参考输入、分辨率、时长、主机内存及 ComfyUI 版本。双卡按可用资源分配，显存不等于两张卡容量直接相加。16+32 GB、32+32 GB 留有自适应支持，但尚未完成这些组合的实机验收。

## 源码与致谢

普通使用只需安装运行包，其中已包含预编译 CUDA 库。对应 CUDA/C++ 源码、必要头文件和构建脚本另随源码包提供，无需安装到 `custom_nodes`。发布采用 GPL-3.0-only，第三方组件保留各自许可，详见 [NOTICE.md](NOTICE.md) 和 [LICENSE](LICENSE)。

感谢 [ComfyUI](https://github.com/Comfy-Org/ComfyUI)、[MiniMax H3 V100 Patch](https://github.com/Icbears/minimax-h3-v100-patch)、[FlashAttention V100](https://github.com/Icbears/flash-attention-v100)、[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/)、[EasyCache](https://github.com/H-EmbodVis/EasyCache) 和 [CUTLASS](https://github.com/NVIDIA/cutlass)。
