# 自定义HIP Kernel示例指南

本目录提供通过pybind11接入PyTorch的直接卷积示例。默认Baseline仍使用PyTorch原生卷积；选手可修改本示例并显式选择`hip`后端。本文命令均在项目根目录执行。

## 1. 文件与调用关系

| 文件 | 职责 |
| --- | --- |
| `../model.py` | U-Net通过`ReplaceableConv2d`调用卷积 |
| `../ops.py` | 选择后端、首次构建并加载扩展、传递当前设备执行流 |
| `bindings.cpp` | pybind11绑定、张量与参数检查、输出分配及Kernel调用 |
| `conv2d.hip` | 卷积Kernel与HIP启动函数，作为主要修改位置 |

调用顺序为：`UNet → ReplaceableConv2d → hip_conv2d → pybind11 conv2d → launch_conv2d_hip → HIP Kernel`。原生后端沿用`torch.nn.functional.conv2d`，权重名称与存储方式相同，仅替换卷积实现无需重新训练。

## 2. 接口要求

Python侧统一调用：

```python
conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

本示例支持以下输入，其余组合明确报错：

- 输入为连续NCHW布局，权重为连续OIHW布局，均为同一DCU上的FP32张量。
- 卷积核为1×1或3×3，stride=1、dilation=1、groups=1，各维padding为0或1。
- 偏置为同设备的连续FP32向量，长度等于输出通道数，也可为`None`。
- 输出尺寸遵循PyTorch Conv2d约定，dtype与device保持不变，不隐式复制输入或转移设备。

示例为逐输出元素计算的直接卷积，内部使用FP64乘加、最后写出FP32，以控制逐层累积误差；这不是高性能实现，也不是比赛对内部计算精度的统一限制。修改Kernel时保留边界检查、批量及通道索引，按传入的当前执行流启动。若调整启动函数签名，须同步修改绑定声明与调用。

自定义后端仅用于推理，不提供反向传播；Python调用须处于`model.eval()`和`torch.inference_mode()`下。框架负责检查与计时所需的设备同步，Kernel不自行增加全设备同步。

## 3. 构建环境

须使用兼容DCU的HIP版PyTorch、DTK、C++编译器和ninja。已验证的开发组合为Python 3.11.5、[PyTorch平台适配包](https://download.sourcefind.cn:65024/file/4/pytorch/DAS1.7/torch-2.7.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl)、DTK 25.04.2、GCC 12.2.0及NumPy 1.26.4；该PyTorch构建使用NumPy 1.x ABI。正式评测软件栈以比赛规定为准，不使用通用开发依赖覆盖其核心运行时。

在已申请DCU的计算作业中激活对应Python环境，再执行以下设置；模块名适用于已验证的开发集群：

```bash
module load compiler/gcc/12.2.0 compiler/dtk/25.04.2
export ROCM_HOME="$DTKROOT"
export MAX_JOBS=4
export PYTORCH_ROCM_ARCH=$(python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')
```

确保该环境的`python`、`ninja`可执行文件位于PATH。HIP版PyTorch仍使用`cuda`设备标识。首次调用`hip`会由`../ops.py`构建并缓存扩展；CPU或CUDA版PyTorch不能运行此示例，不支持时不会自动回退为原生卷积。

## 4. 修改与验证

修改`src/csrc/conv2d.hip`后，先运行接口检查，再用同一权重评估和计时：

```bash
python scripts/check_hip.py --output outputs/hip-check.json
python src/main.py evaluate --checkpoint weights/baseline.pt \
  --device cuda --backend hip --output outputs/hip-evaluation
python src/main.py benchmark --checkpoint weights/baseline.pt \
  --device cuda --backend hip --output outputs/hip-benchmark.json
```

| 输出 | 内容 |
| --- | --- |
| `outputs/hip-check.json` | 编译加载、算子与整网误差、非法输入拒绝等检查结果；成功时`status`为`passed` |
| `outputs/hip-evaluation/metrics.json` | 公开验证集BCE、Dice等指标 |
| `outputs/hip-benchmark.json` | 逐层卷积、整网及含预后处理流程的正确性与性能对照 |

接口检查与benchmark使用`atol=1e-5、rtol=1e-4`逐元素对照，超出容差即失败。该容差用于FP32示例的实现验证，不替代比赛的Dice评分。开发benchmark包含预热，不能作为正式计分时间；完整程序计时包括运行时编译、导入和输入输出。

## 5. 提交自定义Kernel版本

命令行选择`--backend hip`仅影响该次命令。提交时须在`submission_template/submission/main.py`中将后端参数改为`"hip"`：

```python
model, _ = load_checkpoint(Path(__file__).parent / "weights/model.pt", "hip", device)
```

保留三参数入口及模板中的推理上下文，执行：

```bash
python scripts/package_submission.py --source submission_template \
  --reference --output submission-hip.zip
```

打包器会收集参考模块、权重及本目录的两个源码文件，分别存为`submission/csrc/bindings.cpp`和`submission/csrc/conv2d.hip`。本指南不属于运行依赖，不进入参赛提交ZIP。自定义权重可通过`--checkpoint`指定；若新增源码文件，须同步调整扩展构建清单与提交收集逻辑。更换模型或自行组织代码时按主README的提交规范打包，确保所有依赖位于提交目录内。

本接口是可选示例，不限定参赛方案必须使用卷积替换。其他实现可通过`register_backend`接入；额外模块须随提交提供，并在模型创建前完成注册。比赛优化限制见[赛题说明](../../README.md)。
