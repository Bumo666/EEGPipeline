# EEG BCI Pipeline

最终交付在 [GitHub 仓库](https://github.com/Bumo666/EEGPipeline) 和 [Release 下载页](https://github.com/Bumo666/EEGPipeline/releases/latest)。Release 提供三个文件：本 README、`pipeline_summary.pdf` 和 `eeg_bci_pipeline_handoff.zip`。仓库保持私有，接收方需要仓库访问权限。

请下载 Release 附件中的 `eeg_bci_pipeline_handoff.zip` 并完整解压；GitHub 自动生成的 Source code ZIP 不包含 processed 数据。以下路径和命令均相对于解压后的 `eeg_bci_pipeline_handoff` 项目目录。原始下载缓存和虚拟环境不随包交付，需要重新生成数据时可能重新下载。

这个项目负责把 BCI Competition IV Dataset 2a / 2b 处理成统一的 PyTorch 可读格式，后面给 EEGNet、EEGNeX、ASGate、S-TARN 等模型直接调用。

模型组一般不需要再碰原始 GDF / MAT 文件，也不需要重新做划分和标准化。正式数据已经生成在：

```text
processed/
```

## 本次交接更新（2026-09-08）

在已有数据和 DataLoader 的基础上，补充了数据汇总表、EEGNet/EEGNeX 接入检查和最小训练入口。

- `reports/dataset_summary.csv`：2a、2b 共 18 行被试记录，直接核对了实际数组和标签。
- `reports/integration/smoke_test_models.json`：18 个被试乘 2 个模型，共 36 个组合通过，每个组合使用一个真实训练 batch。
- `reports/integration/training_example.json`：2a/2b subject 1 的两个模型各运行了两个训练 batch。
- 当前自动化测试为 24 项通过，安装环境的 `pip check` 通过。

这些结果说明当前数据可以接入指定版本的模型并完成参数更新。尚未完成论文准确率复现，也没有据此评估信噪比。最终 PDF 已同步模型接入结果、现有限制和后续展望；详细检查记录见 integration 报告。

## 给模型组的用法

收到交接压缩包后，先完整解压，在解压后的 `eeg_bci_pipeline_handoff` 文件夹打开 PowerShell。首次使用执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\demo_dataloader.py
```

如果只读取已处理的数据，可以只安装 `numpy torch`。安装依赖需要网络，读取包内 `processed/` 数据不需要下载原始数据。压缩包不包含虚拟环境和 MOABB 缓存。

下面的 F 盘路径是开发机器路径，同学使用时替换为自己的解压目录。运行读取代码时，请保持工作目录在项目根目录；在其他目录调用时，通过 `processed_root` 指定解压后的 `processed` 绝对路径。

进入项目目录：

```powershell
cd "F:\00_Research\mantou'4\pipeline"
```

如果虚拟环境还没激活：

```powershell
.\.venv\Scripts\activate
```

最常用的读取方式：

```python
from loaders.torch_dataset import get_dataloader

train_loader = get_dataloader(
    dataset="2a",
    subject=1,
    split="train",
    batch_size=64,
    model_format="eegnet",
)

test_loader = get_dataloader(
    dataset="2a",
    subject=1,
    split="test",
    batch_size=64,
    model_format="eegnet",
    shuffle=False,
)
```

EEGNeX 用法一样，只需要改 `model_format`：

```python
train_loader = get_dataloader(
    dataset="2a",
    subject=1,
    split="train",
    batch_size=64,
    model_format="eegnex",
)
```

`model_format="eegnet"` 返回：

```text
2a: x = [B, 1, 22, 1000], y = [B]
2b: x = [B, 1, 3, 1000],  y = [B]
```

这里：

```text
B    = batch size
1    = Conv2d 输入通道维度
22   = 2a 的 EEG 通道数
3    = 2b 的 EEG 通道数
1000 = 时间点数
```

如果模型直接吃 `[B, C, T]`，用：

```python
model_format="native"
```

这时返回：

```text
2a: [B, 22, 1000]
2b: [B, 3, 1000]
```

如果是 EEGNeX，也可以直接写：

```python
model_format="eegnex"
```

当前 `eegnex` 和 `eegnet` 的输入格式一致，都是：

```text
[B, 1, C, T]
```

## 当前正式数据

正式 processed 数据使用：

```text
source = MOABB
split = official_session
standardization = train_only
```

也就是说：

```text
读取 MOABB 官方整理数据
按官方 session 划分 train/test
只用 train 计算 mean/std
test 只使用 train 的 mean/std 做标准化
```

已经处理完成：

```text
2a: subject 01-09
2b: subject 01-09
```

保存结构：

```text
processed/
  BCICIV_2a/
    subject_01/
      train.npz
      test.npz
      meta.json
    ...
    subject_09/
      train.npz
      test.npz
      meta.json

  BCICIV_2b/
    subject_01/
      train.npz
      test.npz
      meta.json
    ...
    subject_09/
      train.npz
      test.npz
      meta.json

  summary.json
```

每个 `.npz` 里只有：

```text
X
y
```

保存时的基础格式是：

```text
X: [N, C, T]
y: [N]
```

DataLoader 会根据 `model_format` 决定是否临时加上 `[1, C, T]` 里的 `1`。

## 最终交接接口 shape

模型组实际需要记住下面这张表：

| 数据集 | processed 里的 X | native DataLoader | EEGNet / EEGNeX DataLoader | label |
| --- | --- | --- | --- | --- |
| 2a | `[N, 22, 1000]` | `[B, 22, 1000]` | `[B, 1, 22, 1000]` | `.npz: [N]`，batch: `[B]` |
| 2b | `[N, 3, 1000]` | `[B, 3, 1000]` | `[B, 1, 3, 1000]` | `.npz: [N]`，batch: `[B]` |

这里的 `N` 也可以写成 `D`，意思都是当前 subject、当前 split 里的 trial 数，不需要手动设置。`B` 是训练代码传入的 `batch_size`。`1` 是给 EEGNet / EEGNeX 这类 `Conv2d` 输入使用的维度。

当前已经处理好的真实 shape：

```text
2a train: [288, 22, 1000]
2a test:  [288, 22, 1000]，其中 subject 03 是 [287, 22, 1000]

2b train: [400, 3, 1000] / [420, 3, 1000] / [440, 3, 1000]
2b test:  [280, 3, 1000] / [320, 3, 1000]
```

## 数据集信息

Dataset 2a：

```text
MOABB: BNCI2014_001
subjects: 1-9
classes: left_hand, right_hand, feet, tongue
channels: 22
fs: 250 Hz
time_window: 0-4 s
bandpass: 4-38 Hz
official train session: 0train
official test session: 1test
```

常见 shape：

```text
subject 01 train: [288, 22, 1000]
subject 01 test:  [288, 22, 1000]
```

注意：`2a subject 03` 有 1 个 trial 被质量控制剔除，所以 test 是 `[287, 22, 1000]`。

Dataset 2b：

```text
MOABB: BNCI2014_004
subjects: 1-9
classes: left_hand, right_hand
channels: 3, C3/Cz/C4
fs: 250 Hz
time_window: 0-4 s
bandpass: 4-38 Hz
official train sessions: 0train, 1train, 2train
official test sessions: 3test, 4test
```

常见 shape：

```text
subject 01 train: [400, 3, 1000]
subject 01 test:  [320, 3, 1000]
```

2b 不同 subject 的 trial 数可能不同，这是 MOABB 读到的官方数据差异，不是 bug。具体看：

```text
processed/summary.json
```

## 评估协议

正式 baseline 只使用 official/session split。

不要把所有 trial 混在一起随机切 train/test。之前的 `within_subject` 随机划分只保留给调试用，不作为正式模型结果。

当前正式协议：

```text
2a:
  train = 0train
  test  = 1test

2b:
  train = 0train + 1train + 2train
  test  = 3test + 4test
```

## 防止数据泄漏

当前 pipeline 的顺序是：

```text
读取数据
-> 保留 EEG 通道
-> bandpass: 4-38 Hz
-> epoch: 0-4 s
-> crop: 1001 -> 1000
-> trial QC
-> official/session split
-> train-only channel quality report
-> train-only standardization
-> save processed
```

关键规则：

```text
先划分 train/test
再只用 train 计算 mean/std
test 只用 train 的 mean/std 标准化
```

模型组不要再做这些事情：

```text
不要重新随机划分 train/test
不要把 train + test 拼起来重新标准化
不要用 test accuracy 反复调参
不要为了统一 2a/2b 通道数而补零
不要把本地 GDF local 版本当作正式 baseline 数据
```

如果需要 validation set，必须进一步区分“外层官方训练集”和“内层训练/验证集”。当前 `train.npz` 已经使用整个外层训练集的均值和标准差；直接从这个文件再切 validation，不能声称标准化统计量与 validation 严格隔离。

严格验证流程是：另外生成未标准化的版本，从它的官方 train 内部切出 inner_train/validation，再只用 inner_train 拟合 `EEGStandardizer`。validation 只做 transform；模型、参数和训练轮数选定后，最终训练可以使用全部官方 train 拟合标准化并重新训练，最后评估官方 test。

需要这一版本时可执行下面的命令（会单独保存，不覆盖当前交接数据）：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset 2a --subject 1 --no-standardize --output-root processed_unstandardized
```

切分方式也要记录清楚，例如 train 内部的固定随机种子分层划分，或按 run/session 分组。此次新增的训练示例不划 validation、不读 test，不能用示例损失选最优模型。

## 检查数据状态

PowerShell 里可以运行：

```powershell
@'
import json
from pathlib import Path

for meta_path in sorted(Path("processed").glob("BCICIV_*/subject_*/meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(
        meta_path,
        "source=", meta["source"],
        "split=", meta["split"]["strategy"],
        "standardization=", meta["preprocessing"]["standardization"]["fit_on"]
    )
'@ | .\.venv\Scripts\python.exe -
```

正常情况下 18 行都应该是：

```text
source= MOABB
split= official_session
standardization= train_only
```

## 数据质量可视化

生成当前 processed 数据的质量图：

```powershell
.\.venv\Scripts\python.exe scripts\generate_quality_report.py
```

输出目录：

```text
reports/quality/
```

当前报告结果：

```text
Subjects checked: 18
Total rejected trials: 1
Subjects with bad channel candidates: 0
Source: MOABB
Split: official_session
Standardization: train_only
```

主要文件：

```text
quality_summary.csv
quality_report.md
trial_counts.png
rejected_trials.png
class_balance_2a.png
class_balance_2b.png
channel_std_2a.png
channel_std_2b.png
```

这些图只用于检查和汇报，不会修改 processed 数据。

## 生成汇报 PDF

生成当前 pipeline 汇报 PDF：

```powershell
.\.venv\Scripts\python.exe scripts\generate_pipeline_summary_pdf.py
```

输出文件：

```text
reports/pipeline_summary.pdf
```

PDF 内容包括当前成果、pipeline 协议、质量图展示、存在问题、后续展望和模型组交接说明。

## 重新生成数据

正式重新生成 processed 数据：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_all_subjects.py
```

这个命令默认等价于：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_all_subjects.py --source moabb --split-strategy official
```

如果网络不稳定：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_all_subjects.py --retries 5 --retry-delay 10
```

只处理某个 subject：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset 2a --subject 1
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset 2b --subject 1
```

本地 GDF 只用于调试：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset 2a --subject 1 --source local --split-strategy within_subject --data-root "F:\00_Research\mantou'4\BCICIV_2a_gdf"
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --dataset 2b --subject 1 --source local --split-strategy within_subject --data-root "F:\00_Research\mantou'4\BCICIV_2b_gdf"
```

本地 GDF 的 evaluation 文件没有真实类别标签，所以不要用 local 版本做最终 baseline。

## 快速验证 DataLoader

```powershell
.\.venv\Scripts\python.exe scripts\demo_dataloader.py
```

也可以直接检查 EEGNet 输入：

```powershell
@'
from loaders.torch_dataset import get_dataloader

for dataset, subject in [("2a", 1), ("2b", 1)]:
    loader = get_dataloader(
        dataset=dataset,
        subject=subject,
        split="train",
        batch_size=8,
        model_format="eegnet",
    )
    x, y = next(iter(loader))
    print(dataset, tuple(x.shape), tuple(y.shape))
'@ | .\.venv\Scripts\python.exe -
```

期望输出：

```text
2a (8, 1, 22, 1000) (8,)
2b (8, 1, 3, 1000) (8,)
```

## EEGNet / EEGNeX 接入验证

模型实现固定为 `braindecode==1.5.1` 的 `EEGNet` 和 `EEGNeX`，使用该版本默认网络参数。我们新增的 `models/baselines.py` 只负责输入适配和构建模型。实现来源见 [Braindecode 模型 API](https://braindecode.org/stable/api.html)；具体运行版本、构造参数、随机种子和数据 SHA256 都记在 JSON 报告里。

先安装可选依赖；仅使用 DataLoader 的同学不需要这一步：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

验证两个数据集的 subject 1：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_models.py
```

检查全部 18 个被试：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_models.py --all-subjects
```

每次检查包含真实数据读取、前向传播、交叉熵损失、反向传播和一次 Adam 更新。脚本会检查输出维度、损失、所有可训练参数的梯度和更新后的参数是否有限，并确认参数确实改变；异常时直接报错，不会把失败写成通过。

| 数据集 | DataLoader 输入 | 适配后 Braindecode 输入 | 模型输出 logits |
| --- | --- | --- | --- |
| 2a | `[B,1,22,1000]` | `[B,22,1000]` | `[B,4]` |
| 2b | `[B,1,3,1000]` | `[B,3,1000]` | `[B,2]` |

这里的维度转换只移除第二维的 `1`，不改变 EEG 通道数或时间点数。`model_format="eegnet"`/`"eegnex"` 是本项目的格式约定，不代表所有第三方同名模型都能直接接收这个形状。

## 最小训练入口

下面是默认入口，每个数据集、每个模型只对 subject 1 训练一个 batch：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline_example.py
```

指定模型和被试，执行两个训练 batch：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline_example.py --dataset 2a --model eegnet --subjects 1 --max-batches 2
```

遍历所有被试并各执行一个 batch：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline_example.py --all-subjects
```

`--batch-size` 默认为 8，`--epochs` 默认为 1，`--max-batches` 是每轮的 batch 上限，设为 0 才会遍历完整训练集。默认使用 CPU；有可用 CUDA 环境时可以传 `--device cuda`。每个被试都会重新创建模型和优化器，不继承上一个被试的权重。

模型组可以从 `scripts/train_baseline_example.py` 的 `run_subject()` 接入自己的训练流程。最小调用关系如下，代码应从项目根目录运行：

```python
import torch
from loaders.torch_dataset import get_dataloader
from models.baselines import build_baseline, train_step

loader = get_dataloader("2a", 1, "train", 8, "eegnet")
model = build_baseline("eegnet", n_chans=22, n_times=1000, n_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x, y = next(iter(loader))
result = train_step(model, optimizer, x, y, n_classes=4)
print(result)
```

2b 对应 `n_chans=3, n_classes=2`；脚本会从实际数据和 meta 自动读取，不需要手动改代码。这里用的是 logits 加 `CrossEntropyLoss`，不提前做 softmax。示例不保存正式模型、不报告 test accuracy，完整训练、验证集选参和论文实验协议仍由模型组确定。

## 完整数据汇总表

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset_summary.py
```

生成 `reports/dataset_summary.csv`，每个被试一行。包括实际 train/test 数量、shape、dtype、各类别数量、通道列表、标签映射、采样率、时间窗、带通参数、session 划分、标准化参数和 QC 剔除总数。

`train_left_hand_count` 等列按 `label_map` 计数；2b 的 feet/tongue 留空，表示没有这一类别，不是样本数为零。列表和字典保存为 JSON 字符串，CSV 使用 UTF-8 BOM，方便 Excel 打开。缺少任一被试文件、shape 与 meta 不符或发现未知标签时，脚本会报错。

当前汇总：2a train 共 2592 个 trial、test 共 2591 个；2b train 共 3680 个、test 共 2840 个。这些是汇总数量，训练仍按被试独立进行。2a 的一个测试 trial 已被 QC 剔除，正式对比论文结果时应说明这一差异。

## 更新交接包

```powershell
.\.venv\Scripts\python.exe scripts\package_handoff.py
```

在解压后的工程中执行该命令，会生成带时间戳的 `handoff/*.zip` 和 SHA256 文件。包内包含 `models/`、可选模型依赖、汇总表及接入测试报告。当前对外交付统一命名为 `eeg_bci_pipeline_handoff.zip`，旧版本已清理；文件校验清单位于包内的 `SHA256SUMS.json`。

## 环境和测试

依赖在：

```text
requirements.txt
```

主要包：

```text
moabb
mne
numpy
torch
pytest
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

当前测试结果：

```text
24 passed
```

## 代码结构

```text
pipeline/
|-- datasets/          # 2a/2b 数据读取，含 MOABB 和本地 GDF
|-- preprocessing/     # crop、artifact QC、normalization 等
|-- splits/            # official/session split 和 within-subject split
|-- loaders/           # PyTorch Dataset/DataLoader
|-- models/            # 指定版本的模型适配层，不是自行重写的网络
|-- scripts/           # 数据处理和 demo 脚本
|-- tests/             # 基础测试
|-- processed/         # 已生成的数据，不提交
|-- moabb_data/        # MOABB 缓存，不提交
`-- README.md
```

## 目前边界

这个仓库当前只负责数据 pipeline。

已经交付：

```text
raw/MOABB -> processed .npz/.json -> PyTorch DataLoader
```

还没有做：

```text
EEGNet / EEGNeX / ASGate / S-TARN 正式训练
跨被试评估
ICA / 自动坏通道插值 / ERP / WTC
```
