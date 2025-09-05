# DINOv2-UNet 息肉分割模型

## 项目概述
本项目实现了基于DINOv2预训练模型作为编码器的UNet风格分割模型，专为Kvasir-SEG数据集的息肉分割任务设计。项目包含两个核心脚本：
- `train.py`：模型训练主脚本，支持数据加载、模型训练、性能评估及结果保存
- `show.py`：训练指标可视化脚本，用于展示训练/验证过程中的损失、Dice系数和IoU曲线


## 环境准备

### 依赖安装
```bash
pip install numpy matplotlib torch torchvision timm pillow thop albumentations
```

关键依赖说明：
- `torch/torchvision`：深度学习框架，用于模型构建与训练
- `timm`：加载DINOv2预训练ViT模型作为编码器
- `thop`：（可选）计算模型参数量和FLOPs
- `albumentations`：（可选）数据增强支持
- `matplotlib`：`show.py`用于指标可视化


## 数据集准备
使用Kvasir-SEG息肉分割数据集，需按以下结构组织：
```
data_dir/
  ├─ images/  # 包含所有图像文件（.jpg/.png等格式）
  └─ masks/   # 包含对应掩码文件（.png格式，值为0或255）
```
数据集会自动划分为训练集（80%）、验证集（10%）和测试集（10%）。


## 使用指南

### 1. 模型训练（`train.py`）
```bash
# 基础用法
python train.py --data_dir /path/to/kvasir-seg --img_size 448 --batch_size 8 --epochs 80

# 完整参数示例
python train.py --data_dir ./kvasir-seg --img_size 448 --batch_size 8 --epochs 80 \
  --backbone vit_base_patch14_dinov2 --lr 1e-3 --lr_backbone 1e-5 --freeze_blocks_until 9
```

训练过程说明：
- 自动保存验证集性能最佳的模型（基于Dice+IoU均值）
- 每10个epoch保存一次验证集预测可视化结果
- 训练结束后自动在测试集上评估并保存测试结果


### 2. 训练指标可视化（`show.py`）
训练完成后，可通过以下命令可视化关键指标：
```bash
python show.py
```
脚本会生成3个子图：
- 训练/验证损失曲线
- 训练/验证Dice系数曲线
- 训练/验证IoU曲线

图表标题包含模型关键信息（参数量、FLOPs），支持保存为图片（默认注释，可启用）。


## 核心功能详解

### `train.py` 关键组件
1. **数据集处理**
   - `KvasirSEG`类：自动划分数据集，支持训练集数据增强（水平翻转、旋转、亮度/对比度调整等）
   - 验证集/测试集仅进行尺寸调整和标准化

2. **模型结构**
   - **编码器**（`VitDinoV2Encoder`）：基于DINOv2预训练ViT模型，支持冻结指定层数（`freeze_blocks_until`参数），输出多尺度特征
   - **解码器**（`FPNUNetDecoder`）：FPN-UNet混合结构，通过跨层特征融合和逐步上采样生成分割结果

3. **训练策略**
   - 损失函数：`ComboLoss`（BCE损失+Dice损失，权重各0.5）
   - 优化器：AdamW，支持编码器/解码器不同学习率（`lr`和`lr_backbone`）
   - 学习率调度：warmup预热+cosine衰减

4. **评估指标**：损失值、Dice系数、IoU、像素准确率


### `show.py` 功能说明
- 基于预设的20轮训练数据（损失、Dice、IoU）生成对比曲线
- 支持中文显示，图表包含网格线和图例，便于观察模型收敛趋势
- 标题显示模型关键性能指标（参数量Params=91.79M，FLOPs≈185.73G @ 448x448）


## 参数说明（`train.py`）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据集根目录（必需） | - |
| `--img_size` | 输入图像尺寸 | 448 |
| `--batch_size` | 批次大小 | 8 |
| `--epochs` | 训练轮数 | 80 |
| `--backbone` | DINOv2骨干网络型号 | vit_base_patch14_dinov2 |
| `--freeze_blocks_until` | 冻结编码器前N层（不参与训练） | 9 |
| `--lr` | 解码器学习率 | 1e-3 |
| `--lr_backbone` | 编码器学习率 | 1e-5 |
| `--warmup_epochs` | 学习率warmup轮数 | 5 |
| `--save_dir` | 模型和结果保存目录 | ./runs_dinov2_unet |
| `--no_amp` | 禁用混合精度训练 | False |


## 结果输出
- 模型权重：保存在`--save_dir`中，`best.pt`为验证集性能最佳模型
- 可视化结果：
  - 验证集预测：`save_dir/vis_epXXX`（每10轮保存）
  - 测试集预测：`save_dir/vis_test`
- 训练指标：通过`show.py`生成可视化曲线，反映模型收敛过程


## 模型特点
- 基于DINOv2预训练模型，利用强视觉表征能力提升分割性能
- 支持冻结编码器部分层，在小数据集上实现高效迁移学习
- 结合FPN和UNet解码器优势，有效融合多尺度特征
- 自动计算模型参数量和FLOPs，便于性能评估
- 完整的训练-评估-可视化流程，开箱即用
