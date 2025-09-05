import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import MaxNLocator

# 设置图表风格
style.use('seaborn-v0_8-notebook')
plt.rcParams["figure.figsize"] = (18, 5)
plt.rcParams["font.family"] = ["SimHei"]

# 从提供的数据中提取训练和验证指标
epochs = np.arange(1, 21)

# 训练集指标
train_loss = [0.3606, 0.1667, 0.1162, 0.0886, 0.0738, 0.0701, 0.0600, 0.0550, 0.0495, 0.0384,
              0.0362, 0.0326, 0.0316, 0.0294, 0.0264, 0.0259, 0.0246, 0.0235, 0.0233, 0.0233]
train_dice = [0.7035, 0.8619, 0.8782, 0.9010, 0.9122, 0.9139, 0.9249, 0.9306, 0.9378, 0.9499,
              0.9511, 0.9565, 0.9566, 0.9598, 0.9633, 0.9638, 0.9655, 0.9669, 0.9673, 0.9673]
train_iou = [0.6012, 0.7813, 0.8071, 0.8355, 0.8520, 0.8555, 0.8712, 0.8798, 0.8904, 0.9078,
             0.9109, 0.9188, 0.9197, 0.9244, 0.9305, 0.9315, 0.9343, 0.9368, 0.9375, 0.9374]

# 验证集指标
val_loss = [0.1887, 0.1168, 0.1164, 0.0765, 0.0845, 0.0751, 0.0629, 0.0713, 0.0644, 0.0602,
            0.0617, 0.0584, 0.0585, 0.0540, 0.0567, 0.0553, 0.0554, 0.0550, 0.0549, 0.0551]
val_dice = [0.8646, 0.8904, 0.8768, 0.9122, 0.9102, 0.9137, 0.9245, 0.9206, 0.9258, 0.9282,
            0.9294, 0.9319, 0.9327, 0.9390, 0.9369, 0.9368, 0.9383, 0.9394, 0.9395, 0.9392]
val_iou = [0.7851, 0.8190, 0.8051, 0.8561, 0.8538, 0.8599, 0.8762, 0.8702, 0.8757, 0.8813,
           0.8842, 0.8872, 0.8888, 0.8974, 0.8953, 0.8948, 0.8966, 0.8980, 0.8983, 0.8978]

# 创建三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# 1. 损失曲线
ax1.plot(epochs, train_loss, 'o-', color='#3B82F6', label='训练损失')
ax1.plot(epochs, val_loss, 's-', color='#EF4444', label='验证损失')
ax1.set_title('训练与验证损失', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('损失值', fontsize=12)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴只显示整数
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 2. Dice系数曲线
ax2.plot(epochs, train_dice, 'o-', color='#10B981', label='训练Dice')
ax2.plot(epochs, val_dice, 's-', color='#F59E0B', label='验证Dice')
ax2.set_title('训练与验证Dice系数', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Dice系数', fontsize=12)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# 3. IoU曲线
ax3.plot(epochs, train_iou, 'o-', color='#8B5CF6', label='训练IoU')
ax3.plot(epochs, val_iou, 's-', color='#EC4899', label='验证IoU')
ax3.set_title('训练与验证IoU', fontsize=14)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('IoU值', fontsize=12)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()

# 添加模型信息作为标题
plt.suptitle('模型训练指标可视化 (Params=91.79M, FLOPs≈185.73G @ 448x448)', fontsize=16, y=1.05)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 可选：保存图表
# fig.savefig('model_training_metrics.png', dpi=300, bbox_inches='tight')
