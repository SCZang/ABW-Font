# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2024年01月15日
"""
import matplotlib.pyplot as plt

# 从txt文件中读取损失数据
# 指定你的文本文件路径
file_path = 'basis/save_loss_20240312.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取每个epoch的损失值
steps = []
d_losses = []
g_losses = []
step = 0
for line in lines:
    parts = line.split()
    steps.append(step)
    step=step+100
    d_loss = float(parts[-2][2:-1])  # 提取判别器损失值
    g_loss = float(parts[-1][2:-1])  # 提取生成器损失值
    #print(d_loss)
    #print(g_loss)


    d_losses.append(d_loss)
    g_losses.append(g_loss)

# 绘制损失图像
plt.figure(figsize=(10, 6))
plt.plot(steps, d_losses, label='Discriminator Loss')
plt.plot(steps, g_losses, label='Generator Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(f'tools/loss/0312Loss.png')

