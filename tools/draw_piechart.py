# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2024年06月20日
"""

import os
import torch
import matplotlib.pyplot as plt


def draw_pie_chart(probs):
    """
    绘制饼状图表示概率分布
    :param probs: 一个表示概率分布的Tensor，其和应为1
    """
    # 确保输入的是一个Tensor且和为1
    assert isinstance(probs, torch.Tensor), "Input should be a PyTorch Tensor."
    assert torch.isclose(probs.sum(), torch.tensor(1.0)), "The sum of probabilities should be 1."

    # 将Tensor转换为list以便于matplotlib使用
    labels = [f'Prob {i + 1}' for i in range(len(probs))]
    sizes = probs.tolist()

    # 绘制饼状图
    plt.figure(figsize=(6, 6))
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    #plt.pie(sizes, autopct='%1.1f%%' , startangle=140)
    plt.pie(sizes, autopct=None, startangle=0)

    # 设置饼图中心空白
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # 使图形保持正圆
    plt.axis('equal')
    plt.tight_layout()

    #plt.title('Probability Distribution')
    #plt.legend()

    # 保存图像
    plt.savefig(f'tools/{traindate}/pie_chartCFunseen1tangle0.png')


# 示例：生成一个长度为5的概率分布Tensor

traindate = '0628'
if not os.path.exists(f'tools/{traindate}'):
    os.makedirs(f'tools/{traindate}')
#probs = torch.load(f'basis/{traindate}/ws_400x10_t0.01_187_999.pth')
probs = torch.load(f'basis/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_basis_400_id_10_unseen_ws_100x10_t0.01.pth')
ws = probs[1]
ws /= ws.sum()  # 确保概率之和为1
print(ws)
draw_pie_chart(ws)