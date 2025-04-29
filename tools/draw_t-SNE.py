# # -*- coding:utf-8 -*-
# """
#
# 作者：Exiler
# 日期：2023年12月17日
# """
#
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
#
# # 加载你的数据 (cv_dis) 并进行 KMeans 聚类
# cv_dis = torch.load('../output/embeddings/embedding_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_cv_dis_400x400x16.pth')  # 用实际的路径替换
# cv_dis = cv_dis.mean(-1)
# print(cv_dis.shape)
# kmeans = KMeans(n_clusters=10, random_state=0).fit(cv_dis.numpy())
#
# # 使用 t-SNE 将维度降至 2D
# tsne = TSNE(n_components=2, random_state=0)
# tsne_result = tsne.fit_transform(cv_dis.numpy())
#
# # 绘制带有集群颜色的 2D t-SNE 可视化
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans.labels_, cmap='viridis', s=20)
#
# # 在每个点上添加标签
# for i in range(tsne_result.shape[0]):
#     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='black', fontsize=8, ha='right', va='bottom')
#
# plt.title('Clusters 的 t-SNE 可视化')
# plt.xlabel('t-SNE 分量 1')
# plt.ylabel('t-SNE 分量 2')
# plt.colorbar(scatter, label='集群标签')
#
# # 显示图例
# plt.legend(handles=scatter.legend_elements()[0], title='Cluster Labels')
#
# # 保存图像
# plt.savefig(f'tsne.png')
# plt.show()
#-----------------------------
# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2023年12月17日
"""

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
#
# # 加载你的数据 (cv_dis) 并进行 KMeans 聚类
# cv_dis = torch.load('../output/embeddings/embedding_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_cv_dis_400x400x16.pth')  # 用实际的路径替换
# cv_dis = cv_dis.mean(-1)
# print(cv_dis.shape)
# kmeans = KMeans(n_clusters=10, random_state=0).fit(cv_dis.numpy())
#
# # 使用 t-SNE 将维度降至 2D
# tsne = TSNE(n_components=2, random_state=0)
# tsne_result = tsne.fit_transform(cv_dis.numpy())
# centers = kmeans.cluster_centers_
#
# # 绘制带有集群颜色的 2D t-SNE 可视化
# plt.figure(figsize=(10, 8))
#
# # 绘制聚类点
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans.labels_, cmap='viridis', s=20, label='Data Points')
#
# # 在每个点上添加标签
# for i in range(tsne_result.shape[0]):
#     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='black', fontsize=8, ha='right', va='bottom')
#
# plt.title('Clusters 的 t-SNE 可视化')
# plt.xlabel('t-SNE 分量 1')
# plt.ylabel('t-SNE 分量 2')
# plt.colorbar(scatter, label='集群标签')
#
# # 显示图例
# plt.legend()
#
# # 保存图像
# plt.savefig(f'tsne.png')
# #plt.show()

#------------------------

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import tqdm
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='Choose Model to Cluster')
parser.add_argument('-item', type=int, default=180)
args = parser.parse_args()
item = args.item
# 加载你的数据 (cv_dis) 并进行 KMeans 聚类
file_path = f'output/embeddings/seen_embedding_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0628_{item}/c_src.pth'
#file_path = f'output/embeddings/0306/basis_400_epoch_{item-1}_999.pth'
#file_path = f'output/embeddings/embedding_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_cv_dis_400x400x16.pth'
cv_dis = torch.load(file_path)  # 用实际的路径替换

file_path2 = f'output/embeddings/seen_embedding_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0626_180/c_src.pth'
# #file_path = f'output/embeddings/0306/basis_400_epoch_{item-1}_999.pth'
# #file_path = f'output/embeddings/embedding_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_cv_dis_400x400x16.pth'
cv_dis2 = torch.load(file_path2)  # 用实际的路径替换
# file_path3 = f'output/embeddings/seen_embedding_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0618_192/c_src.pth'
# #file_path = f'output/embeddings/0306/basis_400_epoch_{item-1}_999.pth'
# #file_path = f'output/embeddings/embedding_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_cv_dis_400x400x16.pth'
# cv_dis3 = torch.load(file_path3)  # 用实际的路径替换
#使用未聚类数据要添加下面代码
# get embedding
cvs = cv_dis.reshape(*cv_dis.shape[:2], -1) # [400, n_samples_remain, xxx]=[400,16,102400]
cvs2 = cv_dis2.reshape(*cv_dis2.shape[:2], -1)
# cvs3 = cv_dis3.reshape(*cv_dis2.shape[:2], -1)
print(cvs.shape)
# L1
cv_dis_s = []
per = 5
k = 400
assert k%per == 0
# #原本的方法 注意下面还有2句assert cv_dis.shape[0] == k and cv_dis.shape[1] == k，cv_dis = cv_dis.mean(-1)要取消注释
# for i in tqdm.tqdm(range(k//per)):
#     cv_dis = (cvs[:,None,:] - cvs[i*per:(i+1)*per][None,...]).abs().mean(-1) # [400,1,16,102400] - [1,5,16,102400] -> [400,5,16]
#     #print(cvs[:,None,:].shape)[400, 1, 16, 102400]
#     #print(cvs[i*per:(i+1)*per][None,...].shape)[1, 5, 16, 102400]
#     print(cv_dis.shape)#[400,5,16]
#     cv_dis_s.append(cv_dis)
# cv_dis = torch.cat(cv_dis_s, 1)#[400,400,16]



print(cv_dis.shape)
#cv_dis = cv_dis.mean(-1)

cv_dis = cv_dis.reshape(cv_dis.shape[0],-1)
cv_dis2 = cv_dis2.reshape(cv_dis.shape[0],-1)

cc = torch.load(f'basis/0628/tc_400x10_t0.01_199_999.pth')
cv_dis = cv_dis.to(torch.cuda.current_device())
cc = cc.to(torch.cuda.current_device())
cv_dis = torch.cat((cv_dis,cc),dim=0)
# cv_dis3 = cv_dis3.reshape(cv_dis.shape[0],-1)
print(cv_dis.shape)
tsne = TSNE(n_components=2, random_state=0)
#cv_dis = cv_dis = torch.cat((cv_dis,cv_dis2),dim=0)
print(cv_dis.shape)
tsne_result1 = tsne.fit_transform(cv_dis.cpu().numpy())
tsne_result2 = tsne.fit_transform(cv_dis2.cpu().numpy())
plt.scatter(tsne_result1[:, 0], tsne_result1[:, 1],c='b', marker='s', label=f'Distribution1')
#plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1],c='r', marker='s', label=f'Distribution2')

# 为每个聚类分配一个独特的颜色
cluster_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
# 创建自定义颜色映射
cmap = ListedColormap(cluster_colors)

special_labels = np.arange(10)
print(special_labels)
special_points = plt.scatter(tsne_result1[-10:, 0], tsne_result1[-10:, 1],
                            c=special_labels, cmap=cmap, s=80,
                            edgecolors='black', linewidth=1,
                            label='Special Points')
plt.scatter(tsne_result1[400, 0], tsne_result1[400, 1],c='b', marker='X')
plt.scatter(tsne_result1[406, 0], tsne_result1[406, 1],c='black', marker='X')

plt.title('EPOCH {} Clusters t-SNE Visualization image'.format(item))
plt.xlabel('t-SNE dim1')
plt.ylabel('t-SNE dim2')
#plt.colorbar(scatter, label='集群标签')
#plt.colorbar(scatter)
# 显示图例
plt.legend()
traindate = '0628'
if not os.path.exists(f'tools/{traindate}'):
    os.makedirs(f'tools/{traindate}')
# 保存图像
plt.savefig(f'tools/{traindate}/0628200.png')

# tsne_result2 = tsne.fit_transform(cv_dis2.cpu().numpy())
# tsne_result3 = tsne.fit_transform(cv_dis3.cpu().numpy())
# plt.figure(figsize=(20, 16))
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='red', s=20, label='0')
# scatter2 = plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1], c='green', s=20, label='1')
# scatter3 = plt.scatter(tsne_result3[:, 0], tsne_result3[:, 1], c='blue', s=20, label='10')
# plt.title('EPOCH 0110 Clusters t-SNE Visualization image')
# plt.xlabel('t-SNE dim1')
# plt.ylabel('t-SNE dim2')
# #plt.colorbar(scatter, label='集群标签')
# #plt.colorbar(scatter)
# # 显示图例
# plt.legend()
# traindate = '0618'
# if not os.path.exists(f'tools/{traindate}'):
#     os.makedirs(f'tools/{traindate}')
# # 保存图像
# plt.savefig(f'tools/{traindate}/EPOCHtsne_with_closest_points1.png')
# #plt.show()

#
# save_item = item-1
# cc = torch.load(f'basis/0624/tc_400x10_t0.01_199_999.pth')
# #cc = torch.load(f'basis/0522/original_centers_20240522.pth')
# kmeans = KMeans(n_clusters=10,random_state=0).fit(cv_dis.cpu().numpy())
# centers = kmeans.cluster_centers_
#
# # 计算每个点到所有聚类中心的距离
# #distances = np.linalg.norm(cv_dis.numpy()[:, np.newaxis, :] - centers, axis=-1)
# cos_sim = cosine_similarity(centers, cv_dis.numpy())
# print(np.max(cos_sim, axis=-1), np.argmax(cos_sim, axis=-1))
# # distances = np.abs(centers[:,None,:] - cv_dis.numpy()[None, :, :]).mean(-1) # [10,400]
# # print(np.min(distances, axis=-1), np.argmin(distances, axis=-1))
#
# # 找到距离每个聚类中心最近的点的索引
# #closest_points_indices = np.argmin(distances, axis=-1)
# closest_points_indices = np.argmax(cos_sim, axis=-1)
#
# # 使用 t-SNE 将维度降至 2D
# #tsne = TSNE(n_components=2,perplexity=min(5, max(cv_dis.shape[0], cc.shape[0])-1), random_state=0)
# tsne = TSNE(n_components=2, random_state=0)
#
# print(cc.shape)
# #2
# cv_dis = cv_dis.to(torch.cuda.current_device())
# cc = cc.to(torch.cuda.current_device())
# cv_dis = torch.cat((cv_dis,cc),dim=0)
# print(cv_dis.shape)
# tsne_result = tsne.fit_transform(cv_dis.cpu().numpy())
# #cc_p = tsne.fit_transform(cc.numpy())
# # 绘制带有集群颜色的 2D t-SNE 可视化
# plt.figure(figsize=(20, 16))
# labels = kmeans.labels_
# # sorted_indices = np.empty_like(np.argmin(cos_sim, axis=-1))
# # for i in range(len(np.argmin(cos_sim, axis=-1))):
# #     sorted_indices[i] = np.where(sorted(np.argmax(cos_sim, axis=-1)) == np.argmax(cos_sim, axis=-1)[i])[0][0]
# # sorted_labels = np.empty_like(labels)
# # # 对标签进行重新排序
# # for i in range(len(labels)):
# #    sorted_labels[i] = sorted_indices[labels[i]]
# # labels = sorted_labels
#
# #3
# labels = np.concatenate((labels, np.arange(10)))
#
# # 为每个聚类分配一个独特的颜色
# cluster_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
# # 创建自定义颜色映射
# cmap = ListedColormap(cluster_colors)
# # 绘制聚类点
# #scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans.labels_, cmap='viridis', s=20, label='Data Points')
#
# # # 绘制每个聚类的数据点
# # scatter = None
# # for i in range(len(cluster_colors)):
# #     scatter = plt.scatter(tsne_result[kmeans.labels_ == i, 0], tsne_result[kmeans.labels_ == i, 1], c=cluster_colors[i], s=20, label=f'Cluster {i}')
# # 绘制每个聚类的数据点
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=cmap, s=20, label='Data Points')
# # 添加颜色条
# plt.colorbar(scatter, ticks=np.arange(len(cluster_colors)), label='Cluster')
# # special_points = plt.scatter(tsne_result[-10:, 0], tsne_result[-10:, 1],
# #                             color='red', edgecolors='black', s=50,
# #                             label='Special Points', marker='*')
# # 特别标注最后10个点，使用同样的cmap确保颜色一致
#
# #1
# special_labels = np.arange(10)
# print(special_labels)
# special_points = plt.scatter(tsne_result[-10:, 0], tsne_result[-10:, 1],
#                             c=special_labels, cmap=cmap, s=80,
#                             edgecolors='black', linewidth=1,
#                             label='Special Points')
#
# x = np.loadtxt(f'basis/0624/basis_400_epoch_199_999.txt',dtype=int)
# y = np.loadtxt(f'basis/0624/original_basis_20240624.txt',dtype=int)
# #x = np.loadtxt(f'basis/0522/original_basis_20240522.txt',dtype=int)
# # # 绘制x中的点，每个点用不同的颜色
# # for idx, point in enumerate(cc_p):
# #     plt.scatter(point[0], point[1], color=cluster_colors[idx % len(cluster_colors)], s=200)
#
# # cc_scatter = plt.scatter(cc_p[:,0], cc_p[:,1], marker= 'X', s = 100,label = 'center')
# # x = [109,162,201,205,227,245,289,293,307,327]
# # 在最近的10个点上添加标签
# # for i in closest_points_indices:
# #     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='black', fontsize=8, ha='right', va='bottom')
#
# for i in x:
#     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='brown', fontsize=20, ha='left', va='bottom')
# for i in y:
#     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='red', fontsize=20, ha='left', va='bottom')
# #在每个点上添加标签
# for i in range(tsne_result.shape[0]):
#     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i), color='black', fontsize=8, ha='right', va='bottom')
#
# # xx = [116, 116, 159, 331,  19, 217, 255,  51, 220, 301, 211, 192, 248, 315, 35, 179]
# # plt.scatter(tsne_result[xx, 0], tsne_result[xx, 1], c='red', marker='X', s=50, label='Test Points')
#
# # 绘制距离每个聚类中心最近的点，使用特殊颜色标记
# #plt.scatter(tsne_result[closest_points_indices, 0], tsne_result[closest_points_indices, 1], c='red', marker='X', s=100, label='Our Closest Points')
#
# #原文的聚类点
# #plt.scatter(tsne_result[x, 0], tsne_result[x, 1], c='blue', marker='X', s=100, label='Their Closest Points')
# plt.title('EPOCH {} Clusters t-SNE Visualization image'.format(item))
# plt.xlabel('t-SNE dim1')
# plt.ylabel('t-SNE dim2')
# #plt.colorbar(scatter, label='集群标签')
# #plt.colorbar(scatter)
# # 显示图例
# plt.legend()
# traindate = '0624'
# if not os.path.exists(f'tools/{traindate}'):
#     os.makedirs(f'tools/{traindate}')
# # 保存图像
# plt.savefig(f'tools/{traindate}/EPOCH{item}tsne_with_closest_points_1.png')
# #plt.show()
