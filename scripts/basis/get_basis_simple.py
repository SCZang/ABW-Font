import os
import cv2
import time
import glob
import tqdm
import torch
import argparse
import sklearn
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans #, AffinityPropagation, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='Get ContentFusion basis')
parser.add_argument('-c', '--content', type=str, default='../../../embedding_baseline/c_src.pth', help='path to content embedding')
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-if', '--ignore_font', default=[], type=int, nargs='+', help='the font to drop in basis')
parser.add_argument('-ic', '--ignore_char', default=[], type=int, nargs='+', help='the char to drop in basis')
parser.add_argument('-nb', '--basis_number', default=[10], type=int, nargs='+', help='the number of basis')
parser.add_argument('-lbs', '--load_bs', default=1, type=int, help='the batchsize for cal distance')
args = parser.parse_args()

cvs = torch.load(args.content)#.cpu().numpy()
k, n_samples, _, _, _ = cvs.shape
print(cvs.shape) # (400, 16, 256, 20, 20)

# filter out?
if len(args.ignore_font) == 0 and len(args.ignore_char) == 0:
    n_samples_remain = n_samples
else:
    ignore_font = args.ignore_font
    ignore_char = args.ignore_char
    ignore_char = torch.tensor(ignore_char)
    mask = torch.ones(n_samples, dtype=bool)
    mask.scatter_(0, ignore_char, False)
    n_samples_remain = mask.sum()
    print(f'remain: {n_samples_remain}/{n_samples}')
    cvs = cvs[:, mask]

# get embedding
cvs = cvs.reshape(*cvs.shape[:2], -1) # [400, n_samples_remain, xxx]=[400,16,102400]
print(cvs.shape)
# L1
cv_dis_s = []
per = args.load_bs
assert k%per == 0
#原本的方法 注意下面还有2句assert cv_dis.shape[0] == k and cv_dis.shape[1] == k，cv_dis = cv_dis.mean(-1)要取消注释
for i in tqdm.tqdm(range(k//per)):
    cv_dis = (cvs[:,None,:] - cvs[i*per:(i+1)*per][None,...]).abs().mean(-1) # [400,1,16,102400] - [1,5,16,102400] -> [400,5,16]
    #print(cvs[:,None,:].shape)[400, 1, 16, 102400]
    #print(cvs[i*per:(i+1)*per][None,...].shape)[1, 5, 16, 102400]
    print(cv_dis.shape)#[400,5,16]
    cv_dis_s.append(cv_dis)
cv_dis = torch.cat(cv_dis_s, 1)#[400,400,16]

# #新方法
# for i in tqdm.tqdm(range(400)):
#     cv_dis_i = cvs[i].view(16,-1)
#     # pca = PCA(n_components=16)
#     # cvs_pca = pca.fit_transform(cv_dis_i.numpy())
#     tsne = TSNE(n_components=2, perplexity=8, random_state=0)
#     cvs_tsne = tsne.fit_transform(cv_dis_i.numpy())
#     cv_dis_tensor = torch.from_numpy(cvs_tsne)
#     cv_dis_feature = torch.flatten(cv_dis_tensor)
#     cv_dis_s.append(cv_dis_feature)
# cv_dis = torch.stack(cv_dis_s)
# #一次
# cv_dis_flatten = cvs.view(400,-1)
# pca = PCA(n_components=256)
# cv_dis = torch.from_numpy(pca.fit_transform(cv_dis_flatten.numpy()))



assert cv_dis.shape[0] == k and cv_dis.shape[1] == k
# cv_dis += torch.eye(400)*1000
torch.save(cv_dis, os.path.join(os.path.dirname(args.content),
                                f'{args.model_name}_cv_dis_{k}x{k}x{n_samples_remain}_pca.pth'))

cv_dis = cv_dis.mean(-1)
print(cv_dis.shape)

# kmeans
for nb in tqdm.tqdm(args.basis_number):
    kmeans = KMeans(n_clusters=nb, random_state=0).fit(cv_dis)
    centers = kmeans.cluster_centers_ # [10, 400]
    print(centers.shape)
    dis_mat_l1 = np.abs(centers[:,None,:] - cv_dis.numpy()[None, :, :]).mean(-1) # [10,400]
    print(centers[:,None,:].shape)
    print(cv_dis.numpy()[None, :, :].shape)
    print(dis_mat_l1.shape)
    print(np.min(dis_mat_l1, axis=-1), np.argmin(dis_mat_l1, axis=-1))
    if not os.path.exists('basis'): os.mkdir('basis')
    # np.save(f'basis/{args.model_name}_basis_{k}_id_{nb}.npy', np.array(sorted(np.argmin(dis_mat_l1, axis=-1))))
    np.savetxt(f'basis/0613/{args.model_name}_basis_{k}_id_{nb}_20240328.txt', np.array(np.argmin(dis_mat_l1, axis=-1)), newline=' ',fmt='%d')
    #np.savetxt(f'basis/0613/{args.model_name}_basis_{k}_id_{nb}_20240328.txt', np.array(sorted(np.argmin(dis_mat_l1, axis=-1))), newline=' ',fmt='%d')
# 获取每个样本点所属的类别
labels = kmeans.labels_

#得到用来排序的indices
sorted_indices = np.empty_like(np.argmin(dis_mat_l1, axis=-1))
for i in range(len(np.argmin(dis_mat_l1, axis=-1))):
    sorted_indices[i] = np.where(sorted(np.argmin(dis_mat_l1, axis=-1)) == np.argmin(dis_mat_l1, axis=-1)[i])[0][0]
sorted_labels = np.empty_like(labels)
# 对标签进行重新排序
for i in range(len(labels)):
   sorted_labels[i] = sorted_indices[labels[i]]

# # 获取排序后的索引
# sorted_indices = np.argsort(np.argmin(dis_mat_l1, axis=-1))
# print(sorted_indices)
# sorted_labels = np.empty_like(labels)
# # 对标签进行重新排序
# for i in range(len(labels)):
#     sorted_labels[i] = np.where(sorted_indices == labels[i])[0][0]

# 打印每个样本点的类别
print("class of fonts:", sorted_labels)
# 将每个样本点的类别保存到文件
np.savetxt(f'basis/original_labels_{k}_id_{nb}_20240312.txt', sorted_labels, newline=' ', fmt='%d')

# vis
# imgs = []
# for i in np.array(sorted(np.argmin(dis_mat_l1, axis=-1))):
#     img = cv2.imread('data/data_221_S128F80_Base50_2x2_format/{:04}.png'.format(i))
#     imgs.append(img)

# plt.figure(dpi=200)
# plt.imshow(np.hstack(imgs))
