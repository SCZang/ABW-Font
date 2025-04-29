# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2024年03月24日
"""
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
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='Get ContentFusion basis')
parser.add_argument('-c', '--content', type=str, default='../../../embedding_baseline/c_src.pth', help='path to content embedding')
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-if', '--ignore_font', default=[], type=int, nargs='+', help='the font to drop in basis')
parser.add_argument('-ic', '--ignore_char', default=[], type=int, nargs='+', help='the char to drop in basis')
parser.add_argument('-nb', '--basis_number', default=[10], type=int, nargs='+', help='the number of basis')
parser.add_argument('-lbs', '--load_bs', default=1, type=int, help='the batchsize for cal distance')
parser.add_argument('-up', '--unseen_points', type=str, help='')
args = parser.parse_args()

cvs = torch.load(args.content)#.cpu().numpy()
k, n_samples, _ = cvs.shape
print(cvs.shape) # (400, 16, 128)

unseen_points = torch.load(args.unseen_points)

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

# # get embedding
# cvs = cvs.reshape(*cvs.shape[:2], -1) # [400, n_samples_remain, xxx]=[400,16,102400]
# print(cvs.shape)
# # L1
# cv_dis_s = []
# per = args.load_bs
# assert k%per == 0
# #原本的方法 注意下面还有2句assert cv_dis.shape[0] == k and cv_dis.shape[1] == k，cv_dis = cv_dis.mean(-1)要取消注释
# for i in tqdm.tqdm(range(k//per)):
#     cv_dis = (cvs[:,None,:] - cvs[i*per:(i+1)*per][None,...]).abs().mean(-1) # [400,1,16,102400] - [1,5,16,102400] -> [400,5,16]
#     #print(cvs[:,None,:].shape)[400, 1, 16, 102400]
#     #print(cvs[i*per:(i+1)*per][None,...].shape)[1, 5, 16, 102400]
#     print(cv_dis.shape)#[400,5,16]
#     cv_dis_s.append(cv_dis)
# cv_dis = torch.cat(cv_dis_s, 1)#[400,400,16]
#
# # #新方法
# # for i in tqdm.tqdm(range(400)):
# #     cv_dis_i = cvs[i].view(16,-1)
# #     # pca = PCA(n_components=16)
# #     # cvs_pca = pca.fit_transform(cv_dis_i.numpy())
# #     tsne = TSNE(n_components=2, perplexity=8, random_state=0)
# #     cvs_tsne = tsne.fit_transform(cv_dis_i.numpy())
# #     cv_dis_tensor = torch.from_numpy(cvs_tsne)
# #     cv_dis_feature = torch.flatten(cv_dis_tensor)
# #     cv_dis_s.append(cv_dis_feature)
# # cv_dis = torch.stack(cv_dis_s)
# # #一次
# # cv_dis_flatten = cvs.view(400,-1)
# # pca = PCA(n_components=256)
# # cv_dis = torch.from_numpy(pca.fit_transform(cv_dis_flatten.numpy()))
#
#
#
# assert cv_dis.shape[0] == k and cv_dis.shape[1] == k
# # cv_dis += torch.eye(400)*1000
# torch.save(cv_dis, os.path.join(os.path.dirname(args.content),
#                                 f'{args.model_name}_cv_dis_{k}x{k}x{n_samples_remain}_pca.pth'))
#
# cv_dis = cv_dis.mean(-1)

cv_dis = cvs.reshape(k, -1)
unseen_points = unseen_points.reshape((unseen_points.shape[0],-1))
print(cv_dis.shape)#(400,2048)

# kmeans
for nb in tqdm.tqdm(args.basis_number):
    kmeans = KMeans(n_clusters=nb, random_state=0).fit(cv_dis)
    centers = kmeans.cluster_centers_ # [10, 2048]
    cos_sim = cosine_similarity(centers, cv_dis.numpy())
    # print(centers.shape)
    # dis_mat_l1 = np.abs(centers[:,None,:] - cv_dis.numpy()[None, :, :]).mean(-1) # [10,400]
    # print(centers[:,None,:].shape)
    # print(cv_dis.numpy()[None, :, :].shape)
    # print(dis_mat_l1.shape)
    # print(np.min(dis_mat_l1, axis=-1), np.argmin(dis_mat_l1, axis=-1))
    print(cos_sim.shape)#(10,400)
    print(np.max(cos_sim, axis=-1), np.argmax(cos_sim, axis=-1))
    if not os.path.exists('basis'): os.mkdir('basis')
    # np.save(f'basis/{args.model_name}_basis_{k}_id_{nb}.npy', np.array(sorted(np.argmin(dis_mat_l1, axis=-1))))
    np.savetxt(f'basis/{args.model_name}_basis_{k}_id_{nb}_20240426.txt', np.array(sorted(np.argmax(cos_sim, axis=-1))), newline=' ',fmt='%d')


# 获取每个样本点所属的类别
labels = kmeans.labels_
print("labels:", labels)
#得到用来排序的indices
sorted_indices = np.empty_like(np.argmin(cos_sim, axis=-1))
for i in range(len(np.argmin(cos_sim, axis=-1))):
    sorted_indices[i] = np.where(sorted(np.argmax(cos_sim, axis=-1)) == np.argmax(cos_sim, axis=-1)[i])[0][0]
sorted_labels = np.empty_like(labels)
# 对标签进行重新排序
for i in range(len(labels)):
   sorted_labels[i] = sorted_indices[labels[i]]

# 打印每个样本点的类别
print("class of fonts:", sorted_labels)
# # 将每个样本点的类别保存到文件
# np.savetxt(f'basis/original_labels_{k}_id_{nb}_20240312.txt', sorted_labels, newline=' ', fmt='%d')


cv_dis_np = cv_dis.numpy()

gauss_mean = []
gauss_sigma = []
for cluster_idx in range(nb):  # 遍历每个聚类
    # 选择属于当前聚类的数据点
    print(len(cv_dis_np[np.where(sorted_labels == cluster_idx)]))
    cluster_points = cv_dis_np[np.where(sorted_labels == cluster_idx)]


    # 计算当前聚类的均值向量
    mean_vector = torch.tensor(np.mean(cluster_points, axis=0))

    # 计算当前聚类的协方差矩阵
    #covariance_matrix = torch.tensor(np.cov(cluster_points, rowvar=False))
    #sample_variances = torch.tensor(np.var(cluster_points, axis=0, ddof=1))
    sample_variances = torch.tensor(np.std(cluster_points, axis=0, ddof=1))
    # 转换为对角阵
    diagonal_covariance_matrix = torch.diag(sample_variances)
    # 计算对角阵的行列式
    # print(diagonal_covariance_matrix.shape)
    # print(diagonal_covariance_matrix)
    # part = diagonal_covariance_matrix[:5]
    # determinant = torch.prod(part)
    # print(determinant)
    #print(np.linalg.det(covariance_matrix))

    # 将当前聚类的拉普拉斯分布参数存储到列表中
    gauss_mean.append(mean_vector)
    gauss_sigma.append(diagonal_covariance_matrix)
gauss_mean = torch.stack(gauss_mean)
gauss_sigma = torch.stack(gauss_sigma)

# # 打印每个聚类的拉普拉斯分布参数
# for cluster_idx, (mean_vector, covariance_matrix) in enumerate(laplace_params):
#     print(f"Cluster {cluster_idx + 1}:")
#     print("Mean vector:", mean_vector)
#     print("Covariance matrix:", covariance_matrix.shape)
#     print(diagonal_covariance_matrix.shape)
#     print()

def real_laplace_log_pdf(x, mean, scale_vector):
    # print(x.shape)#(2048)
    # print(mean.shape)#2048
    # print(scale_vector.shape)#2048
    x = x.numpy()
    mean = mean.numpy()
    scale_vector = scale_vector.numpy()
    """
    计算高维独立分量拉普拉斯分布的对数概率密度。

    参数:
        x (ndarray): 样本向量，形状为 (n,) 或 (N, n)，其中 N 是样本数，n 是特征数。
        mean (ndarray): 均值向量，形状为 (n,)。
        scale_vector (ndarray): 尺度向量，形状为 (n,)，每个元素对应一个分量的尺度参数。

    返回:
        ndarray: 对数概率密度值，形状与输入 `x` 相同。如果 `x` 是单个样本，返回标量；如果是多个样本，返回形状为 (N,) 的向量。
    """
    n = len(mean)
    assert x.shape[-1] == n and scale_vector.shape[0] == n

    if x.ndim == 1:  # 处理单个样本的情况
        log_pdf_values = -n * np.log(2) - np.log(scale_vector) - np.abs(x - mean) / scale_vector
        log_pdf_value = np.sum(log_pdf_values)
    else:  # 处理多个样本的情况
        log_pdf_values = -n * np.log(2)[:, np.newaxis] - np.log(scale_vector)[:, np.newaxis] - np.abs(x - mean[:, np.newaxis]) / scale_vector[:, np.newaxis]
        log_pdf_value = np.sum(log_pdf_values, axis=-1)

    return log_pdf_value


def log_bessel_knu(x,v):
    # Implement the log of modified Bessel function of the second kind here.
    # You can use scipy.special.kve for this purpose.
    # Convert the output to PyTorch tensor for later use.
    return torch.tensor(np.log(scipy.special.kv(v,x)))


def log_pdf_multivariate_laplace(x, mean, covariance_matrix):
    k = len(mean)  # dimensionality
    v = (2 - k) / 2  # (2 - k) / 2
    mu = torch.tensor(mean).unsqueeze(0)
    # 计算特征值
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    # 检查所有特征值是否为正
    is_positive_definite = all(eigenvalues > 0)
    print(is_positive_definite)

    Sigma_inv = torch.inverse(torch.tensor(covariance_matrix))
    mu = mu.to(Sigma_inv.dtype)
    sign_0, log_det_covariance = np.linalg.slogdet(covariance_matrix)
    #print(Sigma_inv.shape)
    x_prime = x  # x' = x - mu
    if x_prime.dtype != Sigma_inv.dtype:
        x_prime = x_prime.to(Sigma_inv.dtype)
    #x_prime = x_prime.view(-1, x.shape[-1])  # Flatten x_prime to a 1D vector
    # xp_sigmainv = x_prime.T @ Sigma_inv
    # print(xp_sigmainv.shape)
    # xp_si_x = xp_sigmainv @ x_prime
    # print(xp_si_x)
    x_prime_Sigma_inv_x = x_prime @ Sigma_inv @ x_prime.T # x' Sigma^-1 x
    #print(x_prime_Sigma_inv_x)
    log_x_prime_Sigma_inv_x = np.log(x_prime_Sigma_inv_x)
    #print(log_x_prime_Sigma_inv_x)
    x_prime_Sigma_inv_mu = x_prime @ Sigma_inv @ mu.T
    two_plus_mu_p_Sigma_inv_mu = 2 + (mu @ Sigma_inv @ mu.T)
    #print(two_plus_mu_p_Sigma_inv_mu)
    log_two_plus_mu_p_Sigma_inv_mu = np.log(two_plus_mu_p_Sigma_inv_mu)
    #print(log_two_plus_mu_p_Sigma_inv_mu)
    #log_x_prime_Sigma_inv_x = np.log(x_prime @ Sigma_inv @ x_prime.T)  # x' Sigma^-1 x
    #print(log_x_prime_Sigma_inv_x)
    aaa = np.log(2)+x_prime_Sigma_inv_mu
    bbb = - k/2*np.log(2*np.pi)-0.5*log_det_covariance
    ccc = v/2*log_x_prime_Sigma_inv_x
    ddd = -log_two_plus_mu_p_Sigma_inv_mu
    eee = np.sqrt(two_plus_mu_p_Sigma_inv_mu * x_prime_Sigma_inv_x)
    fff = log_bessel_knu(eee, v)#cant compute
    print(aaa,bbb,ccc,ddd,eee,fff)
    # log_pdf = -np.log(2 * np.pi) * k / 2 - 0.5 * log_det_covariance \
    #           - 0.5 * v * log_x_prime_Sigma_inv_x - log_bessel_knu(v + 0.5 * x_prime_Sigma_inv_x, v) \
    #           - 0.5 * (2 + mu @ Sigma_inv @ mu.T) * x_prime_Sigma_inv_x
    log_pdf = np.log(2)+x_prime_Sigma_inv_mu - k/2*np.log(2*np.pi)-0.5*log_det_covariance \
              +v/2*log_x_prime_Sigma_inv_x-log_two_plus_mu_p_Sigma_inv_mu \
              + log_bessel_knu(np.sqrt(two_plus_mu_p_Sigma_inv_mu * x_prime_Sigma_inv_x),v)
    return log_pdf
def laplace_pdf(x, mean, covariance_matrix):
    d = len(x)

    exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(covariance_matrix)), (x - mean))
    # prefactor = 1 / np.linalg.det(covariance_matrix) ** 0.5
    prefactor = 1 /  torch.det(covariance_matrix.double()) ** 0.5
    prefactor = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(covariance_matrix) ** 0.5)
    pdf_value = prefactor * np.exp(exponent)
    return pdf_value



def laplace_log_pdf(x, mean, covariance_matrix):
    d = len(x)
    #print(d)
    # 计算协方差矩阵的行列式的对数
    # B = np.linalg.det(covariance_matrix)
    # print(B)
    sign, log_det_covariance = np.linalg.slogdet(covariance_matrix)
    #print(log_det_covariance)

    identity_matrix = np.identity(covariance_matrix.shape[0])
    #print(sign)#1
    #print(log_det_covariance)#-29521
    # log_det_covariance = np.log(np.linalg.det(covariance_matrix))
    # print(log_det_covariance)
    # 计算指数部分
    exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(covariance_matrix)), (x - mean))
    #exponent = -0.5 * np.dot(np.dot((x - mean), np.linalg.inv(identity_matrix)), (x - mean))

    #print(torch.sum((x-mean).abs()))
    #print(exponent)#-646
    constant = -1881.9861160031696

    # 计算对数概率密度
    #log_pdf_value = -0.5 * d * np.log(2 * np.pi) - 0.5 * log_det_covariance + exponent
    log_pdf_value = constant - 0.5 * log_det_covariance + exponent

    return log_pdf_value
    #return exponent
# def laplace_pdf_mp(point, miu, cov):
#     d = len(miu)
#     mp.dps = 50  # 设置所需的精度（有效数字位数），此处设置为50位
#     with mp.workdps(dps):  # 创建一个工作上下文，临时改变当前精度
#         covariance_matrix = mp.matrix(cov)
#         det_cov = mp.det(covariance_matrix)
#         inv_cov = mp.inv(covariance_matrix)
#         point_miu_diff = mp.matrix(point - miu)
#         exponent = -0.5 * mp.dot(point_miu_diff, inv_cov * point_miu_diff)
#         prefactor = 1 / (mp.power(2 * mp.pi, d / 2) * mp.sqrt(det_cov))
#         pdf_value = prefactor * mp.exp(exponent)
#         return mp.mpf(pdf_value)


def get_prob(mean, sigma, point):
    log_probs = []
    for i in range(10):
        miu = mean[i]
        cov = sigma[i]
        #cov = torch.tensor(np.diag(cov))
        log_prob = torch.tensor(laplace_log_pdf(point,miu,cov))
        #print(log_prob)
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)
    mean = log_probs.mean()
    std = log_probs.std()
    norm_log_probs = (log_probs-mean)/std

    return norm_log_probs

def get_gamma(probs, pc):
    gamma = []
    gamma_sum = torch.sum(probs * pc)
    for i in range(10):
        gamma_i = probs[i]*pc[i]/gamma_sum
        gamma.append(gamma_i)
    gamma = torch.stack(gamma)
    return gamma



def get_new_center(n_gamma, points):
    n_gamma = n_gamma.float()
    points = points.float()
    sigma_gamma_ni = torch.sum(n_gamma, 0)
    sigma_expanded = sigma_gamma_ni.unsqueeze(1)

    new_centers = torch.einsum('ni,nd->id', n_gamma, points)
    new_centers = new_centers / sigma_expanded
    # print(new_centers[0])
    # testpoint=[]
    # for i in range(5):
    #     testpoint_i = points[i]*n_gamma[i][0]
    #     testpoint.append(testpoint_i)
    # testpoint = torch.stack(testpoint)
    # print(testpoint.shape)
    # testp = torch.sum(testpoint,0)
    # print(testp)
    return new_centers

def get_new_sigma(new_centers, points, n_gamma):
    new_sigma = []
    sigma_gamma_ni = torch.sum(n_gamma,0)
    sigma_gamma_ni_expanded = sigma_gamma_ni.unsqueeze(1)
    for i in range(10):
        column = n_gamma[:,i].unsqueeze(1)
        new_sigma_i = ((new_centers[i]-points).pow(2)*column).sum(0)
        new_sigma.append(new_sigma_i)
    new_sigma = torch.stack(new_sigma)
    new_sigma = new_sigma / sigma_gamma_ni_expanded
    # diag_new_sigma = [torch.diag(new_sigma[i]) for i in range(new_sigma.shape[0])]
    # diag_new_sigma = torch.stack(diag_new_sigma)
    return new_sigma

def get_new_pc(n_gamma):
    new_pc = torch.sum(n_gamma, 0)/n_gamma.shape[0]
    return new_pc


def get_para(cv_dis,unseen_data,mean,sigma,pc):

    # #get original unseen ws
    # unseen_points =unseen_data
    # unseen_n_gamma = []
    # for i, point in enumerate(unseen_points):
    #     logprobs = get_prob(mean, sigma, point)
    #     probs = np.exp(logprobs)
    #     gamma = get_gamma(probs, pc)
    #     probs = probs / torch.sum(probs)
    #     print(probs)
    #     unseen_n_gamma.append(gamma)
    # unseen_n_gamma = torch.stack(unseen_n_gamma)

    #cv20 = cv_dis[:5]
    cv20 = cv_dis
    n_gamma = []

    #cv20 = cv20[3].unsqueeze(dim=0)
    for i, point in enumerate(cv20):
        # print(point)
        # print(mean.shape)
        # print(sigma.shape)
        logprobs = get_prob(mean, sigma, point)

        #print(logprobs)
        probs = np.exp(logprobs)
        #print(probs)
        probs = probs/torch.sum(probs)
        print(probs)

        gamma = get_gamma(probs,pc)
        #print(gamma)
        n_gamma.append(gamma)
    n_gamma = torch.stack(n_gamma)
    #print(n_gamma[0])



    new_centers = get_new_center(n_gamma,cv20)
    #print(new_centers.shape)

    new_sigma = get_new_sigma(new_centers, cv20, n_gamma)
    #print(new_sigma.shape)

    new_pc = get_new_pc(n_gamma)
    #print(new_pc)
    return n_gamma, new_centers, new_sigma, new_pc

#print(len(cv_dis))
old_pc = torch.full((10,), fill_value=0.1)
#print(cv_dis[0])
#print(gauss_sigma.shape)
gauss_sigma = torch.stack([torch.eye(2048)]*10)
diagonal_vectors = [torch.diag(gauss_sigma[i]) for i in range(gauss_sigma.size(0))]
flatten_sigma = torch.stack(diagonal_vectors)
print(flatten_sigma.shape)

#print(gauss_sigma.shape)
#print(-0.5 * 2048 * np.log(2 * np.pi))
print(gauss_mean)
ng,nc,ns,npc = get_para(cv_dis,unseen_points,gauss_mean,gauss_sigma,old_pc)
print(ng.shape)
print(nc.shape)
print(ns.shape)
print(npc.shape)
nc = 0.1*nc + 0.9*gauss_mean
ns = 0.1*ns + 0.9*flatten_sigma
npc = 0.1*npc + 0.9*old_pc
traindate = '0429'
torch.save(ng,f'basis/{traindate}/original_weights_2024{traindate}.pth')
#torch.save(nu,'basis/original_unseen_weights_20240429.pth')
torch.save(nc,f'basis/{traindate}/original_centers_2024{traindate}.pth')
torch.save(ns,f'basis/{traindate}/original_sigma_2024{traindate}.pth')
torch.save(npc,f'basis/{traindate}/original_pc_2024{traindate}.pth')
# prob = laplace_pdf_mp(point,miu,cov)
# print(prob)
# vis
# imgs = []
# for i in np.array(sorted(np.argmin(dis_mat_l1, axis=-1))):
#     img = cv2.imread('data/data_221_S128F80_Base50_2x2_format/{:04}.png'.format(i))
#     imgs.append(img)

# plt.figure(dpi=200)
# plt.imshow(np.hstack(imgs))
