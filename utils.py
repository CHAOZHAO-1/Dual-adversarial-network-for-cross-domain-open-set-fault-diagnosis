#author:zhaochao time:2021/5/18

import torch as t
import torch.nn.functional as F
import numpy as np
import  random





def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / t.norm(x2, dim=1)
    return sim




def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = t.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd



def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * t.log(predict_prob + epsilon)
    return t.sum(instance_level_weight * ce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    mask = predict_prob.ge(0.000001)  # 逐元素比较
    mask_out = t.masked_select(predict_prob, mask)#


    entropy =-mask_out * t.log(mask_out)


#
    return t.sum(instance_level_weight * entropy * class_level_weight) / float(N)

#
def normalization(input):
    kethe=0.0000000001
    output=(input-min(input))/(max(input)-min(input)+kethe)

    return output

