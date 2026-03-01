import argparse
import random
import logging
import time
import setproctitle
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from data.preprocess import process
from torch.utils.data import DataLoader
from models.GLSurv import GLSurv
from models.Gene import GeneEncoder
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score,accuracy_score
from tensorboardX import SummaryWriter
from torch import nn
from ranger import Ranger
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import joblib
import copy
from scipy.stats import chi2
import math

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='UCSF', type=str)

parser.add_argument('--experiment', default='Survival', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--root', default='/home/UCSF-PDGM', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

parser.add_argument('--dataset', default='UCSF', type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='1', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=2000, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def cox_log_rank(hazardsdata,survtime_all,labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

class MultiTaskLossWrapper_OS(nn.Module):
    def __init__(self, ):
        super(MultiTaskLossWrapper_OS, self).__init__()
        self.vars = nn.Parameter(torch.tensor((1.0,1.0),requires_grad=True)) #1.0, 6.0
    def forward(self, loss1,loss2):
        lossd_1 = torch.sum(0.5 * loss1 / (self.vars[0] ** 2) + torch.log(self.vars[0]), -1)
        lossd_2 = torch.sum(0.5 * loss2 / (self.vars[1] ** 2) + torch.log(self.vars[1]), -1)

        loss = torch.mean(lossd_1+lossd_2)
        return loss
def CoxLoss(hazard_pred,survtime, censor):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(censor.device)#censor.device
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def get_changed_form_dict(rank_dict):
    """计算样本相邻epoch的排序波动，返回波动倒数（波动越小，值越大，可信度越高）"""
    fluct_dict = {}
    for name, ranks in rank_dict.items():
        if len(ranks) < 2:
            fluct = 1.0  # 不足2轮，默认波动为1
        else:
            last_two_ranks = ranks[-2:]
            fluct = abs(last_two_ranks[0] - last_two_ranks[1])
            if fluct == 0:
                fluct = 1.0  # 避免除0
        fluct_dict[name] = 1.0 / fluct  # p(t) = 1/|rank_t - rank_{t-1}|
    return fluct_dict

def update_egcm_confidence(confidence_dict, fluct_dict, lambda_=0.5):
    """更新动态动量累积置信度（DMAC）：τ(t) = λτ(t-1) + (1-λ)p(t)"""
    new_confidence = {}
    for name in confidence_dict:
        if name in fluct_dict:
            new_confidence[name] = lambda_ * confidence_dict[name] + (1 - lambda_) * fluct_dict[name]
        else:
            new_confidence[name] = confidence_dict[name]  # 无波动则保持原置信度
    return new_confidence

def calculate_dynamic_lambda(epoch, max_epoch, lambda_init=0.5, lambda_min=0.1, power=0.9):

    if max_epoch == 0:  # 防止除0错误
        return round(lambda_min, 8)
    # 完全复用原学习率的指数衰减核心项
    decay_term = np.power(1 - (epoch) / max_epoch, power)
    # 计算动态λ，确保不低于下限（双重保险）
    lambda_t = lambda_min + (lambda_init - lambda_min) * decay_term
    lambda_t = max(lambda_t, lambda_min)  # 防止数值误差导致λ低于0.1
    return round(lambda_t, 8)

def custom_sort(arr, mid):
    left_part = [x for x in arr if x < mid]
    right_part = [x for x in arr if x > mid]
    left_part = sorted(left_part, reverse=True)
    right_part = sorted(right_part)
    sorted_arr = left_part + right_part
    return sorted_arr, left_part, right_part


def get_all_ci_from_list(true_event_times, pred_risk_scores, event_status):
  
    ordered_time = true_event_times
    ordered_pred_time = [-x for x in pred_risk_scores]  # 取负是必须的，你记忆的完全正确
    ordered_observed = event_status
    return concordance_index(ordered_time, ordered_pred_time, ordered_observed)


def greedy_search(
        pseudo_sample_info, sample_names, idx, start, end
):
   
    best_cindex = 0.0
    best_value = None

    # 遍历候选时间（步长1.1，你的核心要求）
    for possible_value in np.arange(start, end, 1.1):
        # 临时复制样本信息，避免污染原数据
        temp_sample_info = copy.deepcopy(pseudo_sample_info)
        # 假设当前删失样本的时间为possible_value，状态改为1（未删失）
        temp_sample_info[sample_names[idx]]['time'] = possible_value
        temp_sample_info[sample_names[idx]]['status'] = 1

        # 提取计算C-index所需的全部数据（用真实值，保证准确）
        # 注意：你可以选择计算全样本C-index，或仅当前样本后的子集，按需调整
        all_times = [temp_sample_info[name]['time'] for name in sample_names]
        all_risks = [temp_sample_info[name]['pred_score'] for name in sample_names]
        all_status = [temp_sample_info[name]['status'] for name in sample_names]

        # 计算当前候选时间的C-index
        current_cindex = get_all_ci_from_list(all_times, all_risks, all_status)

        # 更新最优值（保留C-index最大的时间）
        if current_cindex > best_cindex:
            best_cindex = current_cindex
            best_value = possible_value

    return best_value


def generate_pseudo_labels(sample_info, confidence_dict, predicted_scores,
                           true_event_times, true_status, p_times, percent=0.3, lambda_=0.4, top_k=15):
    censored_samples = []
    for name, info in sample_info.items():
        # 初始化原始状态/时间（避免首次处理时无此字段）
        if 'original_status' not in info:
            info['original_status'] = info['status']  # 保存原始状态
            info['original_time'] = info['time']  # 保存原始时间
        # 筛选逻辑：原始删失 或 已生成伪标签
        if info['original_status'] == 0 or info.get('is_pseudo', False):
            censored_samples.append(name)

    if len(censored_samples) == 0:
        return sample_info, []

    censored_confidence = {name: confidence_dict[name] for name in censored_samples}
    sorted_censored = sorted(censored_confidence.items(), key=lambda x: x[1], reverse=True)
    top_num = int(len(sorted_censored) * percent)
    if top_num == 0:
        top_num = 1
    top_censored = [name for name, _ in sorted_censored[:top_num]]

    top_censored = [name for name in top_censored if name in p_times]
    print(f"筛选后，当前轮处理的高置信度删失样本数：{len(top_censored)}")

    if not top_censored:
        return sample_info, []

    pseudo_sample_info = copy.deepcopy(sample_info)
    predicted_scores_np = np.array(predicted_scores)
    true_event_times_np = np.array(true_event_times)
    true_status_np = np.array(true_status)
    sample_names = list(pseudo_sample_info.keys())
    pseudo_log = []

    for censored_name in top_censored:
        if censored_name not in sample_names:
            continue
        i = sample_names.index(censored_name)
        current_info = pseudo_sample_info[censored_name]


        target_score = current_info['pred_score']
        # 取原始删失时间，而非已修改的伪时间
        target_censor_time = current_info['original_time']
        sample_confidence = confidence_dict[censored_name]
        p_time = p_times[censored_name]

        if current_info['original_status'] != 0 and not current_info.get('is_pseudo', False):
            continue

        all_risk_scores = np.array([pseudo_sample_info[name]['pred_score'] for name in sample_names])
        risk_distance = np.abs(all_risk_scores - target_score)
        risk_distance[i] = 999999
        nearest_indice = np.argsort(risk_distance)[:top_k]

        valid_times = []
        for j in nearest_indice:
            neighbor_name = sample_names[j]
            neighbor_info = pseudo_sample_info[neighbor_name]
            # 筛选条件：邻居是未删失（原始状态=1） + 生存时间>原始删失时间
            if neighbor_info['original_status'] == 1 and neighbor_info['original_time'] > target_censor_time:
                valid_times.append(neighbor_info['original_time'])

        if len(valid_times) == 0:
            continue
        max_round = np.mean(valid_times)

        if max_round > target_censor_time:
            start = int(target_censor_time)
            end = int(max_round)
            best_time = greedy_search(
                pseudo_sample_info, sample_names, i, start, end
            )
            if best_time is None:
                continue

            pseudo_time = math.sqrt(p_time * best_time)
            pseudo_status = 1

            log_item = {
                'sample_name': censored_name,
                'original_time': target_censor_time,
                'original_status': 0,
                'pseudo_time': round(pseudo_time, 2),
                'pseudo_status': pseudo_status,
                'confidence': round(sample_confidence, 4),
                'p_time': p_time,
                'risk_similar_k': top_k,
                'valid_neighbor_times': [round(t, 2) for t in valid_times],
                'valid_neighbor_avg_time': round(max_round, 2),  # 原代码写错成similar_sample_avg_time
                'greedy_search_best_time': round(best_time, 2),
                'greedy_search_range': f"[{start}, {end}] (步长1.1)",
                'pseudo_time_calc': f"√({p_time} × {round(best_time, 2)}) = {round(pseudo_time, 2)}"
            }
            pseudo_log.append(log_item)

            # 更新伪标签（保留原始状态/时间，只更新伪时间和is_pseudo）
            pseudo_sample_info[censored_name]['status'] = pseudo_status
            pseudo_sample_info[censored_name]['time'] = pseudo_time
            pseudo_sample_info[censored_name]['is_pseudo'] = True

    return pseudo_sample_info, pseudo_log

def main_worker():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    model = GLSurv().float().cuda()
    gene= GeneEncoder().float().cuda()
    MTL = MultiTaskLossWrapper_OS().float().cuda()
    nets = {
        'model': model.cuda(),
        'gene':gene.cuda(),
        'mtl': MTL.cuda(),
    }

    param = [p for v in nets.values() for p in list(v.parameters()) if
             p.requires_grad]  # Only parameters that require gradients
    optimizer = Ranger(
        param,  # 网络的可训练参数
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay  # 权重衰减
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.tensor([1.26, 1.0], device=device)

    criterion1 = CoxLoss
    criterion2 = FocalLoss(weight=weight)
    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    resume = ''
    writer = SummaryWriter()
    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    semi_save_dir = "/home/model"
    os.makedirs(semi_save_dir, exist_ok=True)
    train_statue_risk_time_path = os.path.join(semi_save_dir, "train_statue_risk_time.pkl")

    sample_info = {}
    # 各epoch排序记录：{name: [epoch0_rank, epoch1_rank, ...]}
    rank_history = {}
    egcm_confidence = {}
    # 伪标签启动轮次
    pseudo_start_epoch = 100
    # 高可信度样本比例
    percent_high_conf = 0.2

    lambda_init = 0.5  # λ初始值
    lambda_min = 0.1  # λ下限
    lambda_power = 0.9

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = process(train_list, train_root, args.mode)
    logging.info('Samples for train = {}'.format(len(train_set)))
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    val_root = os.path.join(args.root, args.valid_dir)
    val_set = process(val_list, val_root, 'valid')
    logging.info('Samples for val = {}'.format(len(val_set)))
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    print(len(val_set))
    torch.set_grad_enabled(True)
    # 原有模型保存路径逻辑
    if args.local_rank == 0:
        roott = "/home/semi"
        checkpoint_dir = os.path.join(roott, 'checkpoint', args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    labels = [0, 1]
    risk_names = ['low', 'high']
    best_val_c_index = -1

    # ======================== 训练循环（集成伪标签） ========================
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        print(optimizer.param_groups[0]['lr'])
        lambda_t = calculate_dynamic_lambda(
            epoch=epoch,
            max_epoch=args.end_epoch,
            lambda_init=lambda_init,
            lambda_min=lambda_min,
            power=lambda_power
        )
        print(f"当前Epoch {epoch} - EGCM动态λ: {lambda_t:.8f}")
        train_loss = []
        Closs=[]
        Gloss=[]
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        nets['model'].train()
        nets['gene'].train()
        nets['mtl'].train()
        group_pred_scores = []
        group_preds = []
        group_trues = []
        predicted_score_train = []
        true_event_time_train = []
        true_status_train = []
        p_time_train = {}
        sample_names_train = []
        optimizer.zero_grad()

        # ======================== 第一步：加载/更新样本基础信息字典 ========================
        # 加载历史字典（若存在）
        if os.path.exists(train_statue_risk_time_path):
            sample_info = joblib.load(train_statue_risk_time_path)
        # 初始化新样本的置信度（若有）

        # ======================== 第二步：训练集前向传播（含伪标签替换） ========================
        for i, data in enumerate(train_loader):
            x, gene, time, status, group, p_time, name = data
            # 保存p_time和样本名
            for n, pt in zip(name, p_time.cpu().numpy()):
                p_time_train[n] = pt
            sample_names_train.extend(name)

            if epoch >= pseudo_start_epoch:
                pseudo_time = []
                pseudo_status = []
                for n, t, s in zip(name, time, status):
                    if n in sample_info and sample_info[n].get('is_pseudo', False):
                        # 使用伪标签
                        pseudo_time.append(sample_info[n]['time'])
                        pseudo_status.append(sample_info[n]['status'])
                    else:
                        # 保留原始标签
                        pseudo_time.append(t.item())
                        pseudo_status.append(s.item())
                # 转换为tensor
                time = torch.tensor(pseudo_time, device=time.device)
                status = torch.tensor(pseudo_status, device=status.device)

            # 原有前向传播逻辑
            x = x.cuda(args.local_rank, non_blocking=True).float()
            gene = gene.cuda(args.local_rank, non_blocking=True)
            time = time.cuda(args.local_rank, non_blocking=True)
            status = status.cuda(args.local_rank, non_blocking=True)
            group = group.cuda(args.local_rank, non_blocking=True)
            output_risk, bridge = nets['model'](x)

            output_risk_flat = output_risk.detach().cpu().numpy().flatten()
            if len(output_risk_flat) != len(name):
                print(f"⚠️ 警告：第{i}批次分数数量({len(output_risk_flat)})≠样本名数量({len(name)})，跳过该批次数据收集！")
                continue  # 丢弃异常批次，避免长度错位

            # 收集训练数据
            predicted_score_train += list(output_risk.detach().cpu().numpy().flatten())
            true_event_time_train += list(time.detach().cpu().numpy())
            true_status_train += list(status.detach().cpu().numpy())
            # 保存样本预测风险到字典
            for n, score in zip(name, output_risk.detach().cpu().numpy().flatten()):
                if n not in sample_info:
                    sample_info[n] = {
                        'time': time.detach().cpu().numpy()[list(name).index(n)],
                        'status': status.detach().cpu().numpy()[list(name).index(n)],
                        'pred_score': score,
                        'rank': 0,
                        'is_pseudo': False
                    }
                else:
                    sample_info[n]['pred_score'] = score

            output_group = nets['gene'](gene, bridge)
            surv_loss = criterion1(output_risk.flatten(), time, status)
            group_pred_score = F.softmax(output_group, dim=1)[:, 1].detach().cpu().numpy()
            group_pred = torch.argmax(output_group, dim=1).detach().cpu().numpy()
            group_true = group.detach().cpu().numpy()
            group_pred_scores.append(group_pred_score)
            group_preds.append(group_pred)
            group_trues.append(group_true)
            group_loss = criterion2(output_group, group)
            loss = nets['mtl'](surv_loss, group_loss)
            loss.backward()
            train_loss.append(loss.item())
            Closs.append(surv_loss.item())
            Gloss.append(group_loss.item())
            optimizer.step()
            optimizer.zero_grad()
        for name in sample_names_train:
            if name not in egcm_confidence:
                dmac_confidence[name] = 0.0  # τ(0)=0
            if name not in rank_history:
                rank_history[name] = []
                print(f"初始化样本{name}的rank_history为空列表")

        # 按预测风险排序（风险越高，排名越小）
        score_name_pairs = list(zip(predicted_score_train, sample_names_train))
        score_name_pairs_sorted = sorted(score_name_pairs, key=lambda x: x[0], reverse=True)
        rank_dict = {name: idx + 1 for idx, (_, name) in enumerate(score_name_pairs_sorted)}  # 排名从1开始

        train_C_index = concordance_index(np.array(true_event_time_train), -np.array(predicted_score_train),
                                          np.array(true_status_train))
        print("第" + str(epoch) + "个epoch的" + "c-index is " + str(train_C_index))
        pvalue = cox_log_rank(np.array(predicted_score_train), np.array(true_event_time_train),
                              np.array(true_status_train))
        print("第" + str(epoch) + "个epoch的" + "pvalue is " + str(pvalue))
        # 更新排名历史

        for name in sample_names_train:
            if name in rank_dict:
                print(f"样本{name} - 当前rank_history: {rank_history[name]}")
                print(f"样本{name} - 本次排名: {rank_dict[name]}")
                rank_history[name].append(rank_dict[name])
                sample_info[name]['rank'] = rank_dict[name]


        if epoch >= pseudo_start_epoch and epoch % 10 == 0:
            # 1. 计算当前epoch的排序波动
            fluct_dict = get_changed_form_dict(rank_history)
            # 2. 更新EGCM置信度
            egcm_confidence = update_egcm_confidence(egcm_confidence, fluct_dict, lambda_t)
            # 3. 生成伪标签（接收返回的日志）
            sample_info, pseudo_log = generate_pseudo_labels(
                sample_info, egcm_confidence,
                np.array(predicted_score_train),
                np.array(true_event_time_train),
                np.array(true_status_train),
                p_time_train,
                percent=percent_high_conf,
                lambda_=lambda_t
            )

            if len(pseudo_log) >= 1:
                print(f"\n===== Epoch {epoch} 伪标签修改记录（共{len(pseudo_log)}个样本）=====")
                for idx, log in enumerate(pseudo_log):
                    print(f"[{idx + 1}] 样本名: {log['sample_name']}")
                    print(f"  原始信息：时间={log['original_time']}, 状态={log['original_status']}(存活)")
                    print(f"  伪标签信息：时间={log['pseudo_time']}, 状态={log['pseudo_status']}(死亡)")
                    print(f"  置信度：{log['confidence']} | 伪时间计算：{log['pseudo_time_calc']}")
                    print("-" * 60)
            else:
                print(f"\n===== Epoch {epoch} 无伪标签修改（无满足条件的存活样本）=====")

            log_file_path = "/home/pseudo_labels_log.txt"
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            # 追加写入日志
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Epoch {epoch} - 伪标签修改记录 (修改样本数：{len(pseudo_log)})\n")
                f.write(f"{'=' * 80}\n")
                if len(pseudo_log) > 0:
                    for log in pseudo_log:
                        f.write(f"样本名：{log['sample_name']}\n")
                        f.write(f"  原始信息：时间={log['original_time']} | 状态={log['original_status']}(存活)\n")
                        f.write(f"  伪标签信息：时间={log['pseudo_time']} | 状态={log['pseudo_status']}(死亡)\n")
                        f.write(f"  核心参数：置信度={log['confidence']} | K近邻数={log['risk_similar_k']}\n")
                        f.write(f"  基因层面p_time：{log['p_time']}\n")
                        # 补全K近邻相关字段
                        f.write(f"  K近邻结果：有效邻居时间={log['valid_neighbor_times']}\n")
                        f.write(f"  K近邻平均时间：{log['valid_neighbor_avg_time']}\n")
                        f.write(
                            f"  贪心搜索：范围={log['greedy_search_range']} | 最优best_time={log['greedy_search_best_time']}\n")
                        f.write(f"  伪时间计算：{log['pseudo_time_calc']}\n")
                        f.write(f"-" * 60 + "\n")
                else:
                    f.write("本次无伪标签修改（无满足条件的存活样本）\n")
        else:

            if epoch > 0:
                fluct_dict = get_changed_form_dict(rank_history)
                egcm_confidence = update_egcm_confidence(egcm_confidence, fluct_dict, lambda_t)

        joblib.dump(sample_info, train_statue_risk_time_path)
        logging.info(f"样本信息字典已保存到: {train_statue_risk_time_path}")

        print("第" + str(epoch) + "个epoch的" + "c-index is " + str(train_C_index))
        print("第" + str(epoch) + "个epoch的" + "pvalue is " + str(pvalue))
        print('总损失:', sum(train_loss) / len(train_loss))
        print('生存损失:', sum(Closs) / len(Closs))
        print('基因分类损失:', sum(Gloss) / len(Gloss))
        group_trues = np.concatenate(group_trues).tolist()
        group_preds = np.concatenate(group_preds).tolist()
        group_pred_scores = np.concatenate(group_pred_scores).tolist()
        print('group真实的结果')
        print(group_trues)
        print('group预测的结果')
        print(group_preds)
        print('生存风险的结果')
        show_score_train = [float(score) for score in predicted_score_train]
        print(show_score_train)
        results_group_train = evalution_metirc_boostrap(
            y_true=group_trues,
            y_pred_score=group_pred_scores,
            y_pred=group_preds,
            labels=labels,
            target_names=risk_names
        )

        with torch.no_grad():
                nets['model'].eval()
                nets['gene'].eval()
                nets['mtl'].eval()
                predicted_score_val = []
                true_event_time_val = []
                true_status_val = []
                for i, data in enumerate(val_loader):
                    x, time, status = data
                    x = x.cuda(args.local_rank, non_blocking=True).float()
                    time = time.cuda(args.local_rank, non_blocking=True)
                    status = status.cuda(args.local_rank, non_blocking=True)
                    output_risk, bridge = nets['model'](x)
                    predicted_score_val += list(output_risk.detach().cpu().numpy().flatten())
                    true_event_time_val += list(time.detach().cpu().numpy())
                    true_status_val += list(status.detach().cpu().numpy())

                val_C_index = concordance_index(np.array(true_event_time_val), -np.array(predicted_score_val),
                                                np.array(true_status_val))
                print("第" + str(epoch) + "个epoch的验证集" + "c-index is " + str(val_C_index))
                val_pvalue = cox_log_rank(np.array(predicted_score_val), np.array(true_event_time_val),
                                          np.array(true_status_val))
                print("第" + str(epoch) + "个epoch的验证集" + "pvalue is " + str(val_pvalue))
                if val_C_index > best_val_c_index:
                            best_val_c_index = val_C_index
                            model_filename = f"best_val_model_epoch_{epoch}.pth"
                            model_path = os.path.join(checkpoint_dir, model_filename)
                            torch.save({
                                'epoch': epoch,
                                'model': nets['model'].state_dict(),
                                'gene': nets['gene'].state_dict(),
                                'mtl': nets['mtl'].state_dict(),
                            }, model_path)
                            print(f"最佳验证集模型已保存到: {model_path}")



from sklearn.utils import resample
def bootstrap_c_index(true_event_times, predicted_scores, true_statuses, n_bootstrap=1000):
    c_indexes = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(true_event_times))) # 生成bootstrap样本的索引
        bs_event_times = true_event_times[indices]
        bs_predicted_scores = predicted_scores[indices]
        bs_statuses = true_statuses[indices]

        c_index_value = concordance_index(bs_event_times, -bs_predicted_scores, bs_statuses)
        c_indexes.append(c_index_value)

    # 计算置信区间
    c_indexes = np.array(c_indexes)

    lower_bound = np.percentile(c_indexes, 2.5)  # 2.5%分位数
    upper_bound = np.percentile(c_indexes, 97.5)  # 97.5%分位数

    return lower_bound, upper_bound
def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def initialize_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def evalution_metirc_boostrap(y_true, y_pred_score, y_pred, labels, target_names):
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)
    y_pred = np.array(y_pred)
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

    auc_ = roc_auc_score(y_true, y_pred_score)
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    accuracy_ = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity_ = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity_ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    F1_score_ = f1_score(y_true, y_pred, labels=labels, pos_label=1)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_AUC = []
    bootstrapped_ACC = []
    bootstrapped_SEN = []
    bootstrapped_SPE = []
    bootstrapped_F1 = []
    rng = np.random.RandomState(rng_seed)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_score), len(y_pred_score))
        if len(np.unique(y_true[indices.astype(int)])) < 2:
            # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
            continue
        auc = roc_auc_score(y_true[indices], y_pred_score[indices])
        bootstrapped_AUC.append(auc)

        confusion = confusion_matrix(y_true[indices], y_pred[indices])
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        F1_score = f1_score(y_true[indices], y_pred[indices], labels=labels, pos_label=1)

        bootstrapped_ACC.append(accuracy)
        bootstrapped_SPE.append(specificity)
        bootstrapped_SEN.append(sensitivity)
        bootstrapped_F1.append(F1_score)

    sorted_AUC = np.array(bootstrapped_AUC)
    sorted_AUC.sort()
    sorted_ACC = np.array(bootstrapped_ACC)
    sorted_ACC.sort()
    sorted_SPE = np.array(bootstrapped_SPE)
    sorted_SPE.sort()
    sorted_SEN = np.array(bootstrapped_SEN)
    sorted_SEN.sort()
    sorted_F1 = np.array(bootstrapped_F1)
    sorted_F1.sort()

    results = {
        'AUC': (auc_, sorted_AUC[int(0.05 * len(sorted_AUC))], sorted_AUC[int(0.95 * len(sorted_AUC))]),
        'Accuracy': (accuracy_, sorted_ACC[int(0.05 * len(sorted_ACC))], sorted_ACC[int(0.95 * len(sorted_ACC))]),
        'Specificity': (specificity_, sorted_SPE[int(0.05 * len(sorted_SPE))], sorted_SPE[int(0.95 * len(sorted_SPE))]),
        'Sensitivity': (sensitivity_, sorted_SEN[int(0.05 * len(sorted_SEN))], sorted_SEN[int(0.95 * len(sorted_SEN))]),
        'F1_score': (F1_score_, sorted_F1[int(0.05 * len(sorted_F1))], sorted_F1[int(0.95 * len(sorted_F1))])
    }

    print("Confidence interval for the AUC: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['AUC']))
    print("Confidence interval for the Accuracy: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Accuracy']))
    print("Confidence interval for the Specificity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Specificity']))
    print("Confidence interval for the Sensitivity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Sensitivity']))
    print("Confidence interval for the F1_score: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['F1_score']))
    return results



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
