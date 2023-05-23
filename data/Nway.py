from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
import random
from data.Dataloader import *

torch.manual_seed(666)
random.seed(666)


def createTask(data_list, n_way, k_shot, k_query):
    #  take 5 way 1 shot as example: 5 * 1
    nums = len(data_list)
    assert n_way <= nums
    cls_list = list(range(nums))
    selected_cls = np.random.choice(cls_list, n_way, False)  # False表示不放回抽样
    sum = k_shot + k_query
    support_insts = []
    query_insts = []

    for cur_cls in selected_cls:
        insts = []
        insts = random.sample(data_list[cur_cls], sum)
        support_insts.extend(insts[:k_shot])
        query_insts.extend(insts[k_shot:])

    return support_insts, query_insts


def createBatch(data, config):
    batches = []
    for batchNum in range(config.episodes):
        onebatch = []
        for i in range(config.batchsz):
            task = createTask(data, config.n_way, config.k_shot, config.k_query)
            onebatch.append(task)
        batches.append(onebatch)
    return batches


def createDevBatch(data, config):
    """
    为meta fine-tune 构建一个batch, 包含N个task
    其中support部分用来微调, query部分用来测试模型的表现
    """
    batches = []
    data_list = [data]
    # 任务的数量用验证集的样本数量除以（k_shot +　k_query）取整，差不多能够覆盖整个验证集
    task_num = len(data)//(config.k_shot + config.k_query) + 1
    for i in range(task_num):
        task = createTask(data_list, 1, config.k_shot, config.k_query)
        batches.append(task)
    return batches
