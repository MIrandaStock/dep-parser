import torch
from torch.utils.data import Dataset
# from transformers import AutoTokenizer
import fileinput
from torch.autograd import Variable
import numpy as np
import random

from data.Dependency import readDepTree
from data.Vocab import Vocab


def sentence2id(sentence, vocab):
    result = []
    for dep in sentence:
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        if head == -1:
            relid = -1
        elif head >= 0:
            relid = vocab.rel2id(dep.rel)
        result.append([tagid, head, relid])
    return result


def sentence2token_list(sentences, max_sen_length):
    token_lists = []
    for sentence in sentences:
        padding_length = max_sen_length - len(sentence)
        token_list = []
        for dep in sentence:
            token = dep.form
            token_list.append([token])
        token_list = token_list + ["[PAD]"] * padding_length
        token_lists.append(token_list)
    return token_lists


def batch_to_var_bert(batch_samples, vocab, need_raw_sentence=False):
    max_sen_length = len(max(batch_samples, key=len))
    batch_size = len(batch_samples)
    tags = Variable(torch.LongTensor(batch_size, max_sen_length).zero_(), requires_grad=False)
    masks = Variable(torch.LongTensor(batch_size, max_sen_length).zero_(), requires_grad=False)
    heads = Variable(torch.LongTensor(batch_size, max_sen_length).fill_(-1), requires_grad=False)
    rels = Variable(torch.LongTensor(batch_size, max_sen_length).fill_(-1), requires_grad=False)
    lengths = []

    for num, sentence in enumerate(batch_samples):
        sentence = sentence2id(sentence, vocab)
        length = len(sentence)
        lengths.append(length)
        for id, dep in enumerate(sentence):
            tags[num, id] = dep[0]
            masks[num, id] = 1
            heads[num, id] = dep[1]
            rels[num, id] = dep[2]
    token_lists = sentence2token_list(batch_samples, max_sen_length)

    if need_raw_sentence:
        return tags, heads, rels, lengths, masks, token_lists, batch_samples
    return tags, heads, rels, lengths, masks, token_lists
