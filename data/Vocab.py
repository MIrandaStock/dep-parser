from collections import Counter
from data.Dependency import readDepTree
import numpy as np
import fileinput
import sys
sys.path.extend(["../../../", "../../", "../", "./"])


def reverse(x):
    return dict(zip(x, range(len(x))))


class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2

    def __init__(
            self,
            word_counter,
            tag_counter,
            rel_counter,
            relroot='root',
            min_occur_count=2):
        self._root = relroot
        self._root_form = '<' + relroot.lower() + '>'
        self._id2word = ['<pad>', self._root_form, '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2extword = ['<pad>', self._root_form, '<unk>']
        self._id2tag = ['<pad>', relroot]
        self._id2rel = ['<pad>', relroot, 'pred', 'subj-in', 'frag', 'punc', 'repet', 'exp']
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for tag, count in tag_counter.most_common():
            if tag != relroot and tag not in self._id2tag:
                self._id2tag.append(tag)

        for rel, count in rel_counter.most_common():
            if rel != relroot and rel not in self._id2rel:
                self._id2rel.append(rel)

        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: POS tags dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #tags %d, #rels %d" %
              (self.vocab_size, self.tag_size, self.rel_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            # next(f)  # 跳过第一行的说明，从第二行开始读取训练好的向量
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1  # 减去开头的(编号或者单词)的长度
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            # next(f)
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector  # UNK累加了所有词的向量
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count  # UNK累加了所有词的向量后取平均
        embeddings = embeddings / np.std(embeddings)

        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        def reverse(x):
            return dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")
        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        # return self._rel2id[xs]
        return self._rel2id.get(xs, self.UNK)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    # def tag2id(self, xs):
    #   if isinstance(xs, list):
    #       return [self._tag2id.get(x) for x in xs]
    #   return self._tag2id.get(xs)

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)

    @property
    def print_rels(self):
        return self._id2rel

    @property
    def print_tags(self):
        return self._id2tag


def creatVocab(corpusFile: list, min_occur_count: int):
    print("Start to create vocab and vec.")
    word_counter = Counter()
    tag_counter = Counter()
    rel_counter = Counter()
    root = ''
    max_len = 0
    with fileinput.input(corpusFile) as infile:
        # with open(corpusFile, 'r', encoding='utf-8') as infile:
        for sentence in readDepTree(infile):
            sen_len = len(sentence)
            if sen_len > max_len:
                max_len = sen_len
            for dep in sentence:
                word_counter[dep.form] += 1
                tag_counter[dep.tag] += 1
                if dep.head == -1:  # 补全的数据
                    continue
                if dep.head != 0:
                    rel_counter[dep.rel] += 1
                elif root == '':
                    root = dep.rel
                    rel_counter[dep.rel] += 1
                elif root != dep.rel:
                    print('root = ' + root + ', rel for root = ' + dep.rel)
    print("Finish to create vocab and vec.")
    return Vocab(word_counter, tag_counter, rel_counter, root, min_occur_count), max_len
