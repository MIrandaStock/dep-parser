class Dependency:  # 以词为单位

    def __init__(self, id, form, tag, head, rel):
        self.id = id  # 该词在该句话的id
        self.org_form = form  # 该词语
        self.form = form.lower()
        self.tag = tag  # 该词词性
        self.head = head  # 该词头节点
        self.rel = rel  # 该词的标签

    def __str__(self):
        values = [
            str(self.id), self.org_form, "_", self.tag, "_", "_",
            str(self.head), self.rel, "_", "_"
        ]
        return '\t'.join(values)

    @property
    def pseudo(self):
        return self.id == 0 or self.form == '<eos>'


class DepTree:  # 以句子为单位，判断是否为投影树

    def __init__(self, sentence):
        self.words = list(sentence)
        self.start = 1
        if sentence[0].id == 1:
            self.start = 0
        elif sentence[0].id == 0:
            self.start = 1
        else:
            self.start = len(self.words)

    def isProj(self):
        n = len(self.words)
        words = self.words
        if self.start > 1:
            return False
        if self.start == 0:
            words = [None] + words
        for i in range(1, n):
            hi = words[i].head
            for j in range(i + 1, hi):
                hj = words[j].head
                if (hj - hi) * (hj - i) > 0:
                    return False
        return True


def evalDepTree(gold, predict):
    """将预测的依存树和gold做比较, 计算准确率"""

    # 过滤掉一些不需要参与计算的标点符号
    # PUNCT_TAGS = ['``', "''", ':', ',', '.', 'PU']
    PUNCT_TAGS = []
    ignore_tags = set(PUNCT_TAGS)

    # 判断gold tree和predict tree的id是从0还是从1开始
    start_g = 1 if gold[0].id == 0 else 0
    start_p = 1 if predict[0].id == 0 else 0

    glength = len(gold) - start_g
    plength = len(predict) - start_p

    if glength != plength:
        raise Exception('gold length does not match predict length.')

    arc_total, arc_correct, label_total, label_correct = 0, 0, 0, 0
    for idx in range(glength):
        if gold[start_g + idx].pseudo or gold[
                start_g + idx].tag in ignore_tags or gold[start_g + idx].head == -1:
            continue
        arc_total += 1
        label_total += 1
        if gold[start_g + idx].head == predict[start_p + idx].head:
            arc_correct += 1
            # 在head预测正确的情况下, 判断rel标签是否预测正确
            if gold[start_g + idx].rel == predict[start_p + idx].rel:
                label_correct += 1

    return arc_total, arc_correct, label_total, label_correct


def readDepTree(file, vocab=None):
    """ 读取句子, 并构造dependency结构存储每一个分词, 每一个sentence是由多个dependency构成的List

        该方法共使用两次, 第一次统计word、rel、tag的数量, 构建词表, 第二次构造句子, 在每个句子前加上伪根root.
    """
    proj = 0
    total = 0
    min_count = 1
    if vocab is None:
        min_count = 0
        sentence = []
    else:
        sentence = [
            Dependency(0, vocab._root_form, vocab._root, -1, vocab._root)
        ]
    for line in file:
        tok = line.strip().split('\t')
        # 句子结束后，重置sentence
        if not tok or line.strip() == '' or line.strip().startswith('#'):
            if len(sentence) > min_count:  # sententce至少有一个词
                if DepTree(sentence).isProj():
                    proj += 1
                total += 1
                yield sentence
            if vocab is None:
                sentence = []
            else:
                sentence = [Dependency(0, vocab._root_form, vocab._root, -1, vocab._root)]
        # 补全数据: 没有haed标签的, 设为-1
        elif len(tok) == 10:
            if tok[6] == '_':
                tok[6] = '-1'
            try:
                sentence.append(
                    Dependency(int(tok[0]), tok[1], tok[3], int(tok[6]),
                               tok[7]))
            except Exception:
                pass
        else:
            pass

    # if len(sentence) > min_count:
    #     if DepTree(sentence).isProj():
    #         proj += 1
    #     total += 1
    #     yield sentence

    print("Total num: ", total)
    print("Proj num: ", proj)


def writeDepTree(filename, sentences):
    with open(filename, 'w') as file:
        for sentence in sentences:
            for entry in sentence:
                if not entry.pseudo:
                    file.write(str(entry) + '\n')
            file.write('\n')


def printDepTree(output, sentence, gold=None):
    if gold is None:
        for entry in sentence:
            if not entry.pseudo:
                output.write(str(entry) + '\n')
        output.write('\n')
    else:
        start_g = 1 if gold[0].id == 0 else 0
        start_p = 1 if sentence[0].id == 0 else 1

        glength = len(gold) - start_g
        plength = len(sentence) - start_p

        if glength != plength:
            raise Exception('gold length does not match predict length.')

        for idx in range(glength):
            if gold[start_g + idx].pseudo:
                continue
            values = [
                str(gold[start_g + idx].id), gold[start_g + idx].org_form, "_",
                gold[start_g + idx].tag, "_", "_",
                str(sentence[start_p + idx].head), sentence[start_p + idx].rel,
                "_", "_"
            ]
            output.write('\t'.join(values) + '\n')

        output.write('\n')


# 这里的trees就是sentences构成的batch
def batch_variable_depTree(trees, pre_heads, pre_rels, lengths, vocab):
    """ 用gold句子中的分词id, org_form, tag和预测的head, rel构建预测的依存树 """
    for tree, head, rel, length in zip(trees, pre_heads, pre_rels, lengths):
        assert len(tree) == len(head) == len(rel) == length
        sentence = []
        for dep_id, dep in enumerate(tree):
            sentence.append(
                Dependency(
                    dep_id,
                    dep.org_form,
                    dep.tag,
                    head[dep_id],
                    vocab.id2rel(rel[dep_id])
                )
            )
        yield sentence
