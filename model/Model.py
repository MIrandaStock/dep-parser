import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from model.Layer import Biaffine, MyLSTM, NonLinear, CNN
from data.BertEmbedding import BertEmbedding
from data.Dataset import PositionalEncoding


def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size,
                                          seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size,
                                        seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim=2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim =2i+1
    position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
    return position_enc


class ParserModel(nn.Module):

    def __init__(self, vocab, config, pretrained_embedding):
        super(ParserModel, self).__init__()
        self.config = config
        # nn.Embedding 默认随机初始化为 标准正态分布N(0,1)的张量Tensor
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        # word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        # np.random.randn 生成[0,1)区间内一个或一组数值，服从均匀分布
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.mlp_arc_dep = NonLinear(
            input_size=2 * config.lstm_hiddens,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1)
        )
        self.mlp_arc_head = NonLinear(
            input_size=2 * config.lstm_hiddens,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks, sentences, positions):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond


class BertParserModel(nn.Module):

    def __init__(self, vocab, config):
        super(BertParserModel, self).__init__()
        self.config = config
        self.bert_embed = BertEmbedding()
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))
        # np.random.randn 生成[0,1)区间内一个或一组数值，服从均匀分布
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        # bert词向量后接一层FC
        self.fc = nn.Linear(in_features=768, out_features=config.word_dims, bias=False)

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.mlp_arc_dep = NonLinear(input_size=2 * config.lstm_hiddens,
                                     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                     activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(input_size=2 * config.lstm_hiddens,
                                      hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, vocab.rel_size, bias=(True, True))

    def forward(self, words, tags, masks, sentences):
        # x = (batch size, sequence length, dimension of embedding)
        x_extword_embed = self.bert_embed.get_emb(sentences).to(self.config.device)
        x_extword_embed = self.fc(x_extword_embed)

        x_word_embed = self.word_embed(words)
        x_embed = x_extword_embed + x_word_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep,
                                                 self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head,
                                                  self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond


class AttentionParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(AttentionParserModel, self).__init__()
        self.config = config
        # nn.Embedding 默认随机初始化为 标准正态分布N(0,1)的张量Tensor
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        # word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        # np.random.randn 生成[0,1)区间内一个或一组数值，服从均匀分布
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.multi_atten = nn.MultiheadAttention(
            embed_dim=2 * config.lstm_hiddens,  # 输入的词向量长度
            num_heads=config.num_heads_atten,  # MultiheadAttention的head的数量
            dropout=0.1,
            batch_first=True,
        )

        self.mlp_arc_dep = NonLinear(input_size=2 * config.lstm_hiddens,
                                     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                     activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(input_size=2 * config.lstm_hiddens,
                                      hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks, sentences, positions):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)
        # x_lexical = (batch size, sequence length, dimension of embedding*2)
        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        masks = masks == 0
        attn_output, _ = self.multi_atten(outputs, outputs, outputs, key_padding_mask=masks)

        if self.training:
            attn_output = drop_sequence_sharedmask(attn_output, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(attn_output)
        x_all_head = self.mlp_arc_head(attn_output)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond


class TransformerParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding, max_len):
        super(TransformerParserModel, self).__init__()
        self.config = config
        # self.position_enc = nn.Embedding(max_len, config.word_dims, padding_idx=0)
        # self.position_enc.weight.data = position_encoding_init(max_len, config.word_dims)
        # nn.Embedding 默认随机初始化为 标准正态分布N(0,1)的张量Tensor
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        # word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        # np.random.randn 生成[0,1)区间内一个或一组数值，服从均匀分布
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.pos_emb = PositionalEncoding(config.word_dims, dropout=0.33)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.word_dims + config.tag_dims,
                nhead=config.num_heads_transformer,
                dim_feedforward=config.dim_feedforward,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=config.transformer_layers,
        )

        self.mlp_arc_dep = NonLinear(input_size=config.word_dims + config.tag_dims,
                                     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                     activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(input_size=config.word_dims + config.tag_dims,
                                      hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks, sentences, positions):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)
        # x_lexical = (batch size, sequence length, dimension of embedding*2)
        x_embed = self.pos_emb(x_embed)
        x_tag_embed = self.pos_emb(x_tag_embed)
        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)
        # x_lexical = self.pos_emb(x_lexical)
        # x_lexical = x_embed + x_tag_embed
        # enc_input = torch.cat((x_lexical, pos_input), dim=2)

        masks = masks == 0
        # batch_first = True
        outputs = self.transformer(src=x_lexical, src_key_padding_mask=masks)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond


class EnsembelParserModel(nn.Module):
    "CNN+BiLSTM"

    def __init__(self, vocab, config, pretrained_embedding):
        super(EnsembelParserModel, self).__init__()
        self.config = config
        # nn.Embedding 默认随机初始化为 标准正态分布N(0,1)的张量Tensor
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        # word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        # np.random.randn 生成[0,1)区间内一个或一组数值，服从均匀分布
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.conv = CNN(
            emb_dim=config.word_dims + config.tag_dims,
            n_filters=config.cnn_hiddens,
            kernel_size=3,
            dropout=0.2,
            output_dim=config.cnn_hiddens,
        )

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )
        self.linear = nn.Linear(in_features=2 * config.lstm_hiddens + config.cnn_hiddens, out_features=2 * config.lstm_hiddens)

        self.mlp_arc_dep = NonLinear(input_size=2 * config.lstm_hiddens,
                                     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                     activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(input_size=2 * config.lstm_hiddens,
                                      hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks, sentences, positions):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)  # []
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        # CNN
        cnn_out = self.conv(x_lexical)

        # BiLSTM
        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        # Ensemble
        # outputs: [bz, seq len, 100+lstm_hiddens*2]
        outputs = torch.cat((cnn_out, outputs), dim=2)
        outputs = self.linear(outputs)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond
