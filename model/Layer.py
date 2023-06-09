import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def get_tensor_np(t):
    return t.detach().cpu().numpy()


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    # print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr * Q.dot(QTQmI) / (
                np.abs(Q2 + Q2.sum(axis=0, keepdims=True) +
                       Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print(
            'Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class NonLinear(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size,
                                out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(
                    type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.detach().copy_(torch.from_numpy(W))

        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.detach().copy_(torch.from_numpy(b))


class Biaffine(nn.Module):

    def __init__(self,
                 in1_features,
                 in2_features,
                 out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size),
                     dtype=np.float32)
        self.linear.weight.detach().copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        # input1(x_arc_dep): [bz, max_sen_len, 200]
        # input2(x_arc_head): [bz, max_sen_len, 200]
        # input1(x_rel_dep): [bz, max_sen_len, 100]
        # input2(x_rel_head): [bz, max_sen_len, 100]
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.detach().new(batch_size, len1, 1).fill_(1)
            # input1: [bz, max_sen_len, 200+1]
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.detach().new(batch_size, len2, 1).fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        # 对于rel: [bz, max_sen_len, rel_size*101]
        # 对于arc: [bz, max_sen_len, 201]
        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1,
                                              self.out_features)

        return biaffine


class LSTM(nn.LSTM):

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(
                        self.__getattr__(name)[self.hidden_size *
                                               i:self.hidden_size *
                                               (i + 1), :])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)


'''
class NewLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(NewLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            self.fcells[layer].reset_parameters()
            if self.bidirectional:
                self.bcells[layer].reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.detach().new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.detach().new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.detach().new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = NewLSTM._forward_rnn(cell=self.fcells[layer], \
                         input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = NewLSTM._forward_brnn(cell=self.bcells[layer], \
                       input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
        W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
        self.linear_ih.weight.detach().copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
        self.linear_hh.weight.detach().copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        b = np.zeros(4 * self.hidden_size, dtype=np.float32)
        b[self.hidden_size:2 * self.hidden_size] = -1.0
        self.linear_ih.bias.detach().copy_(torch.from_numpy(b))



    def forward(self, input, hx):
        if hx is None:
            batch_size = input.size(0)
            zero_hx = Variable(
                input.detach().new(batch_size, self.hidden_size).zero_())
            hx = (zero_hx, zero_hx)
        h, c = hx
        lstm_vector = self.linear_ih(input) + self.linear_hh(h)
        i, f, g, o = lstm_vector.chunk(chunks=4, dim=1)
        f = f + 1
        new_c = c*self.sigmoid(f) + self.sigmoid(i)*self.tanh(g)
        new_h = self.tanh(new_c) * self.sigmoid(o)
        return new_h, new_c
'''


class MyLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_first=False,
                 bidirectional=False,
                 dropout_in=0,
                 dropout_out=0):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(
                nn.LSTMCell(input_size=layer_input_size,
                            hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(
                    nn.LSTMCell(input_size=layer_input_size,
                                hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih,
                            self.fcells[layer].weight_hh,
                            self.fcells[layer].bias_ih,
                            self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih,
                                self.bcells[layer].weight_hh,
                                self.bcells[layer].bias_ih,
                                self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
            param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
            param_ih = self.__getattr__(param_ih_name)
            param_hh = self.__getattr__(param_hh_name)
            if layer == 0:
                W = orthonormal_initializer(self.hidden_size,
                                            self.hidden_size + self.input_size)
            else:
                W = orthonormal_initializer(
                    self.hidden_size, self.hidden_size + 2 * self.hidden_size)
            W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
            param_ih.detach().copy_(
                torch.from_numpy(np.concatenate([W_x] * 4, 0)))
            param_hh.detach().copy_(
                torch.from_numpy(np.concatenate([W_h] * 4, 0)))

            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(
                        self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(
                        self.hidden_size,
                        self.hidden_size + 2 * self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.detach().copy_(
                    torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.detach().copy_(
                    torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(self.__getattr__(name), 0)

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None:
                h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None:
                h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        """
            input:(bz, seq_len, emb_dim)
            masks:(bz, seq_len)
        """
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.detach().new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.detach().new(
                    batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask),
                                      requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(
                    -1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.detach().new(
                    batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask),
                                       requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = MyLSTM._forward_rnn(
                cell=self.fcells[layer],
                input=input,
                masks=masks,
                initial=initial,
                drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = MyLSTM._forward_brnn(
                    cell=self.bcells[layer],
                    input=input,
                    masks=masks,
                    initial=initial,
                    drop_masks=hidden_mask)

            h_n.append(
                torch.cat([layer_h_n, blayer_h_n], 1) if self.
                bidirectional else layer_h_n)
            c_n.append(
                torch.cat([layer_c_n, blayer_c_n], 1) if self.
                bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output],
                              2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)


class CNN(nn.Module):
    def __init__(self,
                 emb_dim,
                 n_filters=400,
                 kernel_size=3,
                 dropout=0.2,
                 output_dim=400):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=emb_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=emb_dim),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=n_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=n_filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=n_filters),
            nn.ReLU(),
        )
        self.fc = nn.Linear(in_features=n_filters, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch size, seq len, emb_dims]

        # x: [batch size, emb_dims, seq len]
        x = x.permute(0, 2, 1)
        out1 = self.conv1(x) + x  # resnet
        out2 = self.conv2(out1)
        out2 = self.dropout(out2)

        # out: [batch size, seq len, hidden size]
        out = out2.permute(0, 2, 1)

        # out: [batch size, seq len, output dim]
        out = self.fc(out)
        return out
