import torch
import torch.nn as nn
from model.basic_module import meta_glu, generate_spatial_graph, generate_temporal_graph, meta_Linear, \
    weight_and_initial, glu, matrix_decomposition
import torch.nn.functional as F

class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units * 2, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(1, 1, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        outputs = torch.einsum('bnc, cd ->bnd', concatenation, self.weights) + self.biases
        return outputs

class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.linear1 = GRULinear(self._hidden_dim, self._hidden_dim *2, bias=1.0)
        self.linear2 = GRULinear(self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))

        r, u = torch.chunk(concatenation, chunks=2, dim=-1)

        c = torch.tanh(self.linear2(inputs, r * hidden_state))

        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state


class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super(GRU, self).__init__()
        self._input_dim = input_dim  # num_nodes for prediction
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)

    def forward(self, inputs):

        batch_size, seq_len, num_nodes, channels = inputs.shape
        assert self._input_dim == channels
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(
            inputs
        )
        for i in range(seq_len):
            output, hidden_state = self.gru_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        #last_output = outputs[-1]
        return torch.stack(outputs, dim=1)#[-1]



class start_make_model(nn.Module):
    def __init__(self, config_args, args, mean_std):
        super(start_make_model, self).__init__()
        model_args, task_args, data_args = config_args['model'], config_args['task'], config_args['data']
        self.args = args
        self.mean_std = mean_std
        self.emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])

        self.GRUs = nn.ModuleList([
            GRU(model_args['d_model'], model_args['d_model'])
            for _ in range(4)
        ])
        self.pred = nn.Linear(model_args['d_model'], 12 * model_args['d_model'])
        self.re_emb_s = nn.Linear(model_args['d_model'], model_args['num_of_features'])


    def forward(self, stage, x, ti, di, ep, tdx, S2D=-1, mbx=None, mby=None, tid=None, s2t_id=None, mask=None, delta=None, rate=None, learning=False):
        x = torch.relu(self.emb(x))
        for gru in self.GRUs:
            x = gru(x)
        x = x[:, -1]

        x = self.pred(x).reshape(x.shape[0], x.shape[1], 12, -1).transpose(1, 2)
        x = self.re_emb_s(x)

        y = (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)
        return y, 0, 0



