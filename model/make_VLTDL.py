import torch
import torch.nn as nn
from model.basic_module import weight_and_initial, glu, matrix_decomposition


class dynamic_spatial_graph_neural_network(nn.Module):
    def __init__(self, in_num_of_nodes, out_num_of_nodes, num_of_latent, num_of_times, num_of_days, d_model, drop_rate=0.9):
        super(dynamic_spatial_graph_neural_network, self).__init__()
        self.num_of_times = num_of_times
        self.adjs = matrix_decomposition(in_num_of_nodes, out_num_of_nodes, num_of_latent, num_of_times+num_of_days+1)
        self.drop = nn.Dropout(drop_rate)
        self.glu = glu(d_model)

    def forward(self, x, ti, di, ep, S2D):
        if ep < S2D:
            adj = self.adjs()[-1]
            x = self.glu(torch.einsum('mn, btnc -> btmc', self.drop(adj), x))
        else:
            di = self.num_of_times + di
            adjs = (self.adjs()[ti] + self.adjs()[di] + self.adjs()[-1]) / 3
            x = self.glu(torch.einsum('bmn, btnc -> btmc', self.drop(adjs), x))
        return x


class dynamic_temporal_graph_neural_network(nn.Module):
    def __init__(self, input_length, output_length, num_of_latent, num_of_times, num_of_days, d_model, drop_rate=0):
        super(dynamic_temporal_graph_neural_network, self).__init__()
        num_of_latent = 8
        self.num_of_times = num_of_times
        self.adjs = matrix_decomposition(input_length, output_length, num_of_latent, num_of_times+num_of_days+1)
        self.drop = nn.Dropout(drop_rate)
        self.glu = glu(d_model)

    def forward(self, x, ti, di, ep, S2D):
        if ep < S2D:
            adj = self.adjs()[-1]
            x = self.glu(torch.einsum('tp, btnc -> bpnc', self.drop(adj), x))
        else:
            di = self.num_of_times + di
            adjs = (self.adjs()[ti] + self.adjs()[di] + self.adjs()[-1]) / 3
            x = self.glu(torch.einsum('btp, btnc -> bpnc', self.drop(adjs), x))
        return x


class st_module(nn.Module):
    def __init__(self, d_model, input_length, output_length, num_of_latent, num_of_times, num_of_days, num_of_nodes, pred_num):
        super(st_module, self).__init__()
        self.s_module = dynamic_spatial_graph_neural_network(num_of_nodes, num_of_nodes, num_of_latent, num_of_times, num_of_days, d_model)
        self.t_module = dynamic_temporal_graph_neural_network(input_length, output_length, num_of_latent, num_of_times, num_of_days, d_model)

    def forward(self, x, ti, di, ep, tdx, S2D):
        x = self.s_module(x, ti, di, ep, S2D) + x
        x = self.t_module(x, ti, di, ep, S2D)# + self.t_module_res(x, ti, di, ep, tdx, S2D)
        return x





class start_make_model(nn.Module):
    def __init__(self, config_args, args, mean_std):
        super(start_make_model, self).__init__()
        model_args, task_args, data_args = config_args['model'], config_args['task'], config_args['data']
        self.mean_std = mean_std
        self.emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])

        self.st_models = nn.ModuleList([
            st_module(model_args['d_model'], in_l, out_l, args.num_of_latent,
                      data_args[args.source_dataset]['num_of_times'], data_args[args.source_dataset]['num_of_days'],
                      data_args[args.source_dataset]['node_num'], config_args['task']['pred_num'])
            for _, in_l, out_l in zip(range(model_args['num_of_layers']), [288, 144, 72, 36], [144, 72, 36, 12])
        ])

        self.re_emb_s = nn.Linear(model_args['d_model'], model_args['num_of_features'])


    def forward(self, stage, x, ti, di, ep, tdx, S2D=-1, mbx=None, mby=None, tid=None, s2t_id=None):
        x = torch.relu(self.emb(x))
        for st_module in self.st_models:
            x = st_module(x, ti, di, ep, tdx, S2D)
        x = self.re_emb_s(x)
        return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)




