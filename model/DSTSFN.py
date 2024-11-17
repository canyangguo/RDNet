import torch
import torch.nn as nn
from model.basic_module import meta_glu, generate_spatial_graph, generate_temporal_graph, meta_Linear, weight_and_initial, glu


class dynamic_temporal_graph_convolution(nn.Module):
    def __init__(self, d_model, num_of_patterns, his_num, pred_num):
        super(dynamic_temporal_graph_convolution, self).__init__()
        self.generate_temporal_pattern = weight_and_initial(his_num, pred_num, num=num_of_patterns, bias=None)  # num, P, Q
        self.glu = glu(d_model)

    def forward(self, x, index):

        tp_adj, _ = self.generate_temporal_pattern()  # l, P, Q
        # index:  b, n
        # tp_adj: l, p, q
        t_adj = tp_adj[index]
        x = torch.einsum('bnpq, bpnc -> bqnc', t_adj, x)

        x = self.glu(x)

        return x

class dynamic_spatial_graph_convolution(nn.Module):
    def __init__(self, d_model, num_of_latents, drop_rate=0):
        super(dynamic_spatial_graph_convolution, self).__init__()
        self.s_adj = generate_spatial_graph(d_model, num_of_latents)
        self.drop = nn.Dropout(drop_rate)
        self.glu = meta_glu(d_model, d_model)

    def forward(self, mk, x):
        s_adj = self.s_adj(mk)
        x = torch.einsum('bmn, btnc -> btmc', self.drop(s_adj), x)
        x = self.glu(mk, x)
        return x

class encoder(nn.Module):
    def __init__(self, d_model, num_of_latents, num_of_patterns, his_num, pred_num):
        super(encoder, self).__init__()
        self.psgcn = dynamic_spatial_graph_convolution(d_model, num_of_latents)
        self.ptgcn = dynamic_temporal_graph_convolution(d_model, num_of_patterns, his_num, pred_num)

    def forward(self, x, index):

        x = self.psgcn(mk, x) + x

        x = self.ptgcn(x, index) + x

        return x


class make_predictor(nn.Module):
    def __init__(self, model_args, task_args, num_of_patterns):
        super(make_predictor, self).__init__()
        self.data_emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])

        self.encoders = nn.ModuleList([encoder(model_args['d_model'], model_args['num_of_latents'],
                               num_of_patterns, task_args['his_num'], task_args['pred_num'])
                                       for _ in range(model_args['num_of_layers'])])
        self.reg = nn.Linear(model_args['d_model'], model_args['num_of_outputs'])

    def forward(self, x, index):
        x = self.data_emb(x)
        for encoder in self.encoders:
            x = encoder(x, index)
        x = self.reg(x).squeeze(-1)
        return x