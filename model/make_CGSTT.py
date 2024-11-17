import torch
import torch.nn as nn
from model.basic_module import weight_and_initial, glu, matrix_decomposition, CMD

class spatial_graph_construct(nn.Module):
    def __init__(self, source_node, d_model, times, num_of_patterns, drop_rate):
        super(spatial_graph_construct, self).__init__()
        self.d_model = d_model
        self.num_of_patterns = num_of_patterns
        self.weights = weight_and_initial(d_model, d_model * 2, times, bias=False)
        self.dr = nn.Dropout(drop_rate)

    def forward(self, x):
        weights = self.weights().reshape(-1, self.d_model, self.d_model * 2)
        qks = torch.einsum('kcd, nc -> knd', weights, torch.mean(x, dim=(0, 1)))  # knd
        q, k = torch.chunk(qks, 2, dim=-1)
        score = torch.einsum('kmd, knd -> kmn', q, k) / (self.d_model ** 0.5)
        score = torch.softmax(score, dim=-1)
        score = self.dr(score)
        return score


class dynamic_spatial_graph_convolution(nn.Module):
    def __init__(self, source_node, d_model, num_of_latents, num_of_times, num_of_days, drop_rate=0.1):  # 注意
        super(dynamic_spatial_graph_convolution, self).__init__()
        self.adjs = spatial_graph_construct(source_node, d_model, num_of_times+1, num_of_latents, drop_rate)
        self.glu = glu(d_model)

    def forward(self, x, ep, S2D, ti):
        B, T, N, C = x.shape
        adjs = self.adjs(x).reshape(-1, N, N)
        if ep < S2D:
            x = torch.einsum('mn, btnc -> btmc', adjs[-1], x)
        else:
            adj = adjs[-1] / 2 + adjs[ti] / 2
            x = torch.einsum('bmn, btnc -> btmc', adj, x)
        x = self.glu(x)
        return x


class dynamic_temporal_graph_convolution(nn.Module):
    def __init__(self, d_model, pred_num, input_length, output_length, num_of_times, num_of_days, source_node_num,
                 num_of_patterns):
        super(dynamic_temporal_graph_convolution, self).__init__()
        self.pred_num = pred_num
        self.input_length = input_length
        self.output_length = output_length

        self.adjs = weight_and_initial(input_length, output_length, num_of_times+1, bias=None)
        self.glu = glu(d_model)


    def forward(self, x, ep, S2D, ti):
        adjs = self.adjs()
        if ep < S2D:
            x = torch.einsum('pq, bpnc -> bqnc', adjs[-1], x)
        else:
            adj = adjs[-1:] / 2 + adjs[ti] / 2
            x = torch.einsum('bpq, bpnc -> bqnc', adj, x)
        x = self.glu(x)
        return x

class private_module(nn.Module):
    def __init__(self, d_model, input_length, output_length, num_of_latent, num_of_times, num_of_days, num_of_nodes,
                 drop_rate=0.8):
        super(private_module, self).__init__()

        self.d_model = d_model
        self.s_adjs = matrix_decomposition(num_of_nodes, num_of_nodes, d_model // 2, num_of_times + 1)
        self.drop = nn.Dropout(drop_rate)
        self.glu1 = glu(d_model)

        self.t_adjs = weight_and_initial(input_length, output_length, num_of_times + 1, bias=None)
        self.glu2 = glu(d_model)

    def forward(self, x, ti, ep, S2D):
        if ep < S2D:
            s_adj = self.s_adjs()[-1]
            x = self.glu1(torch.einsum('mn, btnc -> btmc', self.drop(s_adj), x)) + x
            t_adj = self.t_adjs()[-1]
            x = self.glu2(torch.einsum('pq, bpnc -> bqnc', t_adj, x)) + x
        else:
            s_adjs = (self.s_adjs()[ti] + self.s_adjs()[-1]) / 2
            x = self.glu1(torch.einsum('bmn, btnc -> btmc', self.drop(s_adjs), x)) + x

            t_adjs = (self.t_adjs()[ti] + self.t_adjs()[-1]) / 2
            x = self.glu2(torch.einsum('bpq, bpnc -> bqnc', t_adjs, x)) + x
        return x


class st_module(nn.Module):
    def __init__(self, d_model, pred_num, input_length, output_length, num_of_latents, num_of_times, num_of_days,
                 source_node_num):
        super(st_module, self).__init__()
        self.dsgnn = dynamic_spatial_graph_convolution(source_node_num, d_model, num_of_latents, num_of_times,
                                                       num_of_days)
        self.dtgnn = dynamic_temporal_graph_convolution(d_model, pred_num, input_length, output_length, num_of_times, num_of_days,
                                                        source_node_num, num_of_latents)

    def forward(self, x, ti, ni, ep, S2D):
        x = self.dsgnn(x, ep, S2D, ti) + x
        x = self.dtgnn(x, ep, S2D, ti) + x
        return x


class start_make_model(nn.Module):
    def __init__(self, config_args, args, mean_std):
        super(start_make_model, self).__init__()
        model_args, task_args, data_args = config_args['model'], config_args['task'], config_args['data']
        self.mean_std = mean_std
        self.emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

        self.st_models = nn.ModuleList([
            st_module(model_args['d_model'], task_args['pred_num'], task_args['his_num'], task_args['pred_num'], args.num_of_latent,
                      data_args[args.source_dataset]['num_of_times'], data_args[args.source_dataset]['num_of_days'],
                      data_args[args.source_dataset]['node_num'])
            for _ in range(model_args['num_of_layers'])
        ])

        self.re_emb_s = nn.Linear(model_args['d_model'], model_args['num_of_features'])
        self.var_layer1 = private_module(model_args['d_model'], task_args['his_num'], task_args['pred_num'],
                                         args.num_of_latent,
                                         data_args[args.target_dataset]['num_of_times'],
                                         data_args[args.target_dataset]['num_of_days'],
                                         data_args[args.target_dataset]['node_num'])
        self.var_layer2 = private_module(model_args['d_model'], task_args['his_num'], task_args['pred_num'],
                                         args.num_of_latent,
                                         data_args[args.target_dataset]['num_of_times'],
                                         data_args[args.target_dataset]['num_of_days'],
                                         data_args[args.target_dataset]['node_num'])

    def index_selector(self, x1, x2, moments=2, element_wise=False):
        cmds = CMD(x1, x2, moments, element_wise)
        return cmds

    def forward(self, stage, x, ti, di, ep, tdx, S2D=-1, mbx=None, mby=None, tid=None, s2t_id=None):
        if stage == 'source':
            x = torch.relu(self.emb(x))
            for st_module in self.st_models:
                x = st_module(x, ti, ti, ep, S2D)
            x = self.re_emb_s(x[:, :tdx])
            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)

        if stage == 'transition':

            # source embedding layer
            x = torch.relu(self.emb(x))

            # embedding layer
            mbx = torch.relu(self.emb(mbx))

            # target distribution transformation
            mbx = self.var_layer1(mbx, tid, ep, S2D)

            # select index
            sim1, gv1, gi = self.index_selector(mbx, x)  # B

            gi = ti[gi]

            for st_module in self.st_models:
                x = st_module(x, ti, ti, ep, S2D)
                mbx = st_module(mbx, gi, gi, ep, S2D)

            mbx = self.var_layer2(mbx, tid, ep, S2D)  # 需要想想位置要不要变换

            x = self.re_emb_s(x)
            mbx = self.re_emb_s(mbx)

            mby = (mby - self.mean_std[0]) / self.mean_std[1]
            gv2 = self.index_selector(mbx, mby, element_wise=True)  # B

            loss = self.lambda1 * torch.mean(gv1) + self.lambda2 * torch.mean(gv2)  # 是否固定映射呢

            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1), loss

        if stage == 'finetuning':

            x = torch.relu(self.emb(x))
            x = self.var_layer1(x, ti, ep, S2D)

            gi = s2t_id[ti]
            # gi = torch.randint(0, self.sc.shape[0], size=gi.shape)
            # gi = (ti + np.random.randint(0, self.sc.shape[0])) % self.sc.shape[0]

            for st_module in self.st_models:
                x = st_module(x, gi, gi, ep, S2D)

            x = self.var_layer2(x, ti, ep, S2D)  # 需要想想位置要不要变换
            x = self.re_emb_s(x)
            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)




