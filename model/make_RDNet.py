import torch
import torch.nn as nn
from model.basic_module import weight_and_initial, glu, matrix_decomposition
import pandas as pd
def heat(x):
    x = x[:50, :50]
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["font.size"] = 16

    x = x.cpu().detach().numpy()
    ax = sns.heatmap(x)

    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    plt.gca().tick_params(axis='x')  # x轴刻度标签字体大小
    plt.gca().tick_params(axis='y')  # y轴刻度标签字体大小
    for edge in ['top', 'right', 'left', 'bottom']:
        ax.spines[edge].set_visible(True)  # 设置为显示状态
        ax.spines[edge].set_color("#767273")  # 将颜色设置为偏黑色的一个颜色
        ax.spines[edge].set_linewidth(1)  # 线条宽度

    figure = ax.get_figure()

    figure.savefig('heat.pdf')  # 保存图片

    plt.show()

class dynamic_spatial_graph_neural_network(nn.Module):
    def __init__(self, args, model_args, task_args, data_args, drop_rate, num_of_latent=32):
        super(dynamic_spatial_graph_neural_network, self).__init__()
        self.num_of_times = data_args[args.source_dataset]['num_of_times']
        node_num = data_args[args.source_dataset]['node_num']
        self.adjs = matrix_decomposition(node_num, node_num, num_of_latent, 3)
        self.rate = drop_rate
        self.drop = nn.Dropout(drop_rate)  #找出异常,mask掉dropout_layer(drop_rate)#
        self.glu = glu(model_args['d_model'], model_args['d_model'])

    def get_sparse_mask(self, data, missing_rate):  #为什么这么耗时，因为输入到cuda吗
        mask = torch.round(torch.rand(data.shape) + 0.5 - missing_rate)  # 0-1随机数
        return mask.cuda()

    def forward(self, x, mask, ti, di, ep, S2D, learning=False):
        B, T, N, C = x.shape
        adj = torch.sum(torch.relu(self.adjs()), dim=0)  # N*N
        #pd.DataFrame(adj.cpu().detach().numpy()).to_csv('full-adj.csv')

        x = self.glu(torch.einsum('nm, btmc -> btnc', self.drop(adj), x))
        #x = self.glu(torch.einsum('nm, btmc -> btnc', adj, x))

        return x#, l2_loss


class dynamic_temporal_graph_neural_network(nn.Module):
    def __init__(self, args, model_args, task_args, data_args):
        super(dynamic_temporal_graph_neural_network, self).__init__()
        self.num_of_times = data_args[args.source_dataset]['num_of_times']
        #num_of_patterns = self.num_of_times + data_args[args.source_dataset]['num_of_days'] + 1
        self.adjs = weight_and_initial(task_args['his_num']*3, task_args['pred_num'], 1, bias=None)
        self.fc = nn.Linear(model_args['d_model'] * 3, model_args['d_model'])  # 改成glu 试试
        #self.glu = glu(model_args['d_model']*3, model_args['d_model'])

    def forward(self, x, mask, ti, di, ep, S2D):
        B, T, N, C = x.shape
        adj = self.adjs()

        h = torch.einsum('qp, bpnc -> bqnc', adj, x).reshape(B, 3, T, N, C).permute(0, 2, 3, 1, 4).reshape(B, T, N, -1)

        x = torch.relu(self.fc(h)) + x
        #x = self.glu(x)
        #x = self.glu(torch.einsum('qp, bpnc -> bqnc', adj, x))

        return x#, l2_loss



class variable_block(nn.Module):
    def __init__(self, args, model_args, task_args, data_args, drop_rate):
        super(variable_block, self).__init__()
        self.dsgnn = dynamic_spatial_graph_neural_network(args, model_args, task_args, data_args, drop_rate)
        self.dtgnn1 = dynamic_temporal_graph_neural_network(args, model_args, task_args, data_args)
        self.dtgnn2 = dynamic_temporal_graph_neural_network(args, model_args, task_args, data_args)

    def forward(self, x, mask, ti, di, yti, ydi, ep, S2D, learning):
        s = self.dsgnn(x, mask, ti, di, ep, S2D, learning) + x
        x = self.dtgnn1(s, mask, ti, di, ep, S2D) + s
        y = self.dtgnn2(s, mask, yti, ydi, ep, S2D) + s
        return x, y


class invariant_block(nn.Module):
    def __init__(self, args, model_args, task_args, data_args):
        super(invariant_block, self).__init__()  # 另外一种做法： N*288*7*C的参数量
        self.pred_num = task_args['pred_num']
        self.num_of_times = data_args[args.source_dataset]['num_of_times']
        self.num_of_days = data_args[args.source_dataset]['num_of_days']
        self.time_emb = weight_and_initial(data_args[args.source_dataset]['num_of_times'], model_args['d_model'], bias=False)
        self.day_emb = weight_and_initial(data_args[args.source_dataset]['num_of_days'], model_args['d_model'], bias=False)
        self.node_emb = weight_and_initial(data_args[args.source_dataset]['node_num'], model_args['d_model'], bias=False)
        self.fc1 = nn.Linear(model_args['d_model'], model_args['d_model'])
        self.fc2 = nn.Linear(model_args['d_model'], model_args['d_model'])
        self.fc3 = nn.Linear(model_args['d_model'], model_args['d_model'])
        self.glu = glu(model_args['d_model'], model_args['d_model'])

    def forward(self, ti, di):
        ti = (ti.unsqueeze(-1).repeat(1, 12) + torch.arange(0, 12).unsqueeze(0)) % self.num_of_times  # 32,12
        di = di.unsqueeze(-1).repeat(1, 12)
        di = (torch.where(ti < self.pred_num, di + 1, di)) % self.num_of_days

        tem_emb = self.time_emb()[ti].unsqueeze(2)  # B, T, 1, C
        day_emb = self.day_emb()[di].unsqueeze(2)  # B, T, 1, C
        node_emb = self.node_emb().unsqueeze(0).unsqueeze(0)  # 1, 1, N, C
        x = torch.relu(self.fc1(tem_emb)) + torch.relu(self.fc2(day_emb)) + torch.relu(self.fc3(node_emb))
        x = self.glu(x)
        return x


class decouple_module(nn.Module):  # Variable signal and invariant signal
    def __init__(self, args, model_args, task_args, data_args, drop_rate):
        super(decouple_module, self).__init__()
        self.pred_num = task_args['pred_num']
        self.num_of_times = data_args[args.source_dataset]['num_of_times']
        self.num_of_days = data_args[args.source_dataset]['num_of_days']

        self.invariant_block = invariant_block(args, model_args, task_args, data_args)
        self.variable_block = variable_block(args, model_args, task_args, data_args, drop_rate)

    def get_label_index(self, ti, di):
        ti = (ti + self.pred_num) % self.num_of_times
        di = (torch.where(ti < self.pred_num, di+1, di)) % self.num_of_days
        return ti, di

    def forward(self, x, dx, mask, ti, di, ep, S2D, learning, pos=None):

        yti, ydi = self.get_label_index(ti, di)
        ix = self.invariant_block(ti, di)
        iy = self.invariant_block(yti, ydi)

        h = x - ix
        dh = dx - ix

        vx, vy = self.variable_block(h, mask, ti, di, yti, ydi, ep, S2D, learning)
        dvx, dvy = self.variable_block(dh, mask, ti, di, yti, ydi, ep, S2D, learning)

        res_x = x - ix - vx
        res_dx = dx - ix - dvx

        return res_x, res_dx, iy, vy, dvy



class start_make_model(nn.Module):
    def __init__(self, config_args, args, mean_std):
        super(start_make_model, self).__init__()
        model_args, task_args, data_args = config_args['model'], config_args['task'], config_args['data']
        self.args = args
        self.mean_std = mean_std
        self.emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])

        self.decouple_modules = nn.ModuleList([
            decouple_module(args, model_args, task_args, data_args, drop_rate=0.8)
            for _ in range(4)])

        self.re_emb = nn.Linear(model_args['d_model'], model_args['num_of_features'])
        self.criterion3 = torch.nn.HuberLoss(delta=1)

    def re_norm(self, x):
        return (x * self.mean_std[1] + self.mean_std[0])

    def forward(self, stage, x, ti, di, ep, tdx, S2D=-1, mbx=None, mby=None, tid=None, s2t_id=None, mask=None, delta=None, rate=None, learning=False):
        dx = torch.relu(self.emb(x + delta))
        x = torch.relu(self.emb(x))

        ys, iys, vys, dvys = 0, 0, 0, 0
        for module in self.decouple_modules:
            x, dx, iy, vy, dvy = module(x, dx, mask, ti, di, ep, S2D, learning)
            iys = iys + self.re_emb((iy)[:, :tdx])
            ys = ys + self.re_emb((iy+vy)[:, :tdx])

            vys = vys + self.re_emb((vy)[:, :tdx])
            dvys = dvys + self.re_emb((dvy)[:, :tdx])

        iys = self.re_norm(iys).squeeze(-1)
        ys = self.re_norm(ys).squeeze(-1)

        vys = self.re_norm(vys).squeeze(-1)
        dvys = self.re_norm(dvys).squeeze(-1)

        return ys, iys, self.criterion3(vys, dvys)




