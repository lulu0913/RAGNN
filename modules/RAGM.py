import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_factors, a):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.a = a
        self.w = torch.nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]), requires_grad=True)
        self.WW = nn.Linear(64 * 2, 64)
        self.WW_i = nn.Linear(64 * 2, 64)
        self.activation = nn.LeakyReLU()

    def forward(self, entity_emb, user_emb, latent_emb,
                adj_mat, interact_mat):

        device = torch.device("cuda:0")
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """
        user representation learning in u-i bigraph
        """

        user_index, item_index = adj_mat.nonzero()
        user_index = torch.tensor(user_index).type(torch.long).to(device)
        item_index = torch.tensor(item_index).type(torch.long).to(device)
        user_pair = user_emb[user_index]
        item_pair = entity_emb[item_index]
        relation_ui = self.activation(self.WW(torch.cat([user_pair, item_pair], dim=1)))
        relation_ = torch.mm(relation_ui, latent_emb.t())
        relation_ui_type = torch.argmax(relation_, dim=1)

        relation_ui_uni = torch.unique(relation_ui_type)
        user_emb_list = []
        for i in range(n_factors):
            user_emb_list.append(user_emb)

        for i in relation_ui_uni:
            index = torch.where(relation_ui_type == i)
            index = index[0]
            head = user_index[index]
            tail = item_index[index]
            relation_ui_sub = relation_ui[index]
            neigh_emb = entity_emb[tail]
            u_ = scatter_mean(src=neigh_emb, index=head, dim_size=n_users, dim=0)
            center_emb = u_[head]
            # temp = center_emb * neigh_emb
            sim = torch.sum(relation_ui_sub * center_emb, dim=1)
            n, d = neigh_emb.size()
            sim = torch.unsqueeze(sim, dim=1)
            sim.expand(n, d)
            neigh_emb = sim * neigh_emb
            u = scatter_mean(src=neigh_emb, index=head, dim_size=n_users, dim=0)
            squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
            u = squash.unsqueeze(1) * F.normalize(u, dim=1)
            user_emb_list[i] = u + user_emb

        relation_ui = self.activation(self.WW_i(torch.cat([item_pair, user_pair], dim=1)))
        neigh_emb = user_emb[user_index]
        y_ = scatter_mean(src=neigh_emb, index=item_index, dim_size=n_entities, dim=0)
        center_emb = y_[item_index]
        # temp = center_emb * neigh_emb
        sim = torch.sum(relation_ui * center_emb, dim=1)
        n, d = neigh_emb.size()
        sim = torch.unsqueeze(sim, dim=1)
        sim.expand(n, d)
        neigh_emb = sim * neigh_emb
        y = scatter_mean(src=neigh_emb, index=item_index, dim_size=n_entities, dim=0)
        squash = torch.norm(y, dim=1) ** 2 / (torch.norm(y, dim=1) ** 2 + 1)
        y = squash.unsqueeze(1) * F.normalize(y, dim=1)
        entity_agg = y + entity_emb

        user_0 = user_emb_list[0]
        user_1 = user_emb_list[1]
        user_2 = user_emb_list[2]
        w0 = self.w[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w1 = self.w[1].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w2 = self.w[2].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w_0 = torch.exp(w0)/(torch.exp(w0)+torch.exp(w1)+torch.exp(w2))
        w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        user_agg = w_0.mul(user_0) + w_1.mul(user_1) + w_2.mul(user_2)

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, a, n_users,
                 n_factors, adj_mat, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.adj_mat = adj_mat
        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors, a=a))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, latent_emb,
                adj_mat, interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = 0.0

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 adj_mat, interact_mat)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.a = args_config.alpha
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.adj_mat = adj_mat

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         a=self.a,
                         n_users=self.n_users,
                         n_factors=self.n_factors,
                         adj_mat=self.adj_mat,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, cf_batch):
        user = cf_batch['users']
        pos_item = cf_batch['pos_items']
        neg_item = cf_batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]

        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.adj_mat,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e, cor)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.adj_mat,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss + emb_loss, emb_loss, cor

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)