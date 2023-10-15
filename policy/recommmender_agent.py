import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch_geometric as geometric
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''
Change Log:
Restructured recommender_agent file. Origin file has common functions in NeuMF and MF: top_k(), user_encoder(), get_reward(), _l2_loss(), rank()...
Define a BaseRecommender and remove the redefined functions into BaseRecommender
Use NeuMF(BaseRecommender) and MF_Agent(BaseRecommender) classes to inherit functions from BaseRecommender, original NeuMF and MF_Agent ar both NeuMF(nn.Module)
'''

'''
Functional Classes
'''
class BaseRecommender(nn.Module):
    def __init__(self, data_config, args_config):
        super(BaseRecommender, self).__init__()

        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]


        self.user_dict = data_config["user_dict"]
        self.inter_matrix = data_config["inter_matrix"]
        self.emb_size = args_config.emb_size
        self.regs = eval(args_config.regs)
        self.batch = args_config.batch_size
        self.linear = nn.Linear(
            args_config.interaction_len, args_config.emb_size
        )
        self.counter_threshold = args_config.counter_threshold

        self.s = 0
        self.t = 0
        for batch_id in range(self.batch):
            self.s = self.batch * batch_id
            self.t = self.batch * (batch_id + 1)
            if self.t > self.n_users:
                self.t = self.n_users
            if self.s == self.t:
                break


    def top_k(self, k, train_user_dict, user_list, cfe_u_e = None):
        u_e, i_e = torch.split(self.all_embed, [self.n_users, self.all_embed.shape[0]-self.n_users])
        u_e = u_e[user_list, :]

        if cfe_u_e==None:
            score_matrix = torch.matmul(u_e, i_e.t())
        else:
            cfe_u_e = cfe_u_e.reshape(self.batch,-1)
            score_matrix = torch.matmul(u_e, cfe_u_e.t())

        for u in range(self.s, self.t):
            pos = train_user_dict[u]
            idx = pos.index(-1) if -1 in pos else len(pos)
            li = pos[:idx] - self.n_users
            li = [i for i in li if i < self.batch]
            new_score_matrix = (-1e5) * torch.ones(score_matrix[u-self.s][li].shape[0])
            score_matrix[u-self.s][li][:] = new_score_matrix[:]

        topk_score, topk_index = torch.topk(score_matrix, k)
        topk_index = topk_index.cpu().numpy() + self.n_users

        return topk_index.squeeze(), topk_score.squeeze()

    def user_encoder(self, inter_list, user):
        user_embed = torch.tanh(
                self.linear(inter_list) + self.all_embed[user])
        return user_embed


    def get_reward(self, model, user, cfe_item):
        '''
        Counferfactual Inference Reward**
        :param model: recommender model
        :param user: the current user
        :param cfe_item: counterfactual item sampled for the current user by the sampler
        :return: counterfactual reward
        '''
        cfe_u_e = []
        for i in user:
            his_inter = torch.cat((self.inter_matrix[i], torch.randint(low=self.n_users, high=self.n_users + self.n_items, size=(18, )).to('cuda')),0)
            cfe_u_e.append(model.user_encoder(his_inter[:], i))

        cfe_u_e = torch.cat(cfe_u_e,dim=0).to('cuda')
        rec, rec_score = model.top_k(1, self.user_dict, user)
        cfe_rec,cfe_score = model.top_k(1, self.user_dict, user, cfe_u_e)

        reward = torch.zeros(len(rec)).to('cuda')

        if self.args_config.reward == 'all':
            for i in range(len(rec)):
                # score = torch.mm(model.all_embed[[rec[i]]], u_e[i].unsqueeze(0).t())
                if rec_score[i] - cfe_score[i] >= self.counter_threshold:
                    reward[i] = 1 + F.cosine_similarity(self.all_embed[rec[i]], self.all_embed[cfe_rec[i]], dim=0)
                else:
                    # reward[i] = torch.abs(rec_score[i] - score)
                    reward[i] = F.cosine_similarity(self.all_embed[rec[i]], self.all_embed[cfe_rec[i]], dim=0)

        elif self.args_config.reward == 'R-reward':
            for i in range(len(rec)):
                if rec_score[i] - cfe_score[i] >= self.counter_threshold:
                    reward[i] = 1
                else:
                    reward[i] = 0

        elif self.args_config.reward == 'S-reward':
            for i in range(len(rec)):
                reward[i] = F.cosine_similarity(self.all_embed[rec[i]], self.all_embed[cfe_rec[i]], dim=0)

        return reward


    @staticmethod
    def _l2_loss(t):
        return torch.sum(t ** 2) / 2

    def rank(self, users, items):
        u_e = self.all_embed[users.to('cpu')]
        i_e = self.all_embed[items.to('cpu')]


        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender embedding size {}".format(
            self.args_config.emb_size
        )

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    embed CKG and using its embedding to calculate prediction score
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = geometric.nn.GATConv(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = self.dropout(x)
        x = F.normalize(x)
        return x

'''
Recommenders for options 
'''
class MF(BaseRecommender):
    # def __init__(self, baseData, recModel,data_config, args_config):
    def __init__(self, data_config, args_config):

        super(MF, self).__init__(data_config, args_config)

        self.all_embed = self._init_weight()

    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_users + self.n_items, self.emb_size)
        )
        if self.args_config.pretrain_r:
            all_embed.data = self.data_config["all_embed"]
       #  else:
       #      all_embed.data = torch.from_numpy(self.all_emb_items) #***** retrain the BaseRec
       # #******* # numpy.concatenate((all_emb_users, all_emb_items), axis=0)
             #all_embed.requires_grad = False

        else:
            nn.init.xavier_uniform_(all_embed)

        return all_embed

    def forward(self, user, pos_item, neg_item):

        u_e = self.all_embed[user]
        pos_e = self.all_embed[pos_item]
        neg_e = self.all_embed[neg_item]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        return bpr_loss

class NeuMF(BaseRecommender):
    '''
    Unique configs for NeuMF
    '''
    def __init__(self, data_config, args_config):
        super(NeuMF, self).__init__(data_config, args_config)
        num_hidden_layer = 2

        self.all_embed_MF = nn.Embedding(self.n_users + self.n_items, self.emb_size)

        self.all_embed_MLP = nn.Embedding(self.n_users + self.n_items, self.emb_size)

        nn.init.normal_(self.all_embed_MF.weight, mean=0., std=0.01)
        nn.init.normal_(self.all_embed_MLP.weight, mean=0., std=0.01)

        self.all_embed = self.all_embed_MLP.weight
       
        # Layer configuration
        MLP_layers = []
        layers_shape = [self.emb_size * 2]
        for i in range(num_hidden_layer):
            layers_shape.append(layers_shape[-1] // 2)
            MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
            MLP_layers.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_layers)
        print("MLP Layer Shape ::", layers_shape) # MLP Layer Shape :: [256, 128, 64]

        ## Final Layer
        self.final_layer = nn.Linear(layers_shape[-1] + self.emb_size, 1) # (final_layer): Linear(in_features=192, out_features=1, bias=True)

        # Loss function
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, user, pos_item, neg_item):
        pos_score = self.forward_pair(user, pos_item)  # bs x 1
        neg_score = self.forward_pair(user, neg_item)  # bs x 1

        output = (pos_score, neg_score)
        # return output

        #The loss
        pos_score, neg_score = output[0], output[1]

        pred = torch.cat([pos_score, neg_score], dim=0)
        gt = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)

        return self.BCE_loss(pred, gt)

    def forward_pair(self, batch_user, batch_item):

        # MF
        u_mf = self.all_embed_MF(batch_user)  # batch_size x dim
        i_mf = self.all_embed_MF(batch_item)  # batch_size x dim

        mf_vector = (u_mf * i_mf)  # batch_size x dim

        # MLP
        u_mlp = self.all_embed_MLP(batch_user)		# batch_size x dim
        i_mlp = self.all_embed_MLP(batch_item)		# batch_size x dim

        mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vector = self.MLP_layers(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1) #torch.Size([1024, 192])

        output = self.final_layer(predict_vector)

        return output

class BPR(BaseRecommender):
    def __init__(self, data_config, args_config):
        super(BPR, self).__init__(data_config, args_config)
        self.all_emb = nn.Embedding(self.n_users + self.n_items, self.emb_size)
        nn.init.normal_(self.all_emb.weight, mean=0., std=0.01)
        self.all_embed = self.all_emb.weight

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.all_emb(batch_user)
        i = self.all_emb(batch_pos_item)
        j = self.all_emb(batch_neg_item)

        reg_loss = self._l2_loss(u) + self._l2_loss(i) + self._l2_loss(j)
        reg_loss = self.regs * reg_loss

        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        output = (pos_score, neg_score,reg_loss)

        pos_score, neg_score, reg_loss = output[0], output[1], output[2]

        loss = -(pos_score - neg_score).sigmoid().log().sum() + reg_loss

        return loss

class KGAT(BaseRecommender):
    def __init__(self, data_config, args_config):
        super(KGAT, self).__init__(data_config, args_config)
        self.n_nodes = data_config["n_nodes"]
        self.input_channel = 64
        self.output_channel = 64
        self.gcn = GraphConv(self.input_channel, self.output_channel)
        self.n_entities = data_config["n_nodes"]
        self.item_range = data_config["item_range"]
        self.entity_embedding = self._initialize_weight(
            self.n_entities, self.input_channel
        )
        self.all_embed = self._init_weight()

    def _initialize_weight(self, n_entities, input_channel):
        """entities includes items and other entities in knowledge graph"""
        if self.args_config.pretrain_s:
            kg_embedding = self.data_config["kg_embedding"]
            entity_embedding = nn.Parameter(kg_embedding)
        else:
            entity_embedding = nn.Parameter(
                torch.FloatTensor(n_entities, input_channel)
            )
            nn.init.xavier_uniform_(entity_embedding)

        if self.args_config.freeze_s:
            entity_embedding.requires_grad = False

        return entity_embedding

    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.args_config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        return all_embed

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, user, pos_item, neg_item, edges_matrix):
        u_e, pos_e, neg_e = (
            self.all_embed[user],
            self.all_embed[pos_item],
            self.all_embed[neg_item],
        )

        edges = self.build_edge(edges_matrix)
        # x = self.all_embed
        x = self.entity_embedding

        gcn_embedding = self.gcn(x, edges.t().contiguous())

        u_e_, pos_e_, neg_e_ = (
            gcn_embedding[user],
            gcn_embedding[pos_item],
            gcn_embedding[neg_item],
        )

        u_e = torch.cat([u_e, u_e_], dim=1)
        pos_e = torch.cat([pos_e, pos_e_], dim=1)
        neg_e = torch.cat([neg_e, neg_e_], dim=1)

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        # Defining objective function that contains:
        # ... (1) bpr loss
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        # ... (2) emb loss
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss

        loss = bpr_loss + reg_loss

        return loss
