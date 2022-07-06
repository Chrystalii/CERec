import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Agent(nn.Module):
    # def __init__(self, baseData, recModel,data_config, args_config): #unzip this and pass corresponding args if using base CliFModel
    def __init__(self, data_config, args_config):

        super(Agent, self).__init__()
        # self.all_emb_users = recModel(baseData).U
        # self.all_emb_items = recModel(baseData).V

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

        self.all_embed = self._init_weight()

        self.s = 0
        self.t = 0
        for batch_id in range(self.batch):
            self.s = self.batch * batch_id
            self.t = self.batch * (batch_id + 1)
            if self.t > self.n_users:
                self.t = self.n_users
            if self.s == self.t:
                break

    def _init_weight(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_users + self.n_items, self.emb_size)
        )
        if self.args_config.pretrain_r:
            all_embed.data = self.data_config["all_embed"]
       #  else:
       #      all_embed.data = torch.from_numpy(self.all_emb_items)
       # #******* # numpy.concatenate((all_emb_users, all_emb_items), axis=0)

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
        cfe_u_e = []
        for i in user:
            his_inter = torch.cat((self.inter_matrix[i], torch.randint(low=self.n_users, high=self.n_users + self.n_items, size=(18, )).to('cuda')),0)
            cfe_u_e.append(model.user_encoder(his_inter[:], i))

        cfe_u_e = torch.cat(cfe_u_e,dim=0).to('cuda')
        rec, rec_score = model.top_k(1, self.user_dict, user)
        cfe_rec,cfe_score = model.top_k(1, self.user_dict, user, cfe_u_e)

        reward = torch.zeros(len(rec)).to('cuda')

        for i in range(len(rec)):
            if rec_score[i] - cfe_score[i] >= self.counter_threshold:
                reward[i] = 1 + F.cosine_similarity(self.all_embed[rec[i]], self.all_embed[cfe_rec[i]], dim=0)
            else:
                reward[i] = F.cosine_similarity(self.all_embed[rec[i]], self.all_embed[cfe_rec[i]], dim=0)
        return reward


    @staticmethod
    def _l2_loss(t):
        return torch.sum(t ** 2) / 2

    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using MF, embedding size {}".format(
            self.args_config.emb_size
        )
