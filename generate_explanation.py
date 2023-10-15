from common import parse_args
from policy import KGPolicy_device, Agent
from buildData import CFData, KGData, CKGData
from buildData.build import build_loader
import torch
from tqdm import tqdm
import random
import numpy as np
from copy import deepcopy
import os

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def convert_file(file_path, cfe_item, pos_item):
    users = []
    cfe_items = []
    pos_items=[]

    for key in cfe_item.keys():
        for item in cfe_item[key]:
            users.append([key])
            cfe_items.append([item])
        for item in pos_item[key]:
            pos_items.append([item])

    f = open(file_path,'w')
    """The explanation file is with header: u_ids, pos_ids, cfe_ids"""
    for x,y,z in zip(users, pos_items, cfe_items):
    	line = '{}'.format(x[0])+'\t{}'.format(y[0])+'\t{}'.format(z[0])+'\n'
    	f.write(line)

    f.close()

def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    for node in tqdm(graph.nodes, ascii=True, desc="Build sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(CKG.item_range[0], CKG.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        sampled_edge += random.sample(
            range(CKG.item_range[0], CKG.item_range[1] + 1), edge_threshold
        )

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


def get_cfe_list(train_loader, adj_matrix, edge_matrix, counter_example_file):
    tbar = tqdm(train_loader, ascii=True)

    print("\n---------Begin generating Counterfactual examples-----------\n")

    user_list = torch.tensor([], device='cuda')
    cfe_item_list = torch.tensor([], device='cuda')
    pos_item_list = torch.tensor([], device='cuda')

    for batch_data in tbar:
        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        users = batch_data["u_id"]
        pos = batch_data["pos_i_id"]

        selected_cfe_items_list, _ = sampler(batch_data, adj_matrix, edge_matrix)
        user_list = torch.cat([user_list, users])
        cfe_item_list = torch.cat([cfe_item_list, selected_cfe_items_list[-1, :]])
        pos_item_list = torch.cat([pos_item_list, pos])

    user_list = user_list.cpu().detach().numpy().tolist()
    cfe_item_list = cfe_item_list.cpu().detach().numpy().tolist()
    pos_item_list = pos_item_list.cpu().detach().numpy().tolist()

    user_cfe_items = {}
    user_pos_items = {}
    for i in user_list:
        user_cfe_items[i] = []
        user_pos_items[i] = []
    for i, user in enumerate(user_list):
        user_cfe_items[int(user)].append(cfe_item_list[i])
        user_pos_items[int(user)].append(pos_item_list[i])

    print("\n---------Saving file: Counterfactual examples-----------\n")
    convert_file(counter_example_file, user_cfe_items, user_pos_items)

    return user_cfe_items, user_pos_items


def get_des_aspect(in_file, out_file, entity_relation):

    lines = open(in_file, "r").readlines()

    u_ids = []
    pos_ids = []
    cfe_ids = []

    for l in lines:
        tmps = l.strip()
        inters = [int(float(i)) for i in tmps.split("\t")]

        u_id, pos_id, cfe_id = inters[0], inters[1], inters[2]
        u_ids.append(u_id)
        pos_ids.append(pos_id)
        cfe_ids.append(cfe_id)

    pos_attributes = []
    cfe_attributes = []

    for pos in pos_ids:
        pos_attribute = []
        for pos_atts in entity_relation[pos]:
            pos_attribute.append(pos_atts[0])
        pos_attributes.append(pos_attribute)

    for cfe in cfe_ids:
        cfe_attribute = []
        for cfe_atts in entity_relation[cfe]:
            cfe_attribute.append(cfe_atts[0])
        cfe_attributes.append(cfe_attribute)

    des_asp_list = []
    count_asp_list = []

    for pos_att, cfe_att in zip(pos_attributes, cfe_attributes):
        des_asp = set(pos_att) & set(cfe_att)
        count_asp = set(cfe_att) - des_asp

        des_asp_list.append(list(des_asp))
        count_asp_list.append(list(count_asp))

    f = open(out_file,'w')

    """The explanation file is with header: u_ids, pos_ids, cfe_ids , count_asp_list"""
    for x, y, z, i in zip(u_ids, pos_ids, cfe_ids , count_asp_list):

        line = '{}'.format(x)+'\t{}'.format(y)+'\t{}'.format(z)+'\t{}'.format(i)+'\n'
        f.write(line)

    f.close()

    return True


def build_random(in_file, out_file,  entity_range_start, entity_range_end):

    lines = open(in_file, "r").readlines()

    u_ids = []
    pos_ids = []
    cfe_ids = []

    for l in lines:
        tmps = l.strip()
        inters = [int(float(i)) for i in tmps.split("\t")]

        u_id, pos_id, cfe_id = inters[0], inters[1], inters[2]
        u_ids.append(u_id)
        pos_ids.append(pos_id)
        cfe_ids.append(cfe_id)


    random_attributes = []

    for cfe in cfe_ids:
        random_attribute =[]
        for i in range(10):
            random_attribute.append(random.randint(entity_range_start, entity_range_end))
        # random_attributes[cfe] = random.sample(range(entity_range_start, entity_range_end), 10)
        random_attributes.append(random_attribute)
        

    f = open(out_file,'w')

    """The explanation file is with header: u_ids, pos_ids, cfe_ids , random_asp_list"""
    for x, y, z, i in zip(u_ids, pos_ids, cfe_ids , random_attributes):

        line = '{}'.format(x)+'\t{}'.format(y)+'\t{}'.format(z)+'\t{}'.format(i)+'\n'
        f.write(line)

    f.close()

    return True


if __name__ == '__main__':
    args_config = parse_args()
    args_config.epoch = 1
    args_config.dataset='last-fm'
    print('Working on dataset: ', args_config.dataset)
    CKG = CKGData(args_config)
    CF = CFData(args_config)
    KG = KGData(args_config)
    entity_relation_dict = KG.kg_dict

    data_config = {
        "n_users": CKG.n_users,
        "n_items": CKG.n_items,
        "n_relations": CKG.n_relations + 2,
        "n_entities": CKG.n_entities,
        "n_nodes": CKG.entity_range[1] + 1,
        "item_range": CKG.item_range,
    }
    graph = deepcopy(CKG)
    train_loader, test_loader = build_loader(args_config=args_config, graph=graph)
    interactions = CF
    data_config["inter_matrix"] = torch.Tensor(interactions.train_inter_matrix)
    data_config["user_dict"] = interactions.train_user_dict

    recommender = Agent(data_config=data_config, args_config=args_config)
    sampler = KGPolicy_device(recommender, data_config, args_config)

    print("\n--------loading state dict--------\n")

    recommender.load_state_dict(torch.load("./weights/rec_last-fm_24_May.ckpt"))
    recommender.eval()

    sampler.load_state_dict(torch.load("./weights/sampler_last-fm_24-May.ckpt"))
    sampler.eval()

    for param in recommender.parameters():
        param.requires_grad = False

    for param in sampler.parameters():
        param.requires_grad = False

    explain_path = './explanation'
    counter_example_file = os.path.join(explain_path, 'examples_{}.txt'.format(args_config.dataset))
    counter_attributes_file = os.path.join(explain_path,'attributes_{}.txt'.format(args_config.dataset))

    for epoch in range(args_config.epoch):
        if epoch % args_config.adj_epoch == 0:
            adj_matrix, edge_matrix = build_sampler_graph(
                data_config["n_nodes"], args_config.edge_threshold, graph.ckg_graph
            )

            user_cfe_items, user_pos_items = get_cfe_list(train_loader, adj_matrix, edge_matrix, counter_example_file)

    print("\n---------Begin generating Counterfactual Attributes-----------\n")

    get_des_aspect(counter_example_file, counter_attributes_file, entity_relation_dict)

    #unzip the following to get random cfe attributes 
    
    # print("\n---------Begin generating Random Attributes-----------\n")
    # args_config = parse_args()
    # args_config.dataset='amazon-book'
    # args_config.epoch = 1
    # KG = KGData(args_config)
    # entity_range_start = KG.entity_range[0]
    # entity_range_end = KG.entity_range[1]

    # explain_path = './explanation'

    # counter_example_file = os.path.join(explain_path, 'counter_examples_{}.txt'.format(args_config.dataset))
    # random_attributes_file = os.path.join(explain_path,'random_attributes_{}.txt'.format(args_config.dataset))

    # build_random(counter_example_file, random_attributes_file, entity_range_start, entity_range_end)








