import Orange
import numpy as np
from buildData import CFData
from common import parse_args
from BaseRecRepo.model.CliMF import CLiMFLearner
from opt.sgd_optimizers import Adam
import torch
import os
from pathlib import Path
import pickle

def convert_file(file_path,data):
    users = []
    items = []

    for key in data.keys():
        for item in data[key]:
            users.append([key])
            items.append([item])
    classes = [[1] for i in range(len(users))]

    f = open(file_path,'w')
    for x,y,z in zip(users,items,classes):
        line = '{}'.format(x[0])+'\t{}'.format(y[0])+'\t{}'.format(z[0])+'\n'
        f.write(line)
    f.close()

def save_model(file_name, model, args):
    if not os.path.isdir(args.out_rec):
        os.mkdir(args.out_rec)

    model_file = Path(args.out_rec + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model, model_file)


def train_base_rec(train_data_PATH, test_data_PATH,args):
    train_data = Orange.data.Table(train_data_PATH)
    test_data = Orange.data.Table(test_data_PATH)

    # Train recommender
    opt = Adam(learning_rate=args.base_learning_rate,
               beta1=0.9, beta2=0.999, epsilon=1e-8)
    learner = CLiMFLearner(optimizer=opt, num_factors=args.emb_size, num_iter=args.num_iter,
                           learning_rate=args.base_learning_rate, lmbda=args.lmbda)

    #num_iter = 10, learning_rate = 0.0001, lmbda = 0.001
    recommender = learner(train_data)

    # Sample users
    num_users = len(recommender.U)
    num_samples = int(num_users * 0.2)
    users_sampled = np.random.choice(np.arange(num_users), num_samples)

    # Compute Mean Reciprocal Rank (MRR)
    mrr, _ = recommender.compute_mrr(data=test_data, users=users_sampled)
    print('learned embedding for Users', recommender.U.shape)
    print('learned embedding for Items', recommender.V.shape)
    print('MRR: %.4f' % mrr)

    pkl_name = 'recommender_model_' + str(args.emb_size) + '.pkl'

    Pkl_Filename = "Pickle_RL_Model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(learner, file)


    # save_model(pkl_name, learner, args)
    # torch.save(learner, pkl_name)

if __name__ == '__main__':

    """initialize args and dataset"""
    args_config = parse_args()
    print('Training Black-Box Recommendation model with config:', args_config)

    CF = CFData(args_config)

    train_file_path = os.path.join("processed_data",args_config.dataset,"train_RecBase.tab")
    test_file_path = os.path.join("processed_data",args_config.dataset ,"test_RecBase.tab")

    # print("Converting files begin")
    # convert_file(train_file_path,CF.train_user_dict)
    # convert_file(test_file_path, CF.test_user_dict)

    print('Begin train the Black-Box Recommendation model.\n')
    train_base_rec(train_file_path, test_file_path,args_config)


