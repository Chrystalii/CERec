import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RUNING CEREC")

    # ------------------------- BaseRec --------------------------------------------
    parser.add_argument("--num_iter", type=int, default=1, help="num_iter for Base Rec")
    parser.add_argument('--base_learning_rate', type=float, default=0.0001, help="base_learning_rate for Base Rec")
    parser.add_argument('--lmbda', type=float, default= 0.001, help="lambda for Base Rec")
    parser.add_argument( "--out_rec", type=str, default="./save/", help="output directory for black-box rec model")

    # -------------------------Data set --------------------------------------------
    parser.add_argument(
        "--data_path", nargs="?", default="./Data/", help="Input data path."
    )
    parser.add_argument(
        "--dataset", nargs="?", default="last-fm", help="Choose a dataset."
    )
    parser.add_argument("--emb_size", type=int, default=128, help="Embedding size.")
    parser.add_argument(
        "--regs",
        nargs="?",
        default="1e-5",
        help="Regularization for user and item embeddings.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--k_neg", type=int, default=1, help="number of negative items in list"
    )

    # ------------------------- Agents -----------------------------------------
    parser.add_argument("--counter_threshold", type=float, default=0.001, help="counterfactual top-k rank threshold") 
    parser.add_argument(
        "--slr", type=float, default=0.0001, help="Learning rate for sampler."
    ) 
    parser.add_argument(
        "--rlr", type=float, default=0.0001, help="Learning rate recommender."
    )
    parser.add_argument(
        "--inter_threshold",
        type=int,
        default=32,
        help="interaction threshold to construct interaction matrix",
    )
    parser.add_argument(
        "--interaction_len", type=int, default=50, help="Input vector length of user embedding."
    )

    # ------------------------- sampler --------------------------------------------

    parser.add_argument(
        "--edge_threshold",
        type=int,
        default=64,
        help="edge threshold to filter knowledge graph",
    )
    parser.add_argument(
        "--num_sample", type=int, default=32, help="number fo samples from gcn"
    )
    parser.add_argument(
        "--k_step", type=int, default=4, help="k step from current positive items"
    )
    parser.add_argument(
        "--in_channel", type=str, default="[64, 32]", help="input channels for gcn"
    )
    parser.add_argument(
        "--out_channel", type=str, default="[32, 64]", help="output channels for gcn"
    )
    parser.add_argument(
        "--pretrain_s",
        type=bool,
        default=False,
        help="load pretrained sampler data or not",
    )

    # -------------------------Experiments--------------------------------------------
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="batch size for training."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1024, help="batch size for test"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads.")
    parser.add_argument("--epoch", type=int, default=400, help="Number of epoch.")
    parser.add_argument("--show_step", type=int, default=1, help="test step.")
    parser.add_argument(
        "--adj_epoch", type=int, default=1, help="build adj matrix per _ epoch"
    )
    parser.add_argument(
        "--pretrain_r", type=bool, default=False, help="use pretrained model or not"
    )
    parser.add_argument(
        "--freeze_s",
        type=bool,
        default=False,
        help="freeze parameters of recommender or not",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/best_fm.ckpt",
        help="path for pretrain model",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )
    parser.add_argument("--flag_step", type=int, default=64, help="early stop steps")
    parser.add_argument(
        "--gamma", type=float, default=1, help="gamma for reward accumulation"
    ) #0.99
    parser.add_argument('--random_range', type=float, default=0.01,
                    help='[-random_range, random_range] for initialization')

    # ------------------------- experimental settings specific for testing ---------------------------------------------
    parser.add_argument(
        "--Ks", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list"
    )

    return parser.parse_args()
