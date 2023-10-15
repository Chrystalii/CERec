<!-- GETTING STARTED -->
## Getting Started

Our code contains three important steps: 
1. Pretrain recommendation model (optional); 
2. Train counterfactual reasoning agent; 
3. Generate counterfactual explanations.

### Step 1: Pretraining black-box recommendation model (optional)

We provide an example: CliMF in the BaseRecRepo, for training the black-box recommendation model:
The CliMF directly optimizes the Top-$K$ recommendation by modeling parameters through maximizing the Mean Reciprocal Rank (MRR).

1. To train the CliMF model, you should first generate .bat train files by
    ```
   convert_file(train_file_path, CF.train_user_dict)

   convert_file(test_file_path, CF.test_user_dict)
     ```
2. Then manually add the file head to the two files as required in  Orange.data.Table(files) function with:
  ```
   user   item  relevance
   c  c  c
   row=1 col=1 class
  ```
3. Train CliMF and save recommendation model parameters:
  ```
   python train_base_rec.py 
  ```
4. Load pretraining embeddings into CERec: in recommender_agent.py, load embeddings with:
  ```
   all_embed.data = torch.from_numpy(*your_trained_embedding_file) 
  ```
If you do not wish to use pertaining models, apply Xavier on user/item embeddings to initialize embeddings. To do so, simply use:
  ```
   nn.init.xavier_uniform_(all_embed) in recommender_agent.py 
  ```

### Step 2. Counterfactual reasoning agent (mandatory)

1. Edit the configs for training RL agent:
The parser file is placed in common/parser; Important args are:
  ```
  --data_path: Input data path.
  --dataset: Choose a dataset.
  --emb_size: Embedding size.
  --counter_threshold: counterfactual top-k rank threshold.
  --slr: Learning rate for sampler.
  --rlr: Learning rate recommender.
  --inter_threshold: interaction threshold to construct interaction matrix.
  --interaction_len: Input vector length of user embedding.
  --k_step: k step from current positive items
   ```
2. We also provide alternatives for variants; you may choose different graph learning methods, samplers, recommendation models and reward functions by:
  ```
    #Graph Learning Module
    --GCN: GCN representation learning operator, options are [SAGE, GCN, GNN, SG, LG, GAT, Trans]

    #Agent
    --reward: reward function, options are [all, R-reward, S-reward]

    #Sampler: 
    --sampler: samplers, options are [CPS, 1-Uniform, 2-Uniform]

    #Recommendation Model
    --recommender: Recommenders, support [MF, BPR, NeuMF, KGAT]
  ```
3. After configuring the parser, train the agent with:
  ```
 python train_RL_agent.py --your-preferred-configs
  ```
4. You can get the trained sampler and recommender models saved in ./weights/.

### Step 3: Generate counterfactual attribute-level explanations

1. load_state_dict from the trained sampler and recommender models from step 2 by replacing the file names in generate_explanation.py
2. run
  ```
 python generate_explanation.py
    ```
  ```
  ```
The output: .txt files of counterfactual_examples_{dataset}.txt; and counterfactual_attributes_{dataset}.txt, which are placed in ./explanations/
  ```

### Cite information:
To be released after paper acceptance


