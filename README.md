Our code contains three important steps: 
1. Train black-box recommendation model; 
2. Train counterfactual reasoning agent; 
3. Genetrate counterfactual explanations. 

Step 1: black-box recommendation model
We provide two options for training the black-box recommendation model:
(1) Use arbitrary model that can produce the User and Item embeddings; 
you should firstly place it into BaseRecRepo-> to generate User and item embedding.

We provide an example: CliMF in the BaseRecRepo.
The CliMF directly optimizes the Top-$K$ recommendation by modeling parameters through maximizing the Mean Reciprocal Rank (MRR), which is a well-known information retrieval metric for capturing the performance of top-k recommendations.

To softly train the CliMF model, you should firstly generate .bat train files by
```
convert_file(train_file_path, CF.train_user_dict)
-> convert_file(test_file_path, CF.test_user_dict)
```

Then manually add the file head to the two files as required in  Orange.data.Table(files) function with:
user	item	relevance
c	c	c
row=1	col=1	class

After trained the User and Item embeddings, assign all_embed.data in recommender_agent.py with:
$ all_embed.data = torch.from_numpy(*your_trained_embedding) 

(2) You can also apply xavier on user/item embeddings to initialize embeddings. To do so, simply use:
$ nn.init.xavier_uniform_(all_embed) in recommender_agent.py

Step 2. counterfactual reasoning agent
(1) The configs for training the agent is placed in common/parser. Edit the parser with your preferred auguments. 
Important args are:
 $ --data_path: Input data path.
 $ --dataset: Choose a dataset.
 $ --emb_size: Embedding size.
 $ --counter_threshold: counterfactual top-k rank threshold.
 $ --slr: Learning rate for sampler.
 $ --rlr: Learning rate recommender.
 $ --inter_threshold: interaction threshold to construct interaction matrix.
 $ --interaction_len: Input vector length of user embedding.
 $ --k_step: k step from current positive items
 
(2) After configuring the parser, train the agent with:
 $ python train_RL_agent.py
 
(3) You can get the trained sampler and recommender models saved in ./weights/.

Step 3. generate counterfactual aspect-level explanations:
(0) load_state_dict from the trained sampler and recommender models from step 2 by replacing the paths in line 195 and line 198 of generate_explanation.py

(1) run: 
$ python generate_explanation.py
output: .txt files of counterfactual_examples_{dataset}.txt and counterfactual_attributes_{dataset}.txt, which are placed in ./explanations/

(2) We also provide the user-side evaluation of the output counterfactual attributes.
To evaluate, you should firstly extract ground-truth negative atrributes of user-item interactions from SOTA models (e.g., we provide an example negative attibutes in GT_attributes/, which is extracted from KGpolicy).
Then replace the ground_truth_path in user_side_evaluation.py with your trained ground truth attribute file
Finally, run:
$ python user_side_evaluation 


Cite information:
