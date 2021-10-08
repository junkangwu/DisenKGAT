# DisenKGAT
<h1 align="center">
  DisenKGAT
</h1>

<h4 align="center">DisenKGAT: Knowledge Graph Embedding with Disentangled
Graph Attention Network</h4>


<h2 align="center">
  Overview of CompGCN
  <img align="center"  src="./overall_graph33_page-0001.jpg" alt="...">
</h2>

<!-- ![Image](./overall_graph33_page-0001.jpg) -->
### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use FB15k-237 and WN18RR dataset for knowledge graph link prediction. 
- FB15k-237 and WN18RR are included in the `data` directory. 

### Training model:

- Install all the requirements from `requirements.txt.`

- Execute `./setup.sh` for extracting the dataset and setting up the folder hierarchy for experiments.

- Commands for reproducing the reported results on link prediction:

  ```shell
  ##### with TransE Score Function
  # DisenKGAT (Composition: Subtraction)
  python run.py -epoch 1500 -name TransE_sub_K2_D200 -model disenkgat\
        -hid_drop 0.1 -gcn_drop 0.3 \
        -score_func transe -opn sub -data FB15k-237  -gamma 9.0 -max_gamma 5. \
        -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 200 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 -mi_drop

  
  
  # DisenKGAT (Composition: Multiplication)
  python run.py -epoch 1500 -name Mult_InteractE_FB15k_K2_D200_club_b_mi_drop -model disenkgat\
        -score_func interacte -opn mult -gpu 0 \
        -data FB15k-237 -gcn_drop 0.4 \
        -ifeat_drop 0.4 -ihid_drop 0.3 -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_b\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -mi_drop
  
  # DisenKGAT (Composition: Crossover Interaction)
  python run.py -epoch 1500 -name InteractE_FB15k_K2_D200_club_b_mi_drop -model disenkgat\
        -score_func interacte -opn cross -gpu 0 \
        -data FB15k-237 -gcn_drop 0.4 \
        -ifeat_drop 0.4 -ihid_drop 0.3 -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_b\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -mi_drop
  
  ##### with DistMult Score Function
  # DisenKGAT (Composition: Subtraction)
  python run.py -epoch 1500 -name Distmult_cross_K2_D200 -model disenkgat\
        -hid_drop 0.3 -gcn_drop 0.4 \
        -score_func distmult -opn cross -data FB15k-237  -gcn_layer 2 \
        -batch 128 -test_batch 128 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 66 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 
  
  # DisenKGAT (Composition: Multiplication)
  python run.py -epoch 1500 -name Distmult_mult_K2_D200 -model disenkgat\
        -hid_drop 0.3 -gcn_drop 0.4 \
        -score_func distmult -opn mult -data FB15k-237  -gcn_layer 2 \
        -batch 128 -test_batch 128 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 66 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 
  
  # DisenKGAT (Composition: Crossover Interaction)
  python run.py -epoch 1500 -name Distmult_sub_K2_D200 -model disenkgat\
        -hid_drop 0.3 -gcn_drop 0.4 \
        -score_func distmult -opn sub -data FB15k-237  -gcn_layer 2 \
        -batch 128 -test_batch 128 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 66 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 
  
  ##### with ConvE Score Function
  # DisenKGAT (Composition: Subtraction)
  python run.py -epoch 1500 -name SUB_InteractE_FB15k_K2_D200_club_b_mi_drop -model disenkgat\
        -score_func interacte -opn sub -gpu 0 \
        -data FB15k-237 -gcn_drop 0.4 \
        -ifeat_drop 0.4 -ihid_drop 0.3 -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 100 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_b\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -mi_drop
  
  # DisenKGAT (Composition: Multiplication)
  python run.py -epoch 1500 -name TransE_mult_K2_D200 -model disenkgat\
        -hid_drop 0.1 -gcn_drop 0.3 \
        -score_func transe -opn mult -data FB15k-237  -gamma 9.0 -max_gamma 5. \
        -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 200 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 -mi_drop
  
  # DisenKGAT (Composition: Crossover Interaction)
  python run.py -epoch 1500 -name TransE_cross_K2_D200 -model disenkgat\
        -hid_drop 0.1 -gcn_drop 0.3 \
        -score_func transe -opn cross -data FB15k-237  -gamma 9.0 -max_gamma 5. \
        -batch 2048 -test_batch 2048 -iker_sz 9 -head_num 1 -gamma_method norm \
        -num_workers 10 -attention True -early_stop 200 -num_factors 2 -logdir ./log/ \
        -alpha 1e-1 -no_params -init_dim 200 -lr 1e-3 -gcn_dim 200 -mi_train -mi_method club_s\
        -att_mode dot_weight -mi_epoch 1 -score_method dot_rel -score_order after -init_gamma 9.0 -mi_drop
  
  ##### Overall BEST:
  python run.py -name best_model -score_func conve -opn corr 
  ```

  - `-score_func` denotes the link prediction score score function 
  - `-opn` is the composition operation used in **CompGCN**. It can take the following values:
    - `sub` for subtraction operation:  Φ(e_s, e_r) = e_s - e_r
    - `mult` for multiplication operation:  Φ(e_s, e_r) = e_s * e_r
    - `corr` for circular-correlation: Φ(e_s, e_r) = e_s ★ e_r
  - `-name` is some name given for the run (used for storing model parameters)
  - `-model` is name of the model `compgcn'.
  - `-gpu` for specifying the GPU to use
  - Rest of the arguments can be listed using `python run.py -h`


## Acknowledgement
The project is built upon [COMPGCN](https://github.com/malllabiisc/CompGCN)


For any clarification, comments, or suggestions please create an issue or contact me.
