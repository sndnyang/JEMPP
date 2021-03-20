# JEM++/PYLD - Improved Joint Energy Models and SGLD

Includes scripts for training JEM (Joint-Energy Model), evaluating models at various tasks, and running adversarial attacks.

## Usage

### Prerequisite 

model and mean/cov data in  https://1drv.ms/u/s!AgCFFlwzHuH8l0QWQ8E2aMtC0ApO?e=pE43fR

1. Install from the requirements.txt, please check in details ```pip install -r requirements.txt```
2. Download the mean/covariance data from above link, or generate from the notebook notebook/TODO.ipynb


### Training

To train a model on CIFAR10 as in the paper, please refer to scripts/cifar10.sh

```markdown
python train_wrn_jempp.py --lr .0001 --dataset cifar10 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 \
       --sigma .03 --width 10 --depth 28 --plot_uncond --warmup_iters 1000 
```

### Evaluation

To evaluate the classifier (on CIFAR10):
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test
```
To do OOD detection (on CIFAR100)
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval OOD --ood_dataset cifar_100
```
To generate a histogram of OOD scores like Table 2
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn 
```

To generate the xxx in the gif
```shell script
python eval_wrn_jempp.py --eval uncond_samples --n_sample_steps 100 --print_every 1 --reinit_freq 0 --model yopo --norm batch --load_path /path/to/your.pt  --n_steps 40 --buffer_size 100 --batch_size 100 --sgld_std 0
```

To generate new unconditional samples
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval uncond_samples --n_sample_steps {THE_MORE_THE_BETTER (1000 minimum)} --buffer_size 10000 --n_steps 40 --print_every 100 --reinit_freq 0.05
```
To generate conditional samples from a saved replay buffer
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples 
```
To generate new conditional samples
```markdown
python eval_wrn_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --n_sample_steps {THE_MORE_THE_BETTER (1000 minimum)} --buffer_size 10000 --n_steps 40 --print_every 10 --reinit_freq 0.05 --fresh_samples
 ```


### Attacks

To run Linf attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /PATH/TO/YOUR/MODEL.pt --exp_name /YOUR/EXP/NAME --n_steps_refine 1 --distance Linf --random_init --n_dup_chains 5 --base_dir /PATH/TO/YOUR/EXPERIMENTS/DIRECTORY
```
To run L2 attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /cloud_storage/BEST_jempp.pt --exp_name rerun_jempp_1_step_5_dup_l2_no_sigma_REDO --n_steps_refine 1 --distance L2 --random_init --n_dup_chains 5 --sigma 0.0 --base_dir /cloud_storage/adv_results &
 ```
 

Happy Energy-Based Modeling! 
