# JEM++: Improved Techniques for Training JEM

## Usage

### Prerequisite 

model and mean/cov data in  https://1drv.ms/u/s!AgCFFlwzHuH8l0QWQ8E2aMtC0ApO?e=pE43fR

1. Install from the requirements.txt, please check the details ```pip install -r requirements.txt```
2. Download the mean/covariance data from above link

### Training

To train a model on CIFAR10 as in the paper, please refer to scripts/cifar10.sh

```markdown
python train_jempp.py --dataset=cifar10 \
 --lr=.1 --optimizer=sgd \
 --p_x_weight=1.0 --p_y_given_x_weight=1.0 --p_x_y_weight=0.0 \
 --sigma=.03 --width=10 --depth=28 \
 --plot_uncond --warmup_iters=1000 \
 --log_arg=JEMPP-n_steps-in_steps-pyld_lr \
 --model=yopo \
 --norm batch \
 --print_every=100 \
 --n_epochs=150 --decay_epochs 50 100 125 \
 --n_steps=5 \
 --in_steps=5 \
 --pyld_lr=0.2 \
 --gpu-id=3
```

### Evaluation

To evaluate the classifier (on CIFAR10):
```markdown
python eval_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test --model yopo --norm batch
```

To evaluate the FID in the replay buffer (on CIFAR10):
```markdown
python eval_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval fid --model yopo --norm batch
```
To do OOD detection (on CIFAR100)
```markdown
python eval_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval OOD --ood_dataset cifar_100 --model yopo --norm batch
```
To generate a histogram of OOD scores like Table 2
```markdown
python eval_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn  --model yopo --norm batch
```

