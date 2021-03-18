
python train_jempp_simple.py --lr=.0001 --dataset=cifar10 --optimizer=adam \
 --p_x_weight=1.0 --p_y_given_x_weight=1.0 --p_x_y_weight=0.0 --sigma=.03 --width=10 --depth=28 --plot_uncond --warmup_iters=1000 \
 --log_arg=method-n_steps-sgld_lr --method=yopo --log_dir=./run \
 --print_every=100 \
 --n_epochs=150 --decay_epochs 60 90 120 140 \
 --n_steps=5 \
 --in_steps=10 \
 --sgld_lr=1 \
 --gpu-id=3

# YOPO  5x3  5 outer loop 3 inner loop
# p_x_weight -> log p(x)
# n_steps  for SGLD
# clip_norm: 0, no norm,  >0 clipping gradients of SGLD, -1 apply SN on gradients of SGLD
# start_sgld:  pretrain several epochs only with log p(y|x)
# print_every: print more logs
