
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
 --n_steps=10 \
 --in_steps=5 \
 --pyld_lr=0.2 \
 --gpu-id=3
