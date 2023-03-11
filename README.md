# JEM++: Improved Techniques for Training JEM

Official code for the paper [JEM++: Improved Techniques for Training JEM](https://arxiv.org/abs/2109.09032)

## Usage

### Prerequisite 

model and mean/cov data in  https://1drv.ms/u/s!AgCFFlwzHuH8l0QWQ8E2aMtC0ApO?e=pE43fR

Pretrained model is *jempp_M10.pt*

1. Install from the requirements.txt, please check the details ```pip install -r requirements.txt```
2. Download the mean/covariance (*cifar10_mean/cov.pt*) data from above link

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
 --n_steps=10 \
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
ratio >= buffer size, use all images.
```markdown
python eval_jempp.py --load_path /PATH/TO/YOUR/MODEL.pt --eval fid --model yopo --norm batch --ratio 10000  
# jempp_M10.pt's FID is 36.5 
```

## Demo

Trained model samples from scratch

Gif:

![Demo_Gif](https://user-images.githubusercontent.com/2310591/141198046-486e0413-f53b-40c1-889e-f228d05fb3f9.gif)

Video:



https://user-images.githubusercontent.com/2310591/141198289-4695fa50-9c20-4d9c-bf41-257cae8b4d38.mp4



## Citation

If you found this work useful and used it in your research, please consider citing this paper.
```
@article{yang2021jempp,
    title={JEM++: Improved Techniques for Training JEM},
    author={Xiulong Yang and Shihao Ji},
    journal={International Conference on Computer Vision (ICCV)},
    month={Oct.},
    year={2021}
}
```
