

echo svhn
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 svhn
echo cifar100
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 cifar100
echo celeba
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 celeba

