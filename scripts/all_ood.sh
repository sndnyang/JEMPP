

echo px

echo svhn
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset svhn --sigma 0
echo cifar_interp
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_interp --sigma 0
echo cifar100
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_100 --sigma 0
echo celeba
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset celeba --sigma 0

echo py
#
echo svhn
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset svhn --score_fn py --sigma 0
echo cifar_interp
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_interp --score_fn py --sigma 0
echo cifar100
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_100 --score_fn py --sigma 0
echo celeba
python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset celeba --score_fn py --sigma 0

#echo pxgrad
#
#echo svhn
#python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset svhn --score_fn pxgrad --sigma 0
#echo cifar_interp
#python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_interp --score_fn pxgrad --sigma 0
#echo cifar100
#python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset cifar_100 --score_fn pxgrad --sigma 0
#echo celeba
#python eval_wrn_yopo.py  --norm batch --model yopo --load_path $1 --eval OOD --ood_dataset celeba --score_fn pxgrad --sigma 0
#
