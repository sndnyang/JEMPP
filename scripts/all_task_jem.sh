
model="$1";

dir=$(dirname "${model}")
mkdir -p "$dir"/distal;
mkdir -p "$dir"/samples;

rm "$model"_eval.log 2>/dev/null;

function accuracy_calibration() {
  # 1. Evaluate Accuracy on test dataset
  # 2. Calibration with Expected Calibration Error (ECE)
  echo "Eval Accuracy and Calibration"
  echo "Eval Accuracy and Calibration" >> "$model"_eval.log

  x=`python eval_wrn_ebm.py --load_path "$model" --eval cali --dataset cifar_test`
  echo "$x"
  echo "$x" >> "$model"_eval.log
}

function is_fid() {
  # 3. Inception Score and FID score
  echo "IS and FID"
  echo "IS and FID"  >> "$model"_eval.log

  echo "IS replay buffer"
  echo "IS replay buffer"  >> "$model"_eval.log

  if [ ! -d "$dir"/buffer_samples ]
  then
    echo "the buffer_samples directory doesn't exist"
    x=`python eval_wrn_ebm.py --load_path "$model" --eval cond_is --save_dir "$dir"/buffer_samples --buffer_size 10000 --reinit_freq 0.05`
    echo "$x"
    echo "$x"  >> "$model"_eval.log
    mkdir -p "$dir"/fid_buffer_samples
    cd "$dir"
    find buffer_samples -name "*.*" | xargs cp -t fid_buffer_samples
    cd -
  fi

  x=`python eval_is.py "$dir"/buffer_samples`
  echo "$x"
  echo "$x" >> "$model"_eval.log
  echo "FID"
  echo "FID"  >> "$model"_eval.log
  x=`python Task/fid_score.py "$dir"/fid_buffer_samples data/fid_stats_cifar10_train.npz`
  echo "$x"
  echo "$x" >> "$model"_eval.log

#  echo "IS generated from scratch"
#  echo "IS generated from scratch"  >> "$model"_eval.log
#  if [ ! -d "$dir"/samples ]
#  then
#    x=`python eval_wrn_ebm.py --load_path "$model" --eval cond_is --save_dir "$dir"/samples --n_sample_steps 10000 --buffer_size 10000 --n_steps 40 --print_every 10 --reinit_freq 0.05 --fresh_samples`
#    echo "$x" >> "$model"_eval.log
#    mkdir -p "$dir"/fid_samples
#    cd "$dir"
#    cp samples/*/* fid_samples
#    cd -
#  fi
#
  x=`python eval_is.py "$dir"/samples_1000`
  echo "$x"
  echo "$x" >> "$model"_eval.log

  echo "FID"
  echo "FID"  >> "$model"_eval.log
  x=`python Task/fid_score.py "$dir"/fid_samples_1000 data/fid_stats_cifar10_train.npz`
  echo "$x"
  echo "$x" >> "$model"_eval.log
}

function ood() {

  # 4. Out-of-Distribution
  # 4.1 Approximate MASS
  echo "OOD"
  echo "OOD"  >> "$model"_eval.log
  x="python eval_wrn_ebm.py --load_path "$model" --eval OOD --dataset cifar_100"
  echo "$x"
  x=`python eval_wrn_ebm.py --load_path "$model" --eval OOD --ood_dataset cifar_100`
  echo "$x"
  echo "$x" >> "$model"_eval.log
  x="python eval_wrn_ebm.py --load_path "$model" --eval logp_hist --datasets cifar10 cifar100 --save_dir "$dir"/ood_cifar100"
  echo "$x"
  x=`python eval_wrn_ebm.py --load_path "$model" --eval logp_hist --datasets cifar10 cifar100 --save_dir "$dir"/ood_cifar100`
  echo "$x"
  echo "$x" >> "$model"_eval.log

  x=`python eval_wrn_ebm.py --load_path "$model" --eval OOD --ood_dataset svhn`
  echo "$x"
  echo "$x" >> "$model"_eval.log
  x=`python eval_wrn_ebm.py --load_path "$model" --eval logp_hist --datasets cifar10 svhn --save_dir "$dir"/ood_svhn`
  echo "$x"
  echo "$x" >> "$model"_eval.log

}

function robust() {

  # 5. Robustness
  # 5.1 Distal Adversaries
  # distal adversary
  # -t target class 1, namely "car"
  echo "Distal Adversaries"
  echo "Distal Adversaries" >> "$model"_eval.log
#  x=`python distal_jem.py  --load_path "$model" --conf=0.03 -t 1`
#  echo "$x"
#  echo "$x" >> "$model"_eval.log
}

# 6. Label Corruption
# accuracy_calibration
is_fid
# ood
# robust

