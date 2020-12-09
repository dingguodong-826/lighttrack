model_name="supernet_train"
FLOPs=600000000
flops_name="600M"
python Evolution/src/Search/search_DP_advanced.py --log_dir Evolution/${model_name}/${flops_name}/log \
--checkpoint_path snapshot/${model_name}/checkpoint_e30.pth --flops_limit ${FLOPs} --max_flops_backbone 470 \
--model_name ${model_name} --search_back 1 --search_ops 1 --search_head 1

python Evolution/src/Evaluation/eval.py