model_name=back_04502514044521042540+cls_211000022+reg_100000111_ops_32_SP_lr_mul2
path_name=back_04502514044521042540+cls_211000022+reg_100000111_ops_32
# test
python tracking/test_ocean_X_HE.py --arch LightTrackM_Subnet --dataset VOT2019 \
--resume snapshot/${model_name}/checkpoint_e32.pth \
--stride 16 --effi 0 --path_name ${path_name}
# evaluation
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 \
--tracker_result_dir result/VOT2019/${model_name}/ \
--trackers ${model_name}