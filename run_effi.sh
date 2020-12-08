#model_arr=("EffiB0_B32_T8" "EffiB0_B32_T8_base10" "EffiB0_B32_T8_div3" "EffiB0_B32_T8_mul3" "EffiB0_B32_T8_div10" "EffiB0_B32_T8_mul10")
#model_arr=("EffiB0_B32_T8" "EffiB0_B32_T8_mul10")
#model_arr=("EffiB0_B32_T8_base1")
#model_arr=("EffiB0_B32_T8_base1" 'EffiB0_B32_T8_mul10_base1' 'EffiB0_B32_T8_mul10_base3' 'EffiB0_B32_T8_mul30' 'EffiB0_B32_T8_mul10' 'EffiB0_B32_T8_mul10_shuffle')
#model_arr=('EffiB0_B32_T8_mul10_cosine')
#model_arr=('EffiB0_B32_T8_mul10_mix_alpha10' 'EffiB0_B32_T8_mul10_mix_alpha15' 'EffiB0_B32_T8_mul10_mix_alpha05' 'EffiB0_B32_T8_mul10_mix_alpha05_min05' 'EffiB0_B32_T8_mul10_mix_alpha15_min05')
#model_arr=( 'EffiB0_B32_T8_mul10_mix_alpha10_min05')
#model_arr=('EffiB0_B32_T8_mul10_PS_s8' 'EffiB0_B32_T8_residual_mul10' 'EffiB0_B32_T8_mul10_PS_s16')
#model_arr=('EffiB0_B32_T8_IR64_mul10')
#model_arr=('EffiB0_B32_T8_mul10_hswish' 'EffiB0_B32_T8_mul10_mix_alpha05_min05_prob01' 'EffiB0_B32_T8_mul10_mix_alpha05_min05_prob02')
#model_arr=('EffiB0_B32_T8_mul10_wde2' 'EffiB0_B32_T8_mul10_wde3' 'EffiB0_B32_T8_mul10_wde5')
#model_arr=('EffiB0_B32_T8_mul10_freeze_stage2' 'EffiB0_B32_T8_mul10_freeze_stage1')
#model_arr=('EffiB0_B32_T8_mul10_PDP' 'EffiB0_B32_T8_mul10_freeze_stem' 'EffiB0_B32_T8_mul10_freeze_stage2' 'EffiB0_B32_T8_mul10_freeze_stage1' 'EffiB0_B32_T8_mul10_freeze_stage2' 'EffiB0_B32_T8_mul10_MP')
#model_arr=('EffiB0_B32_T8_mul10_NB_112' 'EffiB0_B32_T8_mul10_NB_256' 'EffiB0_B32_T8_mul10_NCA')
#model_arr=('simple_matrix_res101_normal_T4_AA_k3')
#model_arr=('EffiB0_B32_T8_mul10_Simple_matrix' 'EffiB0_B32_T8_mul10_Simple_matrix_BN_before' 'EffiB0_B32_T8_mul10_Simple_matrix_Diou' 'EffiB0_B32_T8_mul10_Simple_matrix_Giou' 'EffiB0_B32_T8_mul10_Simple_matrix_linear_reg' 'EffiB0_B32_T8_mul10_Simple_matrix_one_tower')
#model_arr=('EffiB0_B32_T8_mul10_Simple_matrix_BN_before_linear_reg_mul2_288')
model_arr=('EffiB0_B32_T8_mul10_Simple_matrix_BN_before_linear_reg_mul2_320' 'EffiB0_B32_T8_mul10_Simple_matrix_BN_before_linear_reg_mul2_352')
for((i=0;i<${#model_arr[@]};i++))
do
  #${#array[@]}获取数组长度用于循环
  echo ${model_arr[i]};
  python tracking/onekey_X.py --cfg experiments/train/${model_arr[i]}.yaml
done;


#for i in $(seq 30 50)
#do
#./azcopy_linux_amd64_10.2.1/azcopy copy https://test26183922662.blob.core.windows.net/zhipeng/checkpoints/${model}/checkpoint_e${i}.pth .;
#done
#mv checkpoints/${model} /home/alphabin/test/AutoTrack/snapshot/${model}
#
## step3
#sleep 3h
#model="Mobile_Baseline_Diou"
#for i in $(seq 30 50)
#do
#./azcopy_linux_amd64_10.2.1/azcopy copy https://test26183922662.blob.core.windows.net/zhipeng/checkpoints/${model}/checkpoint_e${i}.pth .;
#done
#mv checkpoints/${model} /home/alphabin/test/AutoTrack/snapshot/${model}
#cd /home/alphabin/test/AutoTrack
#python tracking/onekey_X.py --cfg experiments/train/${model}.yaml
#cd /home/alphabin/test/MSRATools

#ln -s /home/alphabin/test/MSRATools/checkpoints/${model} /home/alphabin/test/AutoTrack/snapshot/${model}
#cd /home/alphabin/test/AutoTrack
#python tracking/onekey_X.py --cfg experiments/train/${model}.yaml
