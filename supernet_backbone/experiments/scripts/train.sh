#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/distributed_train.sh 8 --data ./data/imagenet/ --model_selection 470 --config ./lib_back/configs/470.yaml --path_name back_02320024044124423050+cls_002001122+reg_200000201
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./tools/distributed_train.sh 7 --data ./data/imagenet/ \
#--model_selection 470 --config ./lib_back/configs/470.yaml --path_name back_02041520113420343540+cls_212202200+reg_002221110_ops_33

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/distributed_train.sh 8 --data ./data/imagenet/ \
--model_selection 600 --config ./lib_back/configs/600.yaml --path_name back_0114413213503544142140+cls_2021002000111+reg_0110021100200+ops_30