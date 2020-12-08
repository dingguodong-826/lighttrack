CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/distributed_test.sh 8 --data ./data/imagenet/  --config './lib/configs/test.yaml' --resume './experiments/ckps/42.pth.tar' --model_selection 42
