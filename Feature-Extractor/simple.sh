#vanillar resnet on butterfly

CUDA_VISIBLE_DEVICES=4 qlua main.lua -retrain ../models/resnet-50.t7 -data /mv_users/peiguo/dataset/cub-fewshot/full/ -batchSize 16 -dispIter 200 -LR 1e-3 -nClasses 200
