export CUDA_VISIBLE_DEVICES=2

python llama3_summarization_finetune.py \
	--lora_r 8 \
	--pkg_prefix "summarization_llama3" \
	--pretrained_ckpt "meta-llama/Meta-Llama-3-8B-Instruct" \
	--epochs 10 \
	--dropout 0.0 \
	--dataset "MTS_Dialogue" \
	--train_file "/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv" \
	--per_device_train_batch_size 6

	#--train_file "/home/data2/yongfeng/code3/MTS-Dialog/Augmented-Data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv" \
	#--pretrained_ckpt unsloth/llama-3-8b-bnb-4bit \
	#--pretrained_ckpt ../../llama/llama-2-7b-hf\
    	#--dataset aci-bench
    	#--train_file /home/data2/yongfeng/code3/aci-bench/data/challenge_data/train.csv"
#python llama2_summarization_inference2.py \
#	--experiment_dir experiments/summarization_with_preprocess_csv_data_epochs-10_rank-8_dropout-0.1-None
