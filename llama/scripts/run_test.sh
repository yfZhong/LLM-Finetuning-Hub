
export CUDA_VISIBLE_DEVICES=3
#'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
#'mixedbread-ai/mxbai-embed-large-v1'
#'intfloat/e5-mistral-7b-instruct'


python llama2_summarization_inference_with_postprocess.py \
	--experiment_dir experiments/summarization_llama3_prompt-v2_ds-MTS_Dialogue_epochs-1_rank-8_dropout-0.1 \
	--dataset "MTS_Dialogue" \
	--test_file "/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv" \
	--test_sample_num 200 \
	--temperature 0.01 \
	--repetition_penalty 1.05 \
	--max_new_tokens 256 \
	--num_candidates 1 \
	--num_generated_query 10 \
	--num_selected_query 5 \
	--embed_model_id 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' \
	--sample_rerank \
	--filtered_by_query


	#--test_file "/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv" \
	#--load_results \
	#--experiment_dir experiments/summarization_llama3_ds-MTS_Dialogue_epochs-10_rank-8_dropout-0.1 \
	#--experiment_dir experiments/summarization_llama2_ds-aci-bench_epochs-10_rank-8_dropout-0.1 \
	#--experiment_dir experiments/summarization_with_preprocess_csv_data_epochs-10_rank-8_dropout-0.1-None \
	#--experiment_dir experiments/summarization_llama2__ds-MTS_Dialogue_epochs-20_rank-8_dropout-0.1_msl2048 \
	#--experiment_dir experiments/summarization_llama2_ds-MTS_Dialogue_epochs-10_rank-8_dropout-0.1 \

	#--dataset aci-bench
        #--test_file /home/data2/yongfeng/code3/aci-bench/data/challenge_data/clinicalnlp_taskB_test1_metadata.csv"


