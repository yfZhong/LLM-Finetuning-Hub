import argparse
import torch
import os
import pandas as pd
import evaluate
from datasets import load_dataset
import pickle
import json
import warnings

from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM

from prompts import INFERENCE_SUMMARIZATION_PROMPT, INFERENCE_SUMMARIZATION_PROMPT_v2, INFERENCE_SUMMARIZATION_PROMPT_v3
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import openai
import requests
import statistics

from sent_similarity import Sent_Similar
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_retrieval import rank_based_docs


# Set your API key
openai.api_key ="Your open apt key"

qg_model = None
qg_tokenizer = None
pred_model = None
pred_tokenizer = None

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def prepare_instructions(dialogues, section_headers, section_texts):
    instructions = []
    summaries = []

    prompt = INFERENCE_SUMMARIZATION_PROMPT_v2

    for dialogue, section_header, section_text in zip(dialogues, section_headers, section_texts):
        example = prompt.format(
            dialogue=dialogue,
            # summary='Section header:{}\nSection text:{}'.format(section_header, section_text),
        )
        instructions.append(example)
        summaries.append('Section header:{}\nSection text:{}'.format(section_header, section_text))

    return instructions, summaries

def prepare_mts_instructions(dialogues, section_texts):
    instructions = []
    summaries = []
    # prompt = INFERENCE_SUMMARIZATION_PROMPT
    prompt = INFERENCE_SUMMARIZATION_PROMPT_v2
    for dialogue, section_text in zip(dialogues, section_texts):
        example = prompt.format(
            dialogue=dialogue,
            # summary=section_text
        )
        instructions.append(example)
        summaries.append( section_text)
    return instructions, summaries

# def prepare_mts_instructions(dialogues, section_headers, section_texts):
#     instructions = []
#     summaries = []
#     prompt = INFERENCE_SUMMARIZATION_PROMPT_v2
#
#     for dialogue, section_header, section_text in zip(dialogues, section_headers, section_texts):
#         example = prompt.format(
#             dialogue=dialogue,
#         )
#         instructions.append(example)
#         summaries.append('Section header:{}\nSection text:{}'.format(section_header, section_text))
#
#     return instructions, summaries

def prepare_mtsdialog_data(args):
    # dataset = load_dataset("csv", data_files={"train": args.train_file, "test": args.test_file})
    dataset = load_dataset("csv", data_files={"test": args.test_file})

    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    # section_header = val_dataset["section_header"]
    section_text = val_dataset["section_text"]
    # summaries = train_dataset["summary"]

    validation_instructions, summaries = prepare_mts_instructions(dialogues, section_text)
    return validation_instructions, summaries


def prepare_aci_instructions(dialogues, section_texts):
    instructions = []
    summaries = []

    prompt = INFERENCE_SUMMARIZATION_PROMPT_v3
    for dialogue, section_text in zip(dialogues, section_texts):
        example = prompt.format(
            dialogue=dialogue,
            # summary=section_text
        )
        instructions.append(example)
        summaries.append(section_text)

    return instructions, summaries

def prepare_acibench_data(args):
    # dataset = load_dataset("csv", data_files={"train": args.train_file, "test": args.test_file})
    dataset = load_dataset("csv", data_files={"test": args.test_file})

    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    notes = val_dataset["note"]

    validation_instructions, summaries = prepare_aci_instructions(dialogues, notes)
    return validation_instructions, summaries


def generate_step_with_gpt_model(messages, max_new_tokens=1024, temperature=1.0):
    cycle_i = 1
    MAXCYCLE = 50
    while cycle_i < MAXCYCLE:
        # print(f'cycle_i: {cycle_i}')
        try:

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                # 'https://api.gptapi.us/v1/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {openai.api_key}'
                    # 'Authorization': f'Bearer {my_api_key}'
                },
                json={
                    # 'model': 'gpt-3.5-turbo',
                    # 'model': 'gpt-3.5-turbo-0125',
                    'model': 'gpt-4',
                    'messages': messages,
                    'logprobs': True,
                    'temperature': temperature,
                    # 'n':5,
                }
            )
            # response_gpt4 = requests.post('https://api.gptapi.us/v1/chat/completions',headers={'Content-Type': 'application/json','Authorization': f'Bearer {openai.api_key}'},json={'model': 'gpt-4', 'messages': messages, 'n':5,})
            # print(response.json())

            # time.sleep(5)
            # completion = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=messages,
            #     # max_tokens=max_new_tokens,
            #     temperature=temperature,
            #     # n=5,
            # )
            # response_tmp = completion.choices[0].message.content

            cycle_i = MAXCYCLE
            response = response.json()['choices'][0]['message']['content']

        except Exception as error:
            print("An exception occurred:", error)
            cycle_i += 1
            response = ""
    return response

def question_generate_step_with_llama3(messages, max_new_tokens=1024, temperature=0.01):

    input_ids = qg_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(qg_model.device)

    terminators = [
        qg_tokenizer.eos_token_id,
        qg_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = qg_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=temperature,
        pad_token_id=qg_tokenizer.eos_token_id,
        repetition_penalty=1.0,
    )

    response = qg_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response

def pred_generate_step_with_llama2(messages, max_new_tokens=1024, temperature=0.01, repetition_penalty=1.0):

    input_ids = pred_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(pred_model.device)

    # input_ids = pred_tokenizer(
    #     messages, return_tensors="pt", truncation=True
    # ).input_ids.to(pred_model.device)

    terminators = [
        pred_tokenizer.eos_token_id,
        pred_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pred_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=temperature,
        pad_token_id=pred_tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        # repetition_penalty=1.0,
    )

    response = pred_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response


def main(args):
    global pred_model, pred_tokenizer, qg_model, qg_tokenizer
    if args.dataset == "MTS_Dialogue":
        val_instructions, summaries = prepare_mtsdialog_data(args)
    elif args.dataset == "aci-bench":
        val_instructions, summaries = prepare_acibench_data(args)

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"

    save_dir = os.path.join(experiment, "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.filtered_by_query:
        fbq = 1
    else:
        fbq = 0
    if args.sample_rerank:
        sr = 1
    else:
        sr = 0
    save_file_name = "testchat_test{}_temp{}_sr{}_candidates{}_gen_query{}_filter_query{}_fbq{}_rp{}_mid-{}_mnt-{}.txt".format(
        args.test_sample_num, args.temperature, sr, args.num_candidates, args.num_generated_query,
        args.num_selected_query, fbq, args.repetition_penalty, args.embed_model_id[:5], args.max_new_tokens)

    results = []
    if (args.load_results):
        json_file = os.path.join(save_dir, save_file_name[:-4] + "_result.json")
        if os.path.exists(json_file):
            with open(json_file) as jf:
                result_json = json.load(jf)
            for item in result_json:
                results.append(item['predict'])
        metrics = calculate_metrics(summaries[:args.test_sample_num], results)
        print(metrics)

        gpt_evaluation = False
        if gpt_evaluation:
            print("Start Gpt evaluation")
            R, T, A , Rs = GPT_Evaluation(val_instructions[:args.test_sample_num], results[:args.test_sample_num])
            # R, T, A, Rs= GPT_Evaluation(val_instructions[:2], results[:2])
            print("readability_score: {}\ntruthfulness_score: {}\nOverall Score: {}\n".format(statistics.mean(R), statistics.mean(T), statistics.mean(A)))
            save_info = "readability_score: {}\ntruthfulness_score: {}\nOverall Score: {}\n".format(statistics.mean(R), statistics.mean(T), statistics.mean(A))
            for i in range(len(Rs)):
                save_info += "{}\t{}\t{}\n".format(R[i], T[i], A[i])
            for i in range(len(Rs)):
                save_info +="{}-------------\n{}\n\n".format(i, Rs[i])
            with open(os.path.join(save_dir, save_file_name.replace(".txt", "-GPT-eval.txt")), "wb") as f:
                f.write(save_info.encode())

        # GPT_fact_checking(val_instructions[:args.test_sample_num], results)
        exit()


    # unpatch flash attention
    unplace_flash_attn_with_attn()

    # load base LLM model and tokenizer
    pred_model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # max_seq_length=512
    # tokenizer = AutoTokenizer.from_pretrained(peft_model_id, model_max_length=max_seq_length)
    pred_tokenizer = AutoTokenizer.from_pretrained(peft_model_id, device_map="auto",)

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed_model_id,
        # model_kwargs={
        #   "device": args.device,
        # },
        encode_kwargs={

        },
        # cache_folder=args.cache_dir
    )

    entailment_scorer = Sent_Similar(embeddings)

    qg_model_id = args.qg_model_id
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_id)
    qg_model = AutoModelForCausalLM.from_pretrained(qg_model_id, torch_dtype=torch.bfloat16, device_map="auto", )


    save_info = "Start:\n"
    with open(os.path.join(save_dir, save_file_name), "wb") as f:
        f.write(save_info.encode())
        save_info = ""

    i = 0
    for instruct, summary in zip(val_instructions[:args.test_sample_num], summaries[:args.test_sample_num]):
        if len(results)>=args.test_sample_num:
            break
        save_info += "{}------------------------------------------------------------\n".format(i)
        save_info += "Input: {}\n".format(instruct)
        save_info += "Summary: {}\n".format(summary)

        input_messages = [{"role": "user",
                     "content": '{}'.format(instruct)}]
        if args.dataset == "aci-bench":
            input_messages.append({"role": "system",
                             "content": """You are a helpful assistant who summarizes the given input dialogue into a 
                             medical report having the following sections : "HISTORY OF PRESENT ILLNESS", 
                             "PHYSICAL EXAM", "RESULTS", "ASSESSMENT AND PLAN". """})

        # input_ids = pred_tokenizer(
        #     instruct, return_tensors="pt", truncation=True
        # ).input_ids.cuda()
        with torch.inference_mode():
            num_candidates = args.num_candidates

            if args.sample_rerank and num_candidates > 1:
                sample_results = []
                for j in range(num_candidates):

                    result = pred_generate_step_with_llama2(input_messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, repetition_penalty=args.repetition_penalty)

                    if result not in sample_results:
                        sample_results.append(result)

                save_info += "Sample results:********\n {}\n".format("\n*".join(sample_results))

                if len(sample_results) > 1:
                    # QAG
                    # dialogue = instruct.split('```')[1]
                    prompt = "Generate {} question-answer pairs based on the following dialogue. " \
                             "The answers should come from the dialogue or can be infer from the dialogue. " \
                             "Follow the structure of the following question-answer pair: #Question_ID. Question: Where is the infection located in the patient's body?\nAnswer: The infection is in both lungs of the patient's body. " \
                             "The dialogue is as follows: {}".format(args.num_generated_query, instruct[:-13])

                    messages = [{"role": "user",
                                 "content": '{}'.format(prompt)}]

                    response = question_generate_step_with_llama3(messages)

                    save_info += "Generated questions:********\n {}\n".format(response)

                    # question_list = response.split('\n')[:args.num_query]
                    qa_list = response.replace('\n   \n', '\n\n').split('\n\n')[1:]
                    question_list = []
                    answer_list = []
                    for qa in qa_list:
                        if len(qa.split('\n')) == 2:
                            q, a = qa.split('\n')
                            question_list.append(q)
                            answer_list.append(a)

                    if len(question_list) <= args.num_selected_query:
                        query_answer_scores = [[question_list[k], answer_list[k], 0.5] for k in range(len(question_list))]
                    elif (args.filtered_by_query):
                        query_answer_scores = rank_based_docs(embeddings, question_list, question_list,
                                                              args.num_selected_query)
                    else:
                        query_answer_scores = rank_based_docs(embeddings, question_list, answer_list,
                                                              args.num_selected_query)

                    save_info += "Filtered questions:********\n {}\n".format(
                        "\n*".join([v[0] for v in query_answer_scores]))

                    # QA
                    scores = []
                    for qas in query_answer_scores:
                        q, _, _ = qas

                        # dialogue = instruct.split('```')[1]
                        messages0 = [{"role": "user",
                                      "content": 'Refer to the Knowledge: {} and answer the question {}， don\'t make up any knowledge out of the given knowledge'.format(
                                          instruct[:-13], q)}]
                        answer_0 = question_generate_step_with_llama3(messages0)
                        answer_list = []
                        for sample_sum in sample_results:
                            messages_i = [{"role": "user",
                                           "content": 'Refer to the Knowledge: {} and answer the question {}， don\'t make up any knowledge out of the given knowledge'.format(
                                               sample_sum, q)}]

                            a_i = question_generate_step_with_llama3(messages_i)

                            answer_list.append(a_i)

                        current_scores, _ = entailment_scorer.get_scores(answer_0, answer_list)
                        if len(scores) == 0:
                            scores = current_scores
                        else:
                            scores = list(map(lambda x: x[0] + x[1], zip(scores, current_scores)))
                    if len(scores) != 0:
                        max_index = scores.index(max(scores))
                    else:
                        print("No scores!")
                        print(query_answer_scores)
                        max_index = 0

                    result = sample_results[max_index]
                    save_info += "Selected result:*****************\n {}: {}\n".format(max_index, result)
                else:
                    result = sample_results[0]
                    save_info += "Only one result:*****************\n{}\n".format(result)
            else:
                result = pred_generate_step_with_llama2(input_messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, repetition_penalty=args.repetition_penalty)

                save_info += "Result:*********************\n{}\n".format(result)

                with open(os.path.join(save_dir, save_file_name), "a") as f:
                    f.write(save_info)
                    save_info = ""

            results.append(result)
            i += 1
            print("{}/{}".format(str(i), str(args.test_sample_num)))

    # compute metric
    # rouge = metric.compute(predictions=results, references=summaries, use_stemmer=True)
    # metrics = {metric: round(rouge[metric] * 100, 2) for metric in rouge.keys()}
    # print(metrics)

    result_json = []
    for k in range(args.test_sample_num):
        result_json.append({"input": val_instructions[k], "summary":summaries[k], "predict": results[k]})
    with open(os.path.join(save_dir, save_file_name[:-4]+"_result.json"), "w") as fp:
        json.dump(result_json, fp)


    metrics = calculate_metrics(summaries[:args.test_sample_num], results)
    print(metrics)

    # GPT_fact_checking(val_instructions[:args.test_sample_num], results)

    metrics_names = "\t".join(['{}'.format(k) for k, v in metrics.items()])
    metrics_values = "\t".join(['{}'.format(v) for k, v in metrics.items()])
    save_info += "Metrics**********\n{}\n{}".format(metrics_names, metrics_values)

    with open(os.path.join(save_dir, save_file_name), "a") as f:
        f.write(save_info)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")

def calculate_metrics(references, predictions):
    ######## Load Metrics from HuggingFace ########
    print("Loading ROUGE, BERTScore, BLEURT from HuggingFace")
    scorers = {
        "rouge": (
            {"path": "rouge"},
            {"use_aggregator": False},
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        ),
        "bert_scorer": (
            {"path": "bertscore"},
            {"model_type": "microsoft/deberta-xlarge-mnli", "batch_size": 1},
            ["precision", "recall", "f1"],
            ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
        ),
        "bleurt": ({"path": "bleurt", "config_name": "BLEURT-20"}, {}, ["scores"], ["bleurt"]),
    }

    ######## CALCULATE PER INSTANCE SCORES ########
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        # NOTE: We have re-written this to only load one model into memory at a time
        scores = evaluate.load(**scorer).compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]
    metrics = {metric: round(np.mean(all_scores[metric]), 4) for metric in all_scores.keys()}

    return metrics

def GPT_Evaluation(input_dialogues, predictions):
    R=[]
    T=[]
    A=[]
    responses = []
    for i in range(len(input_dialogues)):
        input_dialogue = input_dialogues[i]
        prediction = predictions[i]
        prompt = "Given an input dialogue and its summarisation, please evaluate the summarization based on the criteria below:\n" \
                 "1. Readability: Assess how easy it is to read and understand each summarization. Consider factors such as clarity, coherence, and organization. Provide a score between 1 and 10, where 1 indicates very poor readability and 10 indicates excellent readability.\n" \
                 "2. Truthfulness: Evaluate how accurately each summarization reflects the information provided in the original dialogue. Provide a score between 1 and 10, where 1 indicates very poor alignment with the original dialogue and 10 indicates perfect alignment.\n" \
                 "3. Overall Score: Based on your assessment of readability and truthfulness, provide an overall score for each summarization. This score should also be between 1 and 10, reflecting your overall evaluation.\n" \
                 "Evaluation output:\n" \
                 "Summarization:\n" \
                 "* Readability: [Insert score between 1 and 10]\n" \
                 "* Truthfulness: [Insert score between 1 and 10]\n" \
                 "* Overall Score: [Insert score between 1 and 10]\n" \
                 "Please provide detailed comments explaining the rationale behind each of your scores.\n\n" \
                 "{}" \
                 "{}".format(input_dialogue.replace("Summarize the following dialogue that is delimited with triple backticks.\n\n", ""), prediction)
        messages = [{"role": "user",
                     "content": '{}'.format(prompt)}]
        response = generate_step_with_gpt_model(messages, max_new_tokens=256, temperature=0.01)
        try:
            readability_score = float(response.split("Readability: ")[1].split('\n')[0])
            truthfulness_score = float(response.split("Truthfulness: ")[1].split('\n')[0])
            overall_score = float(response.split("Overall Score: ")[1].split('\n')[0])
        except:
            readability_score = 0
            truthfulness_score = 0
            overall_score = 0
            print("No proper prediction---------------------------")
            print(response)

        R.append(readability_score)
        T.append(truthfulness_score)
        A.append(overall_score)
        responses.append("-------------{}-----------\n{}\n{}\n{}".format(i, input_dialogue , prediction, response))
    return R, T, A, responses
    # return statistics.mean(R), statistics.mean(T), statistics.mean(A), responses

def GPT_fact_checking(input_dialogues, gt_summaries, predictions):

    gt_facts_num = 0
    pred_facts_num = 0
    co_pred_facts_num = 0

    for dialogue, summary, prediction in zip(input_dialogues, gt_summaries, predictions):

        prompt = "You are a fact-extraction expert. Given an input dialogue and its groundtruth summarisation, please extract fact items from them:\n" \
                  "The output should follows the structure:\n" \
                 "Facts:\n#ID: fact.\n#ID: fact.\n...\n" \
                 "{}" \
                 "{}".format(dialogue, summary)
        messages = [{"role": "user",
                     "content": '{}'.format(prompt)}]
        gt_facts = generate_step_with_gpt_model(messages, max_new_tokens=516, temperature=1.0)

        prompt = "You are a fact-extraction expert. Given a summarisation generated by LLM, please extract fact items from it:\n" \
                 "The output should follows the structure:\n" \
                 "Facts:\n#ID: fact.\n#ID: fact.\n...\n" \
                 "\nHere is the predicted summary:\n {}".format(prediction)
        messages = [{"role": "user",
                     "content": '{}'.format(prompt)}]
        pred_facts = generate_step_with_gpt_model(messages, max_new_tokens=516, temperature=1.0).split('\n')[1:]

        gt_facts_num += len(gt_facts.split('\n')[1:])

        for pred_fact in pred_facts:
            pred_facts_num += 1
            prompt = "You are a fact-extraction expert. Evaluate the truthfulness of a fact item according to the given facts:\n" \
                  "The output should follows the structure:\n" \
                  "Result: Insert True or False\n" \
                    "Fact item: {}\n" \
                    "Given facts: {}".format(pred_fact, gt_facts)

            messages = [{"role": "user",
                         "content": '{}'.format(prompt)}]
            result = generate_step_with_gpt_model(messages, max_new_tokens=516, temperature=1.0)

            try:
                if result.split("Result:")[1].strip() =="True":
                    co_pred_facts_num += 1
            except:
                print("Not good response.")


    precision = round(co_pred_facts_num/pred_facts_num, 4)
    recall = round(co_pred_facts_num/gt_facts_num)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("precision: {}\nrecall: {}\nf1: {}".format(precision, recall, f1))

    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/summarization_epochs-1_rank-64_dropout-0.1",
    )
    parser.add_argument("--train_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv")
    parser.add_argument("--validation_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv")
    parser.add_argument("--test_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv")

    parser.add_argument("--dataset", type=str,
                        default="MTS_Dialogue") #aci-bench
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument('--sample_rerank', action='store_true')
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--num_generated_query", type=int, default=10)
    parser.add_argument("--num_selected_query", type=int, default=5)
    parser.add_argument("--filtered_by_query", action='store_true')
    parser.add_argument("--load_results", action='store_true')

    parser.add_argument("--test_sample_num", type=int, default=20)

    parser.add_argument("--qg_model_id", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embed_model_id", type=str,
                        default="mixedbread-ai/mxbai-embed-large-v1")  # intfloat/e5-mistral-7b-instruct   sentence-transformers/multi-qa-MiniLM-L6-cos-v1

    args = parser.parse_args()
    main(args)
