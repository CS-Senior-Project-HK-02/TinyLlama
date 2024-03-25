import sys
import os

sys.path.append(r"/home/hwalke37")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
from HKLLM.hkllm.promptlm.utils.data import prepare_dataset_for_inference, generate_shot_examples
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

model_name = r"/home/hwalke37/workspace/tiny-llama-pn-annotate-hkllm-ksu-internal"

data = r"/home/hwalke37/dataset/300_gt_deided_case.csv"
df = pd.read_csv(data)

dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297, supp_columns = ["Supp", "OfficerNarrative"])
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)#.to(device)
"""model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )"""

pipe = pipeline(
    "text-generation", 
    model=model_name, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)


system_prompt = "[INST]You are the police department's virtual assistant, you are going to read the following narratives and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample is based on the current report, references to past events inside of the report, do not affect the classification of the report. For your response you must always use <Tag> [Answer] </Tag>. you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other, The text you must classify is as follows: [/INST]"
        
texts = dataset["x"]
labels = dataset["y"]


full_prompts = [system_prompt+text+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning, assume Other if none of the above.[/INST]" for text in texts ]


    
for prompt in full_prompts:
    answer_array = []
    extracted_answers = []
    
    sequences = pipe(
        prompt,
        do_sample = True,
        top_k=10,
        top_p=0.95,
        temperature = 0.7,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=500,
        num_return_sequences=1,
    )
    model_output = sequences[0]['generated_text']
    print(model_output)
    keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other"]
    extracted_answer = (parse_output_for_answer(model_output,keywords=keywords,single_output=True))
    print(extracted_answer)
    if extracted_answer == None or extracted_answers == []:
        extracted_answer = "Non_Answer"
    
    processed_answer = extracted_answer[0].replace(" ","_")
    extracted_answers.append(processed_answer)
    
"""accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"f1: {f1_score:.2f}")"""
