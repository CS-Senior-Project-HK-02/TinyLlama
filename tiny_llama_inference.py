import sys
import transformers
from transformers import AutoTokenizer
import torch
import pandas as pd
sys.path.append(r"/home/hwalke37/")
from HKLLM.promptlm.utils.data import prepare_dataset_for_inference, generate_shot_examples
from HKLLM.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.promptlm.utils.parsers import multc_parser, parse_output_for_answer

model = "PY007/TinyLlama-1.1B-Chat-v0.1"

data = r"/home/hwalke37/dataset/300_gt_deided_case.csv"
df = pd.read_csv(data)


dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297)
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

system_prompt = """[INST]You are the police department's virtual assistant, you are going to read the following narratives
        and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample
        is based on the current report, references to past events inside of the report, do not affect the classification of the report.
        For your response you must always use <Tag> [Answer] </Tag>.
        you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other
        , The text you must classify is as follows: [/INST]"""
        
texts = dataset["x"]
labels = dataset["y"]


full_prompts = [system_prompt+text+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning, assume Other if none of the above.[/INST]" for text in texts ]
for prompt in full_prompts:
    answer_array = []
    extracted_answers = []
    
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=50,
        top_p = 0.9,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    model_output = sequences[0]['generated_text']
    print(sequences[0]['generated_text'])
    keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other"]
    extracted_answer = (parse_output_for_answer(model_output,keywords=keywords,single_output=True))
    print(extracted_answer)
    if extracted_answer == None or extracted_answers == []:
        extracted_answer = "Non_Answer"
    
    processed_answer = extracted_answer[0].replace(" ","_")
    extracted_answers.append(processed_answer)
accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"f1: {f1_score:.2f}")
