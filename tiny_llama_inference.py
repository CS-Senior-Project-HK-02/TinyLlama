import sys
import os

sys.path.append(r"/home/hwalke37")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
from HKLLM.hkllm.promptlm.utils.data import prepare_dataset_for_inference, generate_shot_examples
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer

def import_class_definitions():
    class_def = "Behavioral Health: Can only be applied to people\nMental Health: Involving an individual with a diagnosed mental disorder, like schizophrenia or sucidal ideations\nDomestic Social: Involving multiple Individuals in a home setting, like husband/wife or parent/children domestic disputes\nNonDomestic Social: Involving multiple individuals not in a home setting, like comitting crimes on those not related to the perpretrator (This is rare by the way)\nSubstance Abuse: Individual with persistent drug/alchohol abuse problems, Posession alone is not enough to indicate abuse problems."

def inference_model(model_name, system_prompt = "", lora_adapter=None, use_class_definitions = False, use_knowledge_files = False, override = False):

    model_name = model_name

    data = r"/home/hwalke37/dataset/300_gt_deided_case.csv"
    df = pd.read_csv(data)
    class_def = import_class_definitions()
    dataset = prepare_dataset_for_inference(df=df,text_col="PublicNarrative",class_col="BHR_type",sample_size=297, supp_columns = ["Supp", "OfficerNarrative"])
    #device = "cpu"

    
    """quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                
                
            )"""
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_4bit=False,
            #quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    if lora_adapter:
            lora_adapter_name = lora_adapter
            model.load_adapter(lora_adapter_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        "text-generation", 
        model=model_name, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

            
    texts = dataset["x"]
    labels = dataset["y"]


    if use_class_definitions:
            full_prompts = [str(class_def)+str(system_prompt)+str(text)+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning[/INST]" for text in texts ]
    else:
            full_prompts = [str(system_prompt)+str(text)+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give intermediate reasoning[/INST]" for text in texts ]
    extracted_answers = []
    extracted_answers = []


        
    for i, prompt in enumerate(full_prompts):
        len(full_prompts)
        
        sequences = pipe(
            prompt,
            do_sample = True,
            top_k=150,
            top_p=0.9,
            temperature = 0.95,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
            num_return_sequences=1,
        )
        model_output = sequences[0]['generated_text']
        print(sequences[0]['generated_text'])
        keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other","SubstanceAbuse,DomesticSocial,NonDomesticSocial"]
        extracted_answer = (parse_output_for_answer(model_output,keywords=keywords,single_output=True))
        print(extracted_answer)
            
            
        if extracted_answer == None or extracted_answer == [] or extracted_answer == "":
            processed_answer = 'Other'
            if override == True:
                processed_answer = input("Answer was not correctly processed, Enter the correct Tag: ")
                processed_answer = processed_answer.replace(" ","_")
            extracted_answers.append(processed_answer)
        elif extracted_answer == 'SubstanceAbuse':
            processed_answer = 'Substance Abuse'
            extracted_answers.append(processed_answer)
        elif extracted_answer == 'DomesticSocial':
            processed_answer = 'Domestic Social'
            extracted_answers.append(processed_answer)
        elif extracted_answer == "NonDomesticSocial":
            processed_answer = "NonDomestic Social"
            extracted_answers.append(processed_answer)
        else:
            processed_answer = extracted_answer[0].replace(" ","_")
            extracted_answers.append(processed_answer)
        is_correct = "Correct" if processed_answer == labels[i] else "Incorrect"
        print(f"Sample {i}: Extracted Answer = {processed_answer}, Label = {labels[i]}, {is_correct}")
        print(len(extracted_answers),len(labels))
                    

system_prompt = "[INST]You are the police department's virtual assistant, you are going to read the following narratives and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample is based on the current report, references to past events inside of the report, do not affect the classification of the report. For your response you must always use <Tag> [Answer] </Tag>. you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other, The text you must classify is as follows: [/INST]"
inference_model(model_name=r"/home/hwalke37/workspace/tiny-llama-fine-tuned-model-8", system_prompt = system_prompt, use_class_definitions=True, override=True)    
"""accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"f1: {f1_score:.2f}")"""
