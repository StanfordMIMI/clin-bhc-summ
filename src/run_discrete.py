import os
import sys
import time
from tqdm import tqdm
import transformers
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from process import load_model
import constants
import process
import parser

def main():
    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()
    
    if args.model[0:6] == "falcon":
        if args.case_id >= 200:
            model, tokenizer = load_model(args)
            generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto",
                                              trust_remote_code=True)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(constants.MODELS[args.model], trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            generator = transformers.pipeline("text-generation", model=constants.MODELS[args.model], tokenizer=tokenizer,
                                              device_map="auto", trust_remote_code=True)
    elif args.model[0:6] == "llama2":
        if args.case_id >= 200:
            model, tokenizer = load_model(args)
            generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(constants.MODELS[args.model])
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            generator = transformers.pipeline("text-generation", model=constants.MODELS[args.model], tokenizer=tokenizer,
                                              device_map="auto")
    else:
        if args.case_id >= 200:
            model, tokenizer = load_model(args)
            generator = transformers.pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(constants.MODELS[args.model])
            generator = transformers.pipeline("text2text-generation", model=constants.MODELS[args.model], tokenizer=tokenizer,
                                              device_map="auto")
    
    # preprocess data. list order will be different in dataloader
    list_finding_ = process.load_data(args, task='test')
    
    if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
        def template_dataset(sample):
            sample["sentence"] = f"{process.causal_formatting_test(sample, constants.START_PREFIX)}"
            return sample
        
        def template_dataset_null(sample):
            context = f"### Clinical Note\n{sample['sentence']}" if len(sample["sentence"]) > 0 else None
            response = f"### Brief Hospital Course\n"

            # join all the parts together
            sample['sentence'] = "\n\n".join([i for i in [context, response] if i is not None])
            return sample

        if args.case_id < 5:
            if args.case_id  == 0:
                list_finding_ = list_finding_.map(template_dataset_null)
        else:
            list_finding_ = list_finding_.map(template_dataset)

    # reference findings + summaries and generated summaries
    list_finding, list_sum_ref, list_sum_gen, idcs = [], [], [], []

    # generate summary for each finding
    t0 = time.time()
    for finding in tqdm(list_finding_, total=len(list_finding_)):
        if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
            generated = generator(finding['sentence'], max_new_tokens=constants.MAX_NEW_TOKENS, clean_up_tokenization_spaces=True)
            list_sum_gen.append(generated[0]['generated_text'][len(finding['sentence']):])
        else:
            generated = generator(finding['sentence'], max_length=constants.MAX_NEW_TOKENS, clean_up_tokenization_spaces=True)
            list_sum_gen.append(generated[0]['generated_text'])
            
        list_sum_ref.append(finding['text_label'])
        list_finding.append(finding['sentence'])
        idcs.append(finding['idx']) # track idcs to preserve order
        print('input: \n' + finding['sentence'])
        print('\nlabel: \n' + finding['text_label'])
        if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
            print('\noutput: \n' + generated[0]['generated_text'][len(finding['sentence']):])
        else:
            print('\noutput: \n' + generated[0]['generated_text'])

    print('generated {} samples for {} expmt in {} sec'.format(\
          len(list_finding_), args.expmt_name, time.time() - t0))

    process.postprocess_and_save(args, idcs, list_finding, list_sum_ref, list_sum_gen)

if __name__ == "__main__":
    main()