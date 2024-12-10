import os
from peft import PeftModel, PeftConfig
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import constants
import parser
import process

def main():
    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()

    # load model, tokenizer
    model = load_model(args) # note: case_id determines training disbn

    if args.model[0:6] == "falcon":
        tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    elif args.model[0:6] == "llama2":
        tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    else:
        tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model])
    
    # load data
    test_dataset = process.load_data(args, task='test')
    
    if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
        def template_dataset(sample):
            sample["sentence"] = f"{process.causal_formatting_test(sample, constants.START_PREFIX)}"
            return sample
        
        test_dataset = test_dataset.map(template_dataset)
    
    test_loader = process.get_loader(test_dataset, tokenizer, args.batch_size, args.model)

    # reference findings + summaries and generated summaries
    list_finding, list_sum_ref, list_sum_gen, idcs = [], [], [], []
    model.eval()

    # generate summary for each finding
    t0 = time.time()
    for step, batch in enumerate(tqdm(test_loader)):
        
        idcs.extend(batch['idx']) # idcs preserve order
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], do_sample=False,
                                     max_new_tokens=args.max_new_tokens)
       
        findings = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
            for i in range(args.batch_size):
                decoded_outputs[i] = decoded_outputs[i][len(findings[i]):]

        list_finding.extend(findings)
        list_sum_gen.extend(decoded_outputs)
        print('\ninput\n' + tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)[0])
        print('\noutput\n' + decoded_outputs[0])

        ref_labels = batch['labels']
        if args.model[0:6] != "falcon" and args.model[0:6] != "llama2":
            ref_labels[ref_labels == -100] = 0
        list_sum_ref.extend(tokenizer.batch_decode(ref_labels, skip_special_tokens=True))
        print('\nlabel\n' + tokenizer.batch_decode(ref_labels, skip_special_tokens=True)[0])

    print('generated {} samples for {} expmt in {} sec'.format(len(list_sum_gen), args.expmt_name, time.time() - t0))
    process.postprocess_and_save(args, idcs, list_finding, list_sum_ref, list_sum_gen)

def load_model(args):
    subdirs = [ii[0].split('/')[-1] for ii in os.walk(args.dir_models_tuned)]
    model_epoch = max([int(ii) for ii in subdirs if ii.isdigit()])

    dir_model_peft = os.path.join(args.dir_models_tuned, f'{model_epoch}')
    print(f'evaluating model: {dir_model_peft}')
    config = PeftConfig.from_pretrained(dir_model_peft)
    config.base_model_name_or_path = constants.MODELS[args.model] 
    
    if args.model[0:6] == "llama2":
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16,
                                                     device_map="auto")
        model = PeftModel.from_pretrained(model, dir_model_peft, device_map="auto")
    elif args.model[0:6] == "falcon":
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16,
                                                     device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, dir_model_peft, device_map="auto")
    elif args.model[0:6] == "flan-t":
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map="auto", offload_folder="offload")
        model = PeftModel.from_pretrained(model, dir_model_peft, device_map="auto", offload_folder="offload")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
        
        if args.model == "flan-ul2":
            model = PeftModel.from_pretrained(model, dir_model_peft, device_map="auto", load_in_8bit=True)
        else:
            model = PeftModel.from_pretrained(model, dir_model_peft, device_map="auto")

    return model

if __name__ == '__main__':
    main()