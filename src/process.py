''' misc data processing functions '''

import csv
from datasets import Dataset
import itertools
import os
import numpy as np
import random
import re
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, default_data_collator, AutoTokenizer
import constants
import parser
from peft import PeftModel, PeftConfig

def load_data(args, task):
    ''' load data from .tok files into lists
        inputs:
            args:       as defined in parser.py
            task:       which task to perform i.e. train, validation, test
                        choices: ['trn', 'val', 'test'] '''

    # set data path
    if task not in ['trn', 'val', 'test']:
        raise ValueError('must specify task to be either [trn, val, test]')
    task_dict = {'trn': 'train', 'val': 'validate', 'test': 'test'}
    fn_task = task_dict[task]
    fn_finding = os.path.join(constants.DIR_DATA, f'{fn_task}.findings.tok')
    fn_summary = os.path.join(constants.DIR_DATA, f'{fn_task}.impression.tok')

    # set prefix, prompt
    insert_prefix = constants.cases[args.case_id]['insert_prefix']
    prompt = constants.cases[args.case_id]['prompt']

    list_finding = []
    list_summary = []

    with open (fn_finding, 'r') as f_f, open(fn_summary, 'r') as f_s:
        
        lines_finding = f_f.readlines()
        lines_summary = f_s.readlines()
        assert len(lines_finding) == len(lines_summary)

        count_long_seqs = 0
        for idx, finding in enumerate(lines_finding):
            
            summary = lines_summary[idx]

            if args.model[0:6] == "llama2" or args.model[0:6] == "falcon":
                elem_finding = ' '.join(f'{finding}'.split())
            else:
                elem_finding = ' '.join(f'{insert_prefix}\n{finding}\n{prompt}'.lower().split())
            
            elem_summary = summary.rstrip()

            list_finding.append(elem_finding)
            list_summary.append(elem_summary)
           
    # generate in-context prompts
    if constants.ICL_PROMPT in prompt:
        list_finding, list_summary = get_in_context_prompt(list_finding, list_summary, task, args, )

    # prefixed prompt: add prefix to start of each finding prompt (i.e. before finding)
    if 'start_prefix' in constants.cases[args.case_id]:
        start_prefix = constants.cases[args.case_id]['start_prefix']
        list_finding = [f'{start_prefix}\n{elem}' for elem in list_finding]

    data_list = [
        {
            'idx': i, 
            "sentence": list_finding[i], 
            "text_label": list_summary[i]
        } for i in range(len(list_finding))
    ]
    
    dataset = Dataset.from_list(data_list)

    return dataset

def causal_formatting_func(sample, instruction):
    instruction = f"### Instruction\n{instruction}"
    context = f"### Context\n{sample['sentence']}" if len(sample["sentence"]) > 0 else None
    response = f"### Answer\n{sample['text_label']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def causal_formatting_test(sample, instruction):
    instruction = f"### Instruction\n{instruction}"
    context = f"### Context\n{sample['sentence']}" if len(sample["sentence"]) > 0 else None
    response = f"### Answer\n"

    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def get_in_context_prompt(list_finding, list_summary, task, args, knn=True):
    ''' create in-context learning prompts '''

    tmp_list_finding, tmp_list_summary = [], []
    trn_findings, trn_summaries = [], []
    
    insert_prefix = constants.cases[args.case_id]['insert_prefix']
    prompt = constants.cases[args.case_id]['prompt']

    # get icl directory
    trn_findings_path = os.path.join(constants.DIR_DATA, 'train.findings.tok')
    trn_summaries_path = os.path.join(constants.DIR_DATA, 'train.impression.tok')
    
    with open(trn_findings_path, 'r') as f:
        trn_findings = f.readlines()
    with open(trn_summaries_path, 'r') as f:
        trn_summaries = f.readlines()

    prompt_str = re.findall(f'{constants.ICL_PROMPT.lower()}_.*', list_finding[0])[0]
    
    if args.model[0:6] == "llama2" or args.model[0:6] == "falcon":
        k = 1
    else:
        k = int(prompt_str.split('_')[-1].split(' ')[0]) # number of in-context examples
    

    for i, element in tqdm(enumerate(list_finding), total=len(list_finding)):
        #example_indices = random.choices(trn_findings, k=k)
        example_idcs = random.sample(range(len(trn_findings)), k=k)
        
        if args.model[0:6] == "llama2" or args.model[0:6] == "falcon":
            new_element = '\n'.join([causal_formatting_func({'sentence': trn_findings[j], 'text_label': trn_summaries[j]}) for j in
                                     example_idcs])
            new_element += '\n' + causal_formatting_test({'sentence': list_finding[i]})
        else:
            new_element = '\n\n'.join([' '.join(f'{insert_prefix}\n{trn_findings[j]}\n{prompt}\n{trn_summaries[j]}'.lower().split())
                                       for j in example_idcs])
            new_element += f'\n{list_finding[i]}'

        # need to do lower() because we lowercased all text in element previously
        new_element = new_element.replace(f' {constants.ICL_PROMPT.lower()}_{k}', '')

        # filter out findings longer than maximum token length
        if len(new_element.strip().split(' ')) < constants.MAX_LEN:
            tmp_list_finding.append(new_element)
            tmp_list_summary.append(list_summary[i])

    list_finding = tmp_list_finding
    list_summary = tmp_list_summary

    return list_finding, list_summary


def write_list_to_csv(fn_csv, list_, csv_action='w'):
    ''' write each element of 1d list to csv 
        can also append to existing file w csv_action="a" '''

    with open(fn_csv, csv_action) as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(list_)
    return

def preprocess_function(examples, tokenizer, model):
    # tokenize examples['sentence'] (finding) and examples['text_label'] (summary)
    model_inputs = tokenizer(examples['sentence'], max_length=constants.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(examples['text_label'], max_length=constants.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")

    # extract labels['input_ids']
    labels = labels["input_ids"]
    if model[0:6] != "falcon" and model[0:6] != "llama2":
        labels[labels == tokenizer.pad_token_id] = -100 # replace padding token id's by -100 so it's ignored by the loss
    model_inputs["labels"] = labels

    return model_inputs

def get_loader(dataset, tokenizer, batch_size, model):
    ''' given list of input and target texts, return dataloader '''

    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, model),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=['sentence', 'text_label'],
        desc="Running tokenizer on dataset",
    )

    loader = DataLoader(processed_dataset,
                        collate_fn=default_data_collator,
                        batch_size=batch_size, 
                        shuffle=True, pin_memory=True,
                        drop_last=False)

    return loader

def clean_txt(list_):
    ''' given a list of generated summaries,
        remove junk '''

    # remove everything in brackets and underscores
    list_ = [re.sub("[\(\[].*?[\)\]]", "", ll) for ll in list_]
    # list_ = [ll.replace('_', '') for ll in list_]

    # remove all duplicate adjacent words
    list_ = [' '.join([k for k, g in itertools.groupby(ll.split())]) for ll in list_]

    return list_


def sort_list_per_indices(objects, indices):
    ''' given two lists, e.g.
            objects = ['a', 'c', 'b']
            indices = [0, 2, 1]
        return objects sorted corresponding to increasing indices
            sorted_objects = ['a', 'b', 'c'] 

        used to preserve order of inputs/outputs '''

    if len(objects) == 0:
        return objects

    # convert each list item to int
    indices = [idx.item() if torch.is_tensor(idx) else idx for idx in indices]

    # create a list of tuples, where each tuple contains the object and its corresponding index
    objects_with_indices = [(obj, idx) for obj, idx in zip(objects, indices)]

    # sort the list of tuples based on the indices in ascending order
    objects_with_indices.sort(key=lambda x: x[1])

    # extract the objects from the sorted list of tuples
    sorted_objects = [obj for obj, _ in objects_with_indices]

    return sorted_objects


def sort_lists_per_indices(list_outputs, indices):
    ''' wrapper to sort lists according to indices 
        where list_outputs is a list of individual lists to sort '''

    return [sort_list_per_indices(list_, indices) for list_ in list_outputs]


def save_output(args, list_outputs):
    ''' given lists: reference findings + summaries and generated summaries 
        save each to csv file '''

    output_fns = [constants.FN_FINDING, constants.FN_SUM_REF, constants.FN_SUM_GEN]

    for fn, out in zip(output_fns, list_outputs):

        if len(out) == 0: # i.e. if eval_hidden, don't have access to reference
            continue

        write_list_to_csv(os.path.join(args.dir_out, fn), out)

    return


def postprocess_and_save(args, idcs, list_finding, list_sum_ref, list_sum_gen):
    ''' wrapper function for postprocessing and saving output '''

    # clean text
    list_sum_gen = clean_txt(list_sum_gen)
    list_finding = [l.replace('\n', '') for l in list_finding]

    # group each output list into list of lists
    list_outputs = [list_finding, list_sum_ref, list_sum_gen]

    # sort each output list according to indices s.t. order is preserved
    list_outputs = sort_lists_per_indices(list_outputs, idcs)

    # save each output list to csv
    save_output(args, list_outputs)
    print(f'saved inputs, outputs, labels to {args.dir_out}')

    return 

def load_summaries(args):
    ''' given params for already-run experiment, load summaries into lists
        returns:
                true (list of str): reference summaries
                pred (list of str): generated summaries '''

    # load ground-truth, predictions
    csv_true = os.path.join(args.dir_out, constants.FN_SUM_REF)
    csv_pred = os.path.join(args.dir_out, constants.FN_SUM_GEN)
    with open(csv_pred, 'r') as f_p, open(csv_true, 'r') as f_t:
        pred = list(csv.reader(f_p, delimiter='\n'))
        true = list(csv.reader(f_t, delimiter='\n'))

    assert len(true) == len(pred)

    # kill empty strings generated in some outputs
    true = kill_null_str(true)
    pred = kill_null_str(pred)

    pred = [item for sublist in pred for item in sublist] # flatten list

    return true, pred


def kill_null_str(list_):
    ''' each element in list should be a length-1 list with a str
        sometimes elements are length-2 with a null string
        hence we want to get rid of this null string '''

    count = 0 # count number of null strings

    for idx, elem in enumerate(list_):
        if len(elem) != 1:

            # (rare) if empty output, append empty str
            if len(elem) == 0:
                elem = ['']

            count += 1
            elem_new = []
            elem_new.append(elem[0])
            list_[idx] = elem_new

    if count > 0:
        print(f'found {count} elements w null strings')

    return list_

def load_model(args):
    subdirs = [ii[0].split('/')[-1] for ii in os.walk(args.dir_models_tuned)]
    model_epoch = max([int(ii) for ii in subdirs if ii.isdigit()])

    dir_model_peft = os.path.join(args.dir_models_tuned, f'{model_epoch}')
    print(f'evaluating model: {dir_model_peft}')
    
    if args.model[0:6] == "falcon" or args.model[0:6] == "llama2":
        model = AutoModelForCausalLM.from_pretrained(dir_model_peft, return_dict=True, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(dir_model_peft)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(dir_model_peft, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(dir_model_peft)
            
    return model, tokenizer