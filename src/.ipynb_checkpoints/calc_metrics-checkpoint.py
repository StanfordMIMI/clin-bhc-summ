import evaluate 
from f1chexbert import F1CheXbert
import json
import numpy as np
import os
from radgraph import F1RadGraph

import constants
import parser
import process


def main():

    # parse arguments, set data paths
    args = parser.get_parser()

    true, pred = process.load_summaries(args)
    reader_idx = [4,11,14,19,20,28,29,35,36,40,43,44,46,49,58,64,65,68,70,71,74,78,81,84,85,87,89,92,95,100]
    idx = [i for i in range(1, 101)]
    idx = set(idx) - set(reader_idx)
    
    true = [true[i-1] for i in idx]
    pred = [pred[i-1] for i in idx]

    # load hugging face metrics
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    # compute hugging face metrics
    bleu_results = bleu.compute(predictions=pred, references=true)
    rouge_results = rouge.compute(predictions=pred, references=true)
    bert_results = bertscore.compute(predictions=pred, references=true, lang='en')

    # compute f1-chexbert, f1-radgraph
    true = [item for sublist in true for item in sublist] # flatten list
    if constants.EVAL_CXR_DATA:
        f1chexbert = F1CheXbert(device="cuda")
        accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(hyps=pred, refs=true)
        chexbert_score = class_report_5['micro avg']['f1-score']
    else:
        chexbert_score = 0.
        
    f1radgraph = F1RadGraph(reward_level="partial")
    radgraph_score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=pred, refs=true)

    metrics = {
        'BLEU': bleu_results['bleu'],
        'ROUGE-1': rouge_results['rouge1'],
        'ROUGE-2': rouge_results['rouge2'],
        'ROUGE-L': rouge_results['rougeL'],
        'BERT': np.mean(bert_results['f1']), # options: 'f1', 'recall', 'precision
        'F1-CheXbert': chexbert_score,
        'F1-Radgraph': radgraph_score,
    }

    # scale metrics to be on [0,100] instead of [0,1]
    for key in metrics:
        metrics[key] *= 100.

    with open(os.path.join(args.dir_out, constants.FN_METRICS), 'w') as f:
        f.write(json.dumps(metrics))
    print(f'successfully computed metrics, saved in {args.dir_out}')


if __name__ == '__main__':
    main()
