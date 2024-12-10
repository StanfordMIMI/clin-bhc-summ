import evaluate 
from f1chexbert import F1CheXbert
import json
import numpy as np
import os
from radgraph import F1RadGraph

import constants
import parser
import process
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

def main():

    # parse arguments, set data paths
    args = parser.get_parser()

    true, pred = process.load_summaries(args)
    
    # load hugging face metrics
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    bleu_results = []
    rouge_results = []

    for i in range(len(pred)):
        out, tgt = pred[i], true[i][0]
        
        # compute hugging face metrics
        bleu_results.append(round(sentence_bleu([tgt], out), 4))
        rouge_results.append(round(rouge.compute(predictions=[out], references=[tgt])['rougeL'], 4))
    
    bleu_results = np.array(bleu_results) * 100
    rouge_results = np.array(rouge_results) * 100
    bert_results = np.array([round(i, 4) for i in bertscore.compute(predictions=pred, references=true, lang='en')['f1']]) * 100

    df = pd.DataFrame(bleu_results, columns=["bleu"])
    df['rougeL'] = rouge_results
    df['bert'] = bert_results
    df.to_csv(os.path.join(args.dir_out, "metrics.csv"),)

    print(f'successfully computed metrics, saved in {args.dir_out}')

if __name__ == '__main__':
    main()
