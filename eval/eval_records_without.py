import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import numpy as np
from bert_score import score
import json
from rouge_score import rouge_scorer
import spacy
import warnings
from metrics import records_soft_recall, records_entity_recall
warnings.filterwarnings("ignore")


def calculate_bertscore(candidate, reference):
    P, R, F1 = score(candidate, reference, lang="zh", model_type="bert-base-chinese")
    return P, R, F1

def load_str(path):
    with open(path, 'r', encoding='utf-8') as f:
        return '\n'.join(f.readlines())

def main(args):
    data = pd.read_excel(args.input_path)
    results = []
    reference_list = []
    candidate_list = []
    records_entity_recalls = []
    records_soft_recalls = []
    
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        records = row['按语']
        g_records_path = os.path.join(args.gen_records_path, 'polish_records_notes_without_' + args.type + '_' + str(idx) + '.txt')
        g_records = load_str(g_records_path)
        reference_list.append(records)
        candidate_list.append(g_records)

        recordsentity_recall = float(records_entity_recall(records, g_records))
        recordssoft_recall = float(records_soft_recall(records, g_records))
        records_entity_recalls.append(recordsentity_recall)
        records_soft_recalls.append(recordssoft_recall)

        # Calculate BERTScore for this pair
        P, R, F1 = calculate_bertscore([g_records], [records])
        P, R, F1 = P.item(), R.item(), F1.item()

        # Save individual results
        results.append({
            'idx': idx,
            'BERTScore': {'P': P, 'R': R, 'F1': F1},
            'Records Enity Recall': recordsentity_recall,
            'Records Soft Recall': recordssoft_recall,
        })

    # Calculate and save mean metrics
    P_mean = np.mean([item['BERTScore']['P'] for item in results if item['BERTScore']['P'] != 0])
    R_mean = np.mean([item['BERTScore']['R'] for item in results if item['BERTScore']['R'] != 0])
    F1_mean = np.mean([item['BERTScore']['F1'] for item in results if item['BERTScore']['F1'] != 0])
    records_entity_recall_mean = float(np.mean([recall for recall in records_entity_recalls if recall != 0]))
    records_soft_recall_mean = float(np.mean([recall for recall in records_soft_recalls if recall != 0]))

    # Append mean results to the results list
    results.append({
        'BERTScore Mean': {'P': P_mean, 'R': R_mean, 'F1': F1_mean},
        'Records Enity Recall Mean': records_entity_recall_mean,
        'Records Soft Recall Mean': records_soft_recall_mean,
    })

    # Save all results to JSON
    with open(args.gen_records_path + '/results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str,
                        help='Using csv file to store.')
    parser.add_argument('--type', type=str, default='draft',
                        help='Using csv file to store.')
    parser.add_argument('--gen-records-path', type=str, default='../results/draft_records_notes/',
                        help='...')
    main(parser.parse_args())
