import argparse
import json
import numpy as np
import logging
from collections import defaultdict


def vote_instructions(input_file_list, output_file, key="ndtw", metric="avg"):
    instrid2scores = defaultdict(list)
    path2instrids = defaultdict(list)

    for input_file in input_file_list:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                score = item['result'][key]
                instrid2scores[instr_id].append(score)
                path_id = instr_id.split("_")[0]
                if instr_id not in path2instrids[path_id]:
                    path2instrids[path_id].append(instr_id)

    best_instructions = []
    print("Agent scores metric: ", metric)
    if metric == "avg":
        best_instructions = best_avg(instrid2scores, path2instrids)
    elif metric == "median":
        best_instructions = best_median(instrid2scores, path2instrids)
    elif metric == "mean-std":
        best_instructions = best_mean_std(instrid2scores, path2instrids)

    # get instruction items by best ids
    all_preds = {}
    count = 0
    with open(input_file_list[0]) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            if instr_id in best_instructions:
                all_preds[item['instr_id']] = item
                count += 1

    print("Number of output instructions: ", count)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    logging.info('Saved eval info to %s' % output_file)


def best_avg(instrid2scores, path2instrids):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = np.array([sum(instrid2scores[instr_id]) for instr_id in instr_ids])
        max_instr_idx = np.argmax(instr_scores)
        best_instructions.append(instr_ids[max_instr_idx])
    return best_instructions


def best_median(instrid2scores, path2instrids):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = []
        for instr_id in instr_ids:
            scores = instrid2scores[instr_id]
            instr_scores.append(np.median(scores))
        instr_scores = np.array(instr_scores)
        max_instr_idx = np.argmax(instr_scores)
        best_instructions.append(instr_ids[max_instr_idx])
    return best_instructions


def best_mean_std(instrid2scores, path2instrids):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = []
        for instr_id in instr_ids:
            scores = instrid2scores[instr_id]
            instr_scores.append(np.average(scores) - np.std(scores))
        instr_scores = np.array(instr_scores)
        max_instr_idx = np.argmax(instr_scores)
        best_instructions.append(instr_ids[max_instr_idx])
    return best_instructions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_exp', help='output exp dir')
    parser.add_argument('-input_exps', '--list', nargs='+', help='input exps list', required=True)
    args = parser.parse_args()

    metric = "avg"

    input_file_list = ["snap/"+agent+"_pi_vote/speaker11_val_seen_eval.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "voted_best_" + metric + "_val_seen_eval.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file, metric=metric)

    input_file_list = ["snap/"+agent+"_pi_vote/speaker11_val_unseen_eval.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "voted_best_" + metric + "_val_unseen_eval.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file, metric=metric)
