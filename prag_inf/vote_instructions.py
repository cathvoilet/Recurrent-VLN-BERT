import argparse
import json
import numpy as np
import logging
import sys
from collections import defaultdict
from scipy.special import softmax


def vote_instructions(input_file_list, output_file, result_sample, output_duplicate_instrs, output_all_instrs,
                      key="ndtw", metric="avg", no_prob=0, speaker_weight=0.0, speaker_file=None, normalize_speaker=0,
                      speaker_result_key="speaker_result", speaker_model=None, matcher_weight=0.0):
    instr2speaker_score = defaultdict(float)
    path2speaker_scores = defaultdict(list)
    instr2matcher_score = defaultdict(float)
    print("Speaker weight = ", speaker_weight)
    print("VLNBERT matcher weight = ", matcher_weight)
    if speaker_weight > 0.0 or matcher_weight > 0.0:
        if not speaker_file:
            sys.exit("Error: Speaker weight={}, Matcher weight={}, but there is no speaker file".format(
                speaker_weight, matcher_weight))
        else:
            with open(speaker_file) as f:
                tmp_data = json.load(f)
                for instr_id, item in tmp_data.items():
                    instr2speaker_score[instr_id] = item[speaker_result_key][speaker_model]
                    path_id = instr_id.split("_")[0]
                    path2speaker_scores[path_id].append((instr_id, item[speaker_result_key][speaker_model]))
                    if matcher_weight:
                        instr2matcher_score[instr_id] = item["result"]["vln_match"]

    if normalize_speaker:
        for path_id, instr_speaker_scores in path2speaker_scores.items():
            speaker_scores = np.array([x[1] for x in instr_speaker_scores])
            # normalized_scores = (listener_scores - np.min(listener_scores)) / (np.max(listener_scores) - np.min(listener_scores))
            normalized_scores = softmax(speaker_scores)
            for i in range(len(instr_speaker_scores)):
                instr_id, _ = instr_speaker_scores[i]
                normalized_score = normalized_scores[i]
                instr2speaker_score[instr_id] = normalized_score

    path2instrids = defaultdict(list)

    listener_metric2instrid2scores_list = defaultdict(lambda: defaultdict(list))
    if no_prob:
        listener_metrics = ['score', 'spl', 'ndtw', 'sdtw']
    else:
        listener_metrics = ['score', 'spl', 'ndtw', 'sdtw', 'prob']

    count_scores = 0
    for input_file in input_file_list:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if not result_sample:
                    count_scores += 1
                    for metric_key in listener_metrics:
                        listener_score = item['result'][metric_key]
                        instr_score = compute_instruction_score(float(listener_score), speaker_weight=speaker_weight,
                                                                speaker_score=instr2speaker_score[instr_id],
                                                                matcher_weight=matcher_weight,
                                                                matcher_score=instr2matcher_score[instr_id])
                        listener_metric2instrid2scores_list[metric_key][instr_id].append(instr_score)
                else:
                    metric2scores = defaultdict(list)
                    for k in range(result_sample):
                        result_key = "result_sample_{}".format(k)
                        count_scores += 1
                        for metric_key in listener_metrics:
                            metric_score = item[result_key][metric_key]
                            metric2scores[metric_key].append(float(metric_score))
                    for metric_key in listener_metrics:
                        listener_score = np.average(metric2scores[metric_key])
                        instr_score = compute_instruction_score(listener_score, speaker_weight=speaker_weight,
                                                                speaker_score=instr2speaker_score[instr_id],
                                                                matcher_weight=matcher_weight,
                                                                matcher_score=instr2matcher_score[instr_id])
                        listener_metric2instrid2scores_list[metric_key][instr_id].append(instr_score)

                path_id = instr_id.split("_")[0]
                if instr_id not in path2instrids[path_id]:
                    path2instrids[path_id].append(instr_id)

    print("Number of scores counted:", count_scores)

    # Ensemble listeners variance
    for listener_metric, instrid2scores_list in listener_metric2instrid2scores_list.items():
        variance_list = []
        for instr_id, scores in instrid2scores_list.items():
            variance_list.append(np.var(scores))
        print("Ensemble listener variance for listener metric {}: {}".format(listener_metric, round(np.average(variance_list), 3)))

    # Get best instruction id
    best_instructions = []
    instrid2scores = listener_metric2instrid2scores_list[key]
    print("Agent scores metric: ", metric)
    print("Listener score to choose best instruction: ", key)
    if metric == "avg":
        best_instructions = best_avg(instrid2scores, path2instrids, output_duplicate_instrs)
    # elif metric == "median":
    #     best_instructions = best_median(instrid2scores, path2instrids)
    # elif metric == "mean-std":
    #     best_instructions = best_mean_std(instrid2scores, path2instrids)
    elif metric == "product":
        best_instructions = best_product(instrid2scores, path2instrids, output_duplicate_instrs)

    if output_all_instrs:
        best_instructions = list(instrid2scores.keys())

    # get instruction items by best ids
    all_preds = {}
    count = 0
    with open(input_file_list[0]) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            if instr_id in best_instructions:
                if result_sample:
                    for k in range(result_sample):
                        result_key = "result_sample_{}".format(k)
                        path_key = "pred_path_sample_{}".format(k)
                        del item[result_key]
                        del item[path_key]
                else:
                    del item["result"]
                    del item["pred_path"]
                overall_scores = {}
                for listener_metric, instrid2scores_list in listener_metric2instrid2scores_list.items():
                    scores = instrid2scores_list[instr_id]
                    if metric == "avg":
                        overall_score = np.average(scores)
                    elif metric == "product":
                        # overall_score = sum(np.log(scores))
                        overall_score = np.product(scores)
                    overall_scores[listener_metric] = overall_score
                item['overall_voting_result'] = overall_scores
                all_preds[item['instr_id']] = item
                count += 1

    print("Number of output instructions: ", count)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    print('Saved eval info to %s' % output_file)


def compute_instruction_score(listener_score, speaker_weight=0.0, speaker_score=0.0,
                              matcher_weight=0.0, matcher_score=0.0):
    instr_score = listener_score
    if speaker_weight:
        instr_score = pow(instr_score, 1.0 - speaker_weight) * pow(speaker_score, speaker_weight)
    if matcher_weight:
        scale = 1e-2
        instr_score = pow(instr_score, 1.0 - matcher_weight) * pow(matcher_score * scale, matcher_weight)

    return instr_score


def best_avg(instrid2scores, path2instrids, output_duplicate_instrs):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = np.array([np.average(instrid2scores[instr_id]) for instr_id in instr_ids])
        max_score = max(instr_scores)
        if not output_duplicate_instrs:
            max_instr_idx = np.argmax(instr_scores)
            best_instructions.append(instr_ids[max_instr_idx])
        else:
        #elif max_score > 0.0:
            max_instr_indices = [i for i, j in enumerate(instr_scores) if j == max_score]
            best_instructions += [instr_ids[x] for x in max_instr_indices]

    return best_instructions


def best_product(instrid2scores, path2instrids, output_duplicate_instrs):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = np.array([sum(np.log(instrid2scores[instr_id])) for instr_id in instr_ids])
        max_score = max(instr_scores)
        if not output_duplicate_instrs:
            max_instr_idx = np.argmax(instr_scores)
            best_instructions.append(instr_ids[max_instr_idx])
        else:
        #elif max_score != 0.0:
            max_instr_indices = [i for i, j in enumerate(instr_scores) if j == max_score]
            best_instructions += [instr_ids[x] for x in max_instr_indices]

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
    parser.add_argument('--input_path', help='input file path')
    parser.add_argument('--output_duplicate_instrs', type=int, default=0)
    parser.add_argument('--output_all_instrs', type=int, default=0)
    parser.add_argument('--metric', type=str, default="avg")
    parser.add_argument('--listener_model', type=str, default="ndtw")
    parser.add_argument('--no_prob', type=int, default=0)
    parser.add_argument('-input_exps', '--list', nargs='+', help='input exps list', required=True)
    parser.add_argument('--result_sample', type=int, default=0)  # 0, 10
    parser.add_argument('--speaker_weight', type=float, default=0.0)
    parser.add_argument('--normalize_speaker', default=0, help='speaker normalize')
    parser.add_argument('--speaker_result_key', default="speaker_result", help='speaker key')
    parser.add_argument('--speaker_file', type=str, default=None)
    parser.add_argument('--speaker_model', type=str, default=None)
    parser.add_argument('--matcher_weight', type=float, default=0.0, help='matcher weight')
    args = parser.parse_args()

    score_metric = args.listener_model
    print("Choose by best ", score_metric)
    print("Allow duplicate instrs: ", args.output_duplicate_instrs)
    print("Outputing all instrs: ", args.output_all_instrs)

    input_file_list = ["snap/" + agent + "_" + args.input_path for agent in args.list]
    print("Input file list: ", input_file_list)
    if args.output_all_instrs:
        output_file = args.output_exp + "voted_all_" + args.metric + "_val_seen_eval.json"
    else:
        output_file = args.output_exp + "voted_best_" + args.metric + "_val_seen_eval.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file, args.result_sample, args.output_duplicate_instrs, args.output_all_instrs,
                      metric=args.metric, key=score_metric, no_prob=args.no_prob,
                      speaker_weight=args.speaker_weight, speaker_file=args.speaker_file, speaker_model=args.speaker_model,
                      normalize_speaker=args.normalize_speaker, speaker_result_key=args.speaker_result_key,
                      matcher_weight=args.matcher_weight)
