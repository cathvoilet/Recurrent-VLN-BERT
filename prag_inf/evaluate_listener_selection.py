import json
import argparse
from collections import defaultdict
import numpy as np
import random
from scipy.stats import sem
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score


def compute_listener_score1(input_json_file, input_complete_json_file):
    path_list = []
    model2count = defaultdict(int)
    path2models_counted = defaultdict(list)
    with open(input_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if path_id not in path_list:
                path_list.append(path_id)
            instr_type = item["instr_type"]
            if instr_type not in path2models_counted[path_id]:
                model2count[instr_type] += 1
            path2models_counted[path_id].append(instr_type)

    count_paths = len(path_list)
    print("\nInput file for computing listener score: ", input_json_file)
    print("Number of total paths count: ", count_paths)
    for model, count in model2count.items():
        ratio = 100.0 * count / count_paths
        print("Model {} selection ratio: {}".format(model, str(ratio)))


def compute_listener_score2(input_voted_json_file, input_complete_json_file):
    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            path2voted_instrs[path_id].append(instr_id)

    path2positive_instrs = defaultdict(list)
    path2negative_instrs = defaultdict(list)
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if item['instr_label'] == "positive":
                path2positive_instrs[path_id].append(instr_id)
            elif item['instr_label'] == "negative":
                path2negative_instrs[path_id].append(instr_id)
            else:
                print("Unknown instr label: ", item['instr_label'])

    labels = []
    predictions = []
    for path_id, negative_instrs in path2negative_instrs.items():
        positive_instrs = path2positive_instrs[path_id]
        voted_instrs = path2voted_instrs[path_id]
        if positive_instrs:
            labels.append(1)
        else:
            labels.append(0)
        if set(positive_instrs).intersection(voted_instrs):
            predictions.append(1)
        else:
            predictions.append(0)

        for instr_id in negative_instrs:
            labels.append(0)
            if instr_id in voted_instrs:
                predictions.append(1)
            else:
                predictions.append(0)

    print("Number of labels: ", len(labels))
    print("Number of predictions: ", len(predictions))
    print("Labels:\n", labels)
    print("Predictions:\n", predictions)

    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)
    print("Confusion matrix: labels=[0, 1]")
    print(cm)


def compute_listener_score3(input_voted_json_file, input_complete_json_file, score_metric="ndtw"):
    print("\nRank instructions by: ", score_metric)
    # excluded_models = ["pi_vote-10ila_test-10vln", "speaker_gpt2_db7", "pi_vote-1ila_test-5vln", "speaker-clip_greedy",
    #                   "speaker-clip_vote-10ila", "speaker-clip_vote-1ila", "speaker-gpt_pi-10ila-sample", "speaker-clip10_pi-10ila-sample"]
    count_paths = 0
    avg_precision_list = []
    ref_systems = [0, 1, 2]
    for ref_system in ref_systems:
        path2voted_instrs = defaultdict(list)
        with open(input_voted_json_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if item["ref_id"] != ref_system:
                    continue
                path_id = instr_id.split("_")[0]
                path2voted_instrs[path_id].append((instr_id, item['overall_voting_result'][score_metric]))

        path2positive_instrs = defaultdict(list)
        path2negative_instrs = defaultdict(list)
        with open(input_complete_json_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if item["ref_id"] != ref_system:
                    continue
                path_id = instr_id.split("_")[0]
                if item['instr_label'] == "positive" or item['model'] == "speaker_ref_agent1_eval":
                    path2positive_instrs[path_id].append(instr_id)
                elif item['instr_label'] == "negative":
                    path2negative_instrs[path_id].append(instr_id)
                else:
                    print("Unknown instr label: ", item['instr_label'])

        # count_paths = 0
        path2instr_labels = defaultdict(list)
        # avg_precision_list = []
        sorted_path_ids = sorted(list(path2negative_instrs.keys()))

        for path_id in sorted_path_ids:
            count_paths += 1
            positive_instrs = path2positive_instrs[path_id]
            negative_instrs = path2negative_instrs[path_id]
            voted_instrs = list(path2voted_instrs[path_id])
            instr_labels = []
            for instr_id, score in voted_instrs:
                if instr_id in positive_instrs:
                    instr_labels.append((instr_id, 1))
                elif instr_id in negative_instrs:
                    instr_labels.append((instr_id, 0))
                else:
                    print("WARNING: instr id not in either positive or negative category: ", instr_id)
            path2instr_labels[path_id] = instr_labels
            y_true = np.array([x[1] for x in instr_labels])
            y_predict = np.array([x[1] for x in voted_instrs])
            avg_precision = average_precision_score(y_true, y_predict)
            avg_precision_list.append(avg_precision)

    mean_avg_precision = sum(avg_precision_list) * 1.0 / count_paths
    print("\nAvg precision_list: \n", avg_precision_list)
    print("\nNumber of paths counted: ", count_paths)
    print("Mean average precision: ", mean_avg_precision)


def compute_listener_score(input_voted_json_file, input_complete_json_file, score_metric="ndtw", speaker_weight=0.0, speaker_model=None):
    print("\n\nRank instructions by: ", score_metric)
    print("Listener weight: ", 1.0-speaker_weight)
    print("Speaker weight: ", speaker_weight)
    print("Speaker score model: ", speaker_model)
    # excluded_models = ["pi_vote-10ila_test-10vln", "speaker_gpt2_db7", "pi_vote-1ila_test-5vln", "speaker-clip_greedy",
    #                   "speaker-clip_vote-10ila", "speaker-clip_vote-1ila", "speaker-gpt_pi-10ila-sample", "speaker-clip10_pi-10ila-sample"]

    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if score_metric not in item['overall_voting_result']:
                print("Exiting: metric {} not in voting file!".format(score_metric))
                return False, False
            path2voted_instrs[path_id].append((instr_id, item['overall_voting_result'][score_metric]))

    path2positive_instrs = defaultdict(list)
    path2negative_instrs = defaultdict(list)
    instr2speaker_score = {}
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            # if item['instr_label'] == "positive" or item['model'] == "speaker_ref_agent1_eval":
            if item['instr_label'] == "positive":
                path2positive_instrs[path_id].append(instr_id)
            elif item['instr_label'] == "negative":
                path2negative_instrs[path_id].append(instr_id)
            else:
                print("Unknown instr label: ", item['instr_label'])

            if speaker_weight:
                instr2speaker_score[instr_id] = item["speaker_result"][speaker_model]

    # count_paths = 0
    # avg_precision_list = []
    sorted_path_ids = sorted(list(path2negative_instrs.keys()))
    num_negatives_per_group = 4
    count_paths = 0
    count_groups = 0
    num_pos_instrs, num_neg_instrs = 0, 0
    avg_precision_list = []

    for path_id in sorted_path_ids:
        positive_instrs = path2positive_instrs[path_id]
        negative_instrs = path2negative_instrs[path_id]
        voted_instrs = list(path2voted_instrs[path_id])
        count_paths += 1
        num_pos_instrs += len(positive_instrs)
        num_neg_instrs += len(negative_instrs)

        random.shuffle(negative_instrs)
        negative_groups = [negative_instrs[i:i + num_negatives_per_group] for i in range(0, len(negative_instrs), num_negatives_per_group)]
        for positive_instr in positive_instrs:
            for negative_group in negative_groups:
                count_groups += 1
                instr_labels = []
                instr_preds = []
                for instr_id, score in voted_instrs:
                    if speaker_weight:
                        speaker_score = instr2speaker_score[instr_id]
                        score = pow(score, 1.0-speaker_weight) * pow(speaker_score, speaker_weight)
                        # if score == 0.0 or speaker_score == 0.0:
                        #     score = -1.7976931348623157e+308
                        # else:
                        #     score = (1.0-speaker_weight) * np.log(score) + speaker_weight * np.log(speaker_score)
                    if instr_id == positive_instr:
                        instr_labels.append((instr_id, 1))
                        instr_preds.append((instr_id, score))
                    elif instr_id in negative_group:
                        instr_labels.append((instr_id, 0))
                        instr_preds.append((instr_id, score))
                    #else:
                    #    print("WARNING: instr id not in either positive or negative category: ", instr_id)
                y_true = np.array([x[1] for x in instr_labels])
                y_predict = np.array([x[1] for x in instr_preds])
                avg_precision = average_precision_score(y_true, y_predict)
                avg_precision_list.append(avg_precision)

    print("Number of paths counted: ", count_paths)
    print("Number of positive instrs counted: ", num_pos_instrs)
    print("Number of negative instrs counted: ", num_neg_instrs)

    # mean_avg_precision = sum(avg_precision_list) * 1.0 / count_groups
    mean_avg_precision = np.average(avg_precision_list)
    ste_ap = 1.44 * sem(avg_precision_list)
    # print("\nAvg precision_list: \n", avg_precision_list)
    print("Number of groups counted: ", count_groups)
    output_str = "Mean average precision +- 1.44*ste = {:.1f}+-{:.1f}".format(100 * mean_avg_precision, 100 * ste_ap)
    print(output_str)
    return mean_avg_precision, output_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voted_json_file', help='input voted file')
    parser.add_argument('--input_complete_json_file', help='input original file')
    args = parser.parse_args()

    metrics = ['ndtw', 'sdtw', 'spl', 'score', 'prob']
    speaker_weights = [0.1 * x for x in range(0, 11)]
    # speaker_weights = [0.0]

    metric2best_score = defaultdict(float)
    metric2best_string = defaultdict(str)
    for speaker_weight in speaker_weights:
        for metric in metrics:
            score, output_str = compute_listener_score(args.input_voted_json_file, args.input_complete_json_file, score_metric=metric, speaker_weight=speaker_weight, speaker_model="finetuned_gpt")
            if score > metric2best_score[metric]:
                print("New best score for metric {}: {}".format(metric, score))
                metric2best_score[metric] = score
                metric2best_string[metric] = output_str + " (lda={})".format(1.0-speaker_weight)

    for metric in metrics:
        print("\nFinal best score for metric: ", metric)
        print(metric2best_string[metric])
