import json
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score


def compute_listener_score1(input_json_file):
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


def compute_listener_score(input_voted_json_file, input_complete_json_file, score_metric="ndtw"):
    print("Rank instructions by: ", score_metric)
    # excluded_models = ["pi_vote-10ila_test-10vln", "speaker_gpt2_db7", "pi_vote-1ila_test-5vln", "speaker-clip_greedy",
    #                   "speaker-clip_vote-10ila", "speaker-clip_vote-1ila", "speaker-gpt_pi-10ila-sample", "speaker-clip10_pi-10ila-sample"]

    path2voted_instrs = defaultdict(list)
    with open(input_voted_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            path2voted_instrs[path_id].append((instr_id, item['overall_voting_result'][score_metric]))  # TODO: try exponential

    path2positive_instrs = defaultdict(list)
    path2negative_instrs = defaultdict(list)
    with open(input_complete_json_file) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            path_id = instr_id.split("_")[0]
            if item['instr_label'] == "positive" or item['model'] == "speaker_ref_agent1_eval":
                path2positive_instrs[path_id].append(instr_id)
            elif item['instr_label'] == "negative":
                path2negative_instrs[path_id].append(instr_id)
            else:
                print("Unknown instr label: ", item['instr_label'])

    count_paths = 0
    path2instr_labels = defaultdict(list)
    avg_precision_list = []
    for path_id, negative_instrs in path2negative_instrs.items():
        count_paths += 1
        positive_instrs = path2positive_instrs[path_id]
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
        # if count_paths <= 2:
        #     print("\nPath id: ", path_id)
        #     print("Instr labels: ", instr_labels)
        #     print("Instr predicted scores: ", voted_instrs)
        #     print("y_true: ", y_true)
        #     print("y_predict: ", y_predict)
        #     print("avg precision: ", avg_precision)

    mean_avg_precision = sum(avg_precision_list) * 1.0 / count_paths
    print("\nAvg precision_list: ", avg_precision_list)
    print("\nNumber of paths counted: ", count_paths)
    print("Mean average precision: ", mean_avg_precision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voted_json_file', help='input voted file')
    parser.add_argument('--input_complete_json_file', help='input original file')
    args = parser.parse_args()
    compute_listener_score(args.input_voted_json_file, args.input_complete_json_file)
