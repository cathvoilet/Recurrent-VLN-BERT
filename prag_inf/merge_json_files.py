import json


def merge_files_gpt(input_files, output_file):
    all_preds = {}
    count = 0
    for i, input_file in enumerate(input_files):
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if i == 0:
                    #item["model"] = "speaker_gpt2_db7"
                    path_id = instr_id.split("_")[0]
                    instr_id = path_id + "_" + "beam"
                    item["instr_id"] = instr_id
                #else:
                #    item["model"] = "pi_vote-10ila_test-10vln"
                all_preds[instr_id] = item
                count += 1

    print("Saved number of items {} from two files:". format(count))
    print(input_files)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    print('Saved eval info to %s' % output_file)


def merge_files(input_files, output_file):
    all_preds = {}
    count = 0
    for input_file in input_files:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if instr_id not in all_preds:
                    all_preds[item['instr_id']] = item
                    count += 1

    print("Saved number of items {} from two files:". format(count))
    print(input_files)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    print('Saved eval info to %s' % output_file)

#input_files = ["/vulcanscratch/lzhao/repos/CLIP-ViL/Matterport3DSimulator/snap/speaker_clip_vit/decoded_outputs/val_seen_greedy_prob.pred", "/vulcanscratch/lzhao/repos/Matterport3DSimulator/experiments/pi_vote-10ila_speaker-clip/voted_best_avg_val_seen_eval.json"]
#output_file = "/vulcanscratch/lzhao/repos/Matterport3DSimulator/clip_outputs/speaker-clip_greedy_vote_10ila_combined_val_seen.json"
input_files = ["/vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/speaker_gpt2_db7_val_seen.pred", "/vulcanscratch/lzhao/repos/Matterport3DSimulator/experiments/pi_vote-10ila_test-10vln/voted_best_avg_val_seen_eval.json"]
output_file = "/vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/speaker-gpt_beam_vote_10ila_combined_val_seen.json"

merge_files_gpt(input_files, output_file)
