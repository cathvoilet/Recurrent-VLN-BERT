import argparse
import json
import numpy as np
import logging
from collections import defaultdict


def vote_instructions(input_file_list, output_file):
    instrid2items = {}
    count = 0
    for input_file in input_file_list:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if instr_id in instrid2items:
                    saved_item = instrid2items[instr_id]
                else:
                    saved_item = item
                score = item['result']
                if 'follower_results' not in saved_item:
                    saved_item['follower_results'] = [score]
                else:
                    saved_item['follower_results'].append(score)
                instrid2items[instr_id] = saved_item
                count += 1

    print("Number of voting outputs read: ", count)

    with open(output_file, 'w') as f:
        json.dump(instrid2items, f, indent=2)
    logging.info('Saved eval info to %s' % output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_exp', help='output exp dir')
    parser.add_argument('-input_exps', '--list', nargs='+', help='input exps list', required=True)
    args = parser.parse_args()

    input_file_list = ["snap/"+agent+"_pi_test_outputs/random_ref_val_seen_eval.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "random_ref_val_seen.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file)

    input_file_list = ["snap/"+agent+"_pi_test_outputs/speaker_gpt2_db7_val_seen.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "speaker_gpt2_db7_val_seen.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file)

    input_file_list = ["snap/"+agent+"_pi_test_outputs/pi_vote-10ila_test-10vln_val_seen.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "pi_vote-10ila_test-10vln_val_seen.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file)

    input_file_list = ["snap/"+agent+"_pi_test_outputs/pi_vote-1ila_test-5vln_val_seen.json" for agent in args.list]
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "pi_vote-1ila_test-5vln_val_seen.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file)
