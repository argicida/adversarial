# Author: Nate Glod
# Filename: analyze_json.py
# Script to intepret the json results for test_patch.py. Automatically tests all the
import pandas as pd


def interpret_results(filename):
    df = pd.read_json(filename)
    total_count = df.count().values[0]
    df = df.drop(df[df.score < 0.4].index)
    success_count = df.count().values[0]
    success_rate = success_count/total_count
    print("Success Rate for " + filename + ": " + str(success_rate))


def main():
    interpret_results("class_only.json")
    interpret_results("class_shift.json")
    interpret_results("clean_results.json")
    interpret_results("noise_results.json")
    interpret_results("patch_results.json")
    interpret_results("patch_simen.json")
    interpret_results("patch_up.json")
    interpret_results("up_results.json")


if __name__ == '__main__':
    main()

