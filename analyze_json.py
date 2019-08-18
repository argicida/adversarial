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
    '''
    The 'score' category in the json measures object score, how likely the algorithm this there is an object where the
    person is in the picture. The threshold for saying a person is there is 0.4, and this script prints the fraction
    of images which have an object score above 0.4 for each approach
    :return:
    '''
    # No patch, no noise, just a clean picture
    interpret_results("clean_results.json")
    # Results with a randomized noise patch
    interpret_results("noise_results.json")
    # Results with a trained patch
    interpret_results("patch_results.json")


if __name__ == '__main__':
    main()

