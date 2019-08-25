# Author: Nate Glod
# Filename: analyze_json.py
# Script to intepret the json results for test_patch.py.
import pandas as pd


def interpret_results():
    # number of total detected boxes from clean photos
    total_positives = pd.read_json('clean_results.json').count()[0]
    # number of detected boxes from photos with randomly generated patches
    r_patch_positives = pd.read_json('noise_results.json').count()[0]
    # number of detected boxes from photos with adversarial patches
    patch_positives = pd.read_json('patch_results.json').count()[0]

    # if patched images have false positives, then this recall rate is an upperbound
    print('clean image recall rate: 1 by definition')
    print('randomly patched recall rate: %f' % (r_patch_positives/total_positives))
    print('generated patched recall rate: %f' % (patch_positives/total_positives))


def main():
    interpret_results()


if __name__ == '__main__':
    main()

