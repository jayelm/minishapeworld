"""
Convert to L3 format
"""

import numpy as np
import os
import json


def preprocess_hints(hints):
    hints = list(hints)
    return hints


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='./l3', help='Dataset to load')
    parser.add_argument('--save_dir', default='./l3_sw', help='Directory to save to')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for split in ['train', 'val', 'val_same', 'test', 'test_same']:
        print(split)
        fname = os.path.join(args.dataset, f"{split}.npz")
        data = np.load(fname)
        # Positive examples: first 4
        examples = data['imgs'][:, :4]
        examples = np.transpose(examples, (0, 1, 3, 4, 2))  # CHW -> HWC
        inputs = data['imgs'][:, -1]
        inputs = np.transpose(inputs, (0, 2, 3, 1))
        hints = preprocess_hints(data['langs'])
        labels = data['labels'][:, -1]
        assert np.all(data['labels'][:, :4] == 1)

        split_dir = os.path.join(args.save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        examples_fname = os.path.join(split_dir, 'examples.npy')
        np.save(examples_fname, examples)

        inputs_fname = os.path.join(split_dir, 'inputs.npy')
        np.save(inputs_fname, inputs)

        hints_fname = os.path.join(split_dir, 'hints.json')
        with open(hints_fname, 'w') as f:
            json.dump(hints, f)

        labels_fname = os.path.join(split_dir, 'labels.npy')
        np.save(labels_fname, labels)
