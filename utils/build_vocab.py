import argparse
import pickle
import json
from vocabulary import build_vocab

def main(args):
    vocab = build_vocab(directory=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../data', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)