# load.py
# Preprocess directories and load data for translate.py.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob
import re

def load_directory(directory, subset):
    lsent = []
    csent = []
    filenames = []
    if subset == 0: # Full data.
        for filename in glob.glob(directory):
            file = open(filename, 'r')
            tokens = file.read().split()
            
            lsent.append(tokens[:tokens.index('~')])
            csent.append(tokens[tokens.index('~')+1:])
            filenames.append(filename)
    elif subset == 1: # Th'm only.
        for filename in glob.glob(directory):
            file = open(filename, 'r')
            tokens = file.read().split()
            
            lsent.append(tokens[:tokens.index(r'\begin{proof}')])
            csent.append(tokens[tokens.index('~')+1:tokens.index('Proof')])
            filenames.append(filename)
    elif subset == 2: # Proof only.
        for filename in glob.glob(directory):
            file = open(filename, 'r')
            tokens = file.read().split()
            
            lsent.append(tokens[tokens.index(r'\begin{proof}'):tokens.index('~')])
            csent.append(tokens[tokens.index('Proof'):])
            filenames.append(filename)
    return lsent, csent, np.array(filenames)


def get_vocabulary(lsent, csent):
    lcounts = {}
    ccounts = {}

    # Count LaTeX tokens occurrences
    for example in lsent:
        for token in example:
            if re.match("<nat:.+", token):
                token = "<nat>"
            elif re.match("<var:.+", token):
                token = "<var>"
            elif re.match("<def:.+", token):
                token = "<def>"
            else:
                token = token.lower()
            if token in lcounts:
                lcounts[token] += 1
            else:
                lcounts[token] = 1

    # Count Coq token occurrences
    for example in csent:
        for token in example:
            if re.match("<nat:.+", token):
                token = "<nat>"
            if re.match("<var:.+", token):
                token = "<var>"
            if re.match("<def:.+", token):
                token = "<def>"
            if re.match("<genH:.+", token):
                token = "<genH>"
            if token in ccounts:
                ccounts[token] += 1
            else:
                ccounts[token] = 1

    # Create initial vocabs
    vocab_latex = sorted(lcounts)
    vocab_coq = sorted(ccounts)

    vocab_latex = ['<pad>', '<oov>', '<start>', '<eof>', '<def>'] + ['<nat' + str(x) + '>' for x in range(20)] + ['<var' + str(x) + '>' for x in range(20)] + vocab_latex
    vocab_coq = ['<pad>', '<oov>', '<start>', '<eof>', '<def>'] + ['<nat' + str(x) + '>' for x in range(20)] + ['<var' + str(x) + '>' for x in range(20)] + vocab_coq

    return vocab_latex, vocab_coq

def get_input_seq(sentences):
    input_seqs = []
    copy_vocabs = []

    for example in sentences:
        input_seq = [1]
        copy_vocab = ["<pad>", "<start>", "<eof>"]

        for token in example:
            if token not in copy_vocab:
                copy_vocab.append(token)
            input_seq.append(copy_vocab.index(token))

        input_seq.append(2)
        input_seqs.append(input_seq)
        copy_vocabs.append(copy_vocab)

    return input_seqs, copy_vocabs

def convert_ldict_keys(sentences, vocab, input_seqs, copy_vocabs, names=None):
    token_maps = []
    for i in range(len(sentences)):
        sentences[i].insert(0, "<start>")
        sentences[i].append("<eof>")

        token_map = {}
        nats = 0
        vars = 0
        for j in range(len(sentences[i])):
            if re.match("<nat:.+", sentences[i][j]):
                # Replace with generic token.
                if not sentences[i][j] in token_map:
                    token_map[sentences[i][j]] = "<nat" + str(min(nats, 19)) + ">"
                    nats += 1
                sentences[i][j] = vocab.index(token_map[sentences[i][j]])
            elif re.match("<var:.+", sentences[i][j]):
                if not sentences[i][j] in token_map:
                    token_map[sentences[i][j]] = "<var" + str(min(vars, 19)) + ">"
                    vars += 1
                sentences[i][j] = vocab.index(token_map[sentences[i][j]])
            elif re.match("<def:.+", sentences[i][j]):
                sentences[i][j] = vocab.index("<def>")
            elif sentences[i][j] == "<start>" or sentences[i][j] == "<eof>":
                sentences[i][j] = vocab.index(sentences[i][j])
            else:
                try:
                    sentences[i][j] = vocab.index(sentences[i][j].lower())
                except ValueError:
                    sentences[i][j] = vocab.index("<oov>")
        token_maps.append(token_map)
    return {'sentences': sentences, 'input_seqs': input_seqs, 'copy_vocabs': copy_vocabs, 'token_maps': token_maps}


def convert_cdict_keys(sentences, vocab, copy_vocabs):
    # For testing theorems, we need to condition out a small subset of examples where copying info is only in the proof.
    bad_examples = []

    for i in range(len(sentences)):
        sentences[i].insert(0, "<start>")
        sentences[i].append("<eof>")

        for j in range(len(sentences[i])):
            if re.match("<nat:.+", sentences[i][j]):
                # OOV token should correspond to vocab + position in the copy vocabulary.
                try:
                    sentences[i][j] = len(vocab) + copy_vocabs[i].index(sentences[i][j])
                except:
                    bad_examples.append(i)
                    break
            elif re.match("<var:.+", sentences[i][j]):
                try:
                    sentences[i][j] = len(vocab) + copy_vocabs[i].index(sentences[i][j])
                except:
                    bad_examples.append(i)
                    break
            elif re.match("<def:.+", sentences[i][j]):
                try:
                    sentences[i][j] = len(vocab) + copy_vocabs[i].index(sentences[i][j])
                except:
                    bad_examples.append(i)
                    break
            elif re.match("<genH:.+", sentences[i][j]):
                sentences[i][j] = "<genH>"
                sentences[i][j] = vocab.index(sentences[i][j])
            else:
                try:
                    sentences[i][j] = vocab.index(sentences[i][j])
                except:
                    bad_examples.append(i)
                    break
    return {'sentences': sentences}, bad_examples


def pad_sentences(latex, coq):
    latex['sentences'] = np.asarray(latex['sentences'])

    latex['input_seqs'] = np.asarray(latex['input_seqs'])
    latex['copy_vocabs'] = np.asarray(latex['copy_vocabs'])
    latex['token_maps'] = np.asarray(latex['token_maps'])

    coq['sentences'] = np.asarray(coq['sentences'])

    return latex, coq

def remove_bad_cases(latex_sentences, coq_sentences, bad_cases):
    original = len(bad_cases)
    
    while bad_cases:
        idx = bad_cases[0]
        offset = original - len(bad_cases)
        for key in latex_sentences:
            latex_sentences[key].pop(idx - offset)
        for key in coq_sentences:
            coq_sentences[key].pop(idx - offset)

        bad_cases.pop(0)

    return latex_sentences, coq_sentences


def fetch_data(vocab_latex, vocab_coq, names, subset):
    lsent = []
    csent = []
    for fname in names:
        l, c, _ = load_directory(fname, subset)
        lsent += l
        csent += c

    seq_latex, copy = get_input_seq(lsent)

    lkeys = convert_ldict_keys(lsent, vocab_latex, seq_latex, copy, names)
    ckeys, bad_cases = convert_cdict_keys(csent, vocab_coq, copy)

    lkeys, ckeys = remove_bad_cases(lkeys, ckeys, bad_cases)

    latex, coq = pad_sentences(lkeys, ckeys)

    return {'latex': latex, 'coq': coq}


def load_data(dataset_path, test_path, subset):
    train_lsent, train_csent, train_names = load_directory(os.path.join(dataset_path, "training/*.txt"), subset)
    test_lsent, _, test_names = load_directory(os.path.join(dataset_path, test_path + "/*.txt"), subset)
    
    vocab_latex, vocab_coq = get_vocabulary(train_lsent, train_csent)

    max_oov = 0
    for one in test_lsent:
        if len(one) > max_oov:
            max_oov = len(one)

    return max_oov, {'latex': vocab_latex, 'coq': vocab_coq}, {'train': train_names, 'test': test_names}
