# translate.py
# Driver for training and testing of transformer model for LaTeX to Coq translation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from load import fetch_data, load_data
from model import make_model
from util import *

parser = argparse.ArgumentParser('Translate Proofs')
parser.add_argument('--dataset_path',
                    type=str,
                    default='data',
                    help='Path of the dataset directory containing subfolders to training/test/etc.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to run training for.')
parser.add_argument('--batch_size',
                    type=int,
                    default=10,
                    help='Batch size for training.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='Initial learning rate for Adam.')
parser.add_argument('--N',
                    type=int,
                    default=4,
                    help='Stacked layers of encoders/decoders.')
parser.add_argument('--T',
                    type=int,
                    default=4,
                    help='Recursive passes through encoder/decoder stack.')
parser.add_argument('--d_model',
                    type=int,
                    default=64,
                    help='Size of encoder/decoder states.')
parser.add_argument('--d_ff',
                    type=int,
                    default=512,
                    help='Width of the feed-forward layers.')
parser.add_argument('--H',
                    type=int,
                    default=16,
                    help='Number of attention heads.')
parser.add_argument('--k',
                    type=int,
                    default=2,
                    help='Width of relative self-attention clipping.')
parser.add_argument('--dropout',
                    type=float,
                    default=0.25,
                    help='Dropout probability.')
parser.add_argument('--alpha',
                    type=float,
                    default=0.0,
                    help='Label smoothing rate.')
parser.add_argument('--test',
                    type=str,
                    default='',
                    help='Load pre-trained model and jump to evaluation on specified dataset.')
parser.add_argument('--subset',
                    type=int,
                    default=0)
parser.add_argument('--load',
                    action='store_true',
                    help='Optionally load model for training.')
parser.add_argument('--model',
                    type=str,
                    default='ltc',
                    help='Name of the .pt file the model will be saved to/loaded from.')
parser.add_argument('--scheduler',
                    type=str,
                    default='step',
                    help='Learning rate scheduler. Usage: --scheduler [step (default) | exponential | plateau]')
parser.add_argument('--step',
                    type=int,
                    default=25,
                    help='Step size for StepLR learning rate annealing.')
parser.add_argument('--gamma',
                    type=float,
                    default=0.9,
                    help='Gamma value for ExponentialLR learning rate annealing.')
parser.add_argument('--patience',
                    type=int,
                    default=10,
                    help='Patience for ReduceLROnPlateau learning rate annealing.')

args, _ = parser.parse_known_args()

test_name = args.test if args.test else "test"

# Initial load of data, returns file names from train/test data.
max_oov, vocabs, example_names = load_data(args.dataset_path, test_name, args.subset)
train_size = len(example_names['train'])
test_size = len(example_names['test'])

coq_size = len(vocabs['coq'])
max_latex_size = len(vocabs['latex'])
max_output_size = len(vocabs['coq']) + max_oov

use_gpu = torch.cuda.is_available()
tensor_type = torch.cuda.LongTensor if use_gpu else torch.LongTensor

if not args.test:
    # Load or create new model.
    if args.load:
        model = to_cuda(torch.load(args.model + ".pt"))
    else:
        model = to_cuda(make_model(max_latex_size, coq_size, N=args.N, T=args.T, d_model=args.d_model, d_ff=args.d_ff, h=args.H, k=args.k, dropout=args.dropout))

    criterion = to_cuda(LabelSmoothing(size=max_output_size, padding_idx=0, smoothing=args.alpha))

    # Init. optimizer and choose LR scheduling.
    model_opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    if args.scheduler == "step":
        model_sched = torch.optim.lr_scheduler.StepLR(model_opt, step_size=args.step, gamma=0.1**0.5)
    elif args.scheduler == "exponential":
        model_sched = torch.optim.lr_scheduler.ExponentialLR(model_opt, gamma=args.gamma)
    elif args.scheduler == "plateau":
        model_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(model_opt, factor=0.1**0.5, patience=args.patience)

    train_losses = []
    train_thresh = np.inf

    train_set = fetch_data(vocabs['latex'], vocabs['coq'], example_names['train'], args.subset)
    train_size = len(train_set['latex']['sentences'])

    # Run training / eval for no. of epochs.
    model.train()
    for epoch in range(args.num_epochs):
        # Init some vars.
        s = 0
        loss = 0
        start = time.time()

        # Shuffle data for current epoch.
        train_shuffle = np.arange(train_size)
        np.random.shuffle(train_shuffle)

        train_loss = 0
        while s < train_size:
            e = min(s + args.batch_size, train_size)

            # Get batches of examples from source and target.
            train_src = train_set['latex']['sentences'][train_shuffle][s:e]
            train_seq = train_set['latex']['input_seqs'][train_shuffle][s:e]
            train_copy = train_set['latex']['copy_vocabs'][train_shuffle][s:e]
            train_trg = train_set['coq']['sentences'][train_shuffle][s:e]
            train_batch = Batch(train_src, train_trg, train_seq, train_copy, vocabs['coq'])

            # Update loop.
            s = e

            # Make sure model is in training mode and train on batch.
            train_loss += run_batch(train_batch, model, SimpleLossCompute(model.generator, criterion, opt=model_opt))

            del train_batch
            if use_gpu:
                torch.cuda.empty_cache()

        with torch.no_grad():
            # Plot loss every epoch.
            train_loss /= np.ceil(train_size / args.batch_size)
            train_losses.append(train_loss)

            if train_loss <= train_thresh:
                train_thresh = train_loss
                torch.save(model, args.model + ".pt")

        if args.scheduler == "plateau":
            model_sched.step(train_loss)
        else:
            model_sched.step()

        # Output some data on current epoch.
        elapsed = time.time() - start
        print("Epoch Step: %d Time: %f\n\tTraining Loss: %f" % (epoch, elapsed, train_loss), file=sys.stderr)

    del train_set
    if use_gpu:
        torch.cuda.empty_cache()

# If in testing mode, load the model from file and start optimizer.
model = torch.load(args.model + ".pt")
model.eval()

with torch.no_grad():
    s = 0
    extra_vars = 0
    wrong_pars = 0
    elapsed = []
    perfects = []
    accs = []
    streaks = []
    sub_accs = {}
    sub_char_accs = {}

    test_set = fetch_data(vocabs['latex'], vocabs['coq'], example_names['test'], args.subset)
    test_size = len(test_set['latex']['sentences'])

    while s < test_size:
        start = time.time()
        
        # Get batches of examples.
        test_src = test_set['latex']['sentences'][s:s + 1]
        test_seq = test_set['latex']['input_seqs'][s:s + 1]
        test_copy = test_set['latex']['copy_vocabs'][s:s + 1]
        test_trg = test_set['coq']['sentences'][s:s + 1]
        batch = Batch(test_src, test_trg, test_seq, test_copy, vocabs['coq'])

        # Slice out an example from batch and pass it for decoding.
        src = batch.src[:1]
        src_mask = (src != 0).unsqueeze(-2)

        out = greedy_decode(model, src, src_mask, batch.input_seqs, batch.copy_vocab, vocabs['coq'], max_len=batch.trg.size(1))

        # Print out the translation.
        print("\n\n=== Example " + str(s) + ": " + example_names['test'][s] + " ===")
        print("Translation:", end="\t")
        trans = ""
        trans_sym_count = 0
        
        for i in range(1, out.size(1)):
            sym = "<unk>"
            if out[0, i] < coq_size:
                sym = vocabs['coq'][out[0, i]]
            elif out[0, i] - coq_size < len(test_set['latex']['copy_vocabs'][s]):
                sym = test_set['latex']['copy_vocabs'][s][out[0, i] - coq_size]

            if sym == "<eof>": break
            trans += sym + " "
            trans_sym_count += 1

        print(trans)

        # Print out the target.
        print("Target:", end="\t")
        streak = 0
        correct = 0
        exact = 1

        sym_count = 0
        target = ""

        print("\n" + example_names['test'][s], file=sys.stderr)

        for i in range(1, batch.trg.size(1)):
            sym = vocabs['coq'][batch.trg_y.data[0, i - 1]] if batch.trg_y.data[0, i - 1] < coq_size else test_set['latex']['copy_vocabs'][s][batch.trg_y.data[0, i - 1] - coq_size]
            
            sym_count += 1
            # Check for correctness of translation.
            if out[0, i] < coq_size and sym == vocabs['coq'][out[0, i]]:
                if exact: streak += 1
                correct += 1
            elif out[0, i] >= coq_size and sym == test_set['latex']['copy_vocabs'][s][out[0, i] - coq_size]:
                if exact: streak += 1
                correct += 1
            else:
                if exact:
                    print("Translation " + str(s) + " -- Generated " + (vocabs['coq'][out[0, i]] if out[0, i] < coq_size else test_set['latex']['copy_vocabs'][s][out[0, i] - coq_size]) + " instead of " + sym, file=sys.stderr)
                exact = 0

            if sym == "<eof>": break
            target += sym + " "

        print(target)

        # Pattern match the file name to get the dataset example is from.
        if "even-odd" in example_names['test'][s]:
            subset = "even-odd_" + str(example_names['test'][s].count('_') - 1)
        elif "powers" in example_names['test'][s]:
            subset = "powers"
        elif "composites" in example_names['test'][s]:
            subset = "composites_" + example_names['test'][s].split('_')[-2]
        elif "poly" in example_names['test'][s]:
            subset = "poly_" + str(target.count('%assertion'))
        else:
            subset = "misc"

        # Save stats by subset.
        if subset in sub_accs:
            sub_accs[subset].append(exact)
            sub_char_accs[subset].append(correct / (1.0 * sym_count))
        else:
            sub_accs[subset] = [exact]
            sub_char_accs[subset] = [correct / (1.0 * sym_count)]

        if target != trans:
            exact = 0
            sym_count = max(sym_count, trans_sym_count)

        # Print immediate results, incl. rolling averages.
        elapsed.append(time.time() - start)
        print("Translation %d Time: %f Accuracy: (%d / %d)" % (s, elapsed[s], correct, sym_count), file=sys.stderr)
        print("[" + subset + "] Seq.: " + str(torch.sum(torch.Tensor(sub_accs[subset])) / (1.0 * len(sub_accs[subset]))) + "\tChar.: " + str(torch.sum(torch.Tensor(sub_char_accs[subset])) / (1.0 * len(sub_char_accs[subset]))), file=sys.stderr)

        perfects.append(exact)
        streaks.append(streak)
        accs.append(correct / (1.0 * sym_count))

        # Update loop.
        s += 1

        del batch
        if use_gpu:
            torch.cuda.empty_cache()

    del test_set
    if use_gpu:
        torch.cuda.empty_cache()

print("\n\n\n\n\n===   TESTING STATS   ===", file=sys.stderr)
print("Total Translation Time: %f(s)" % torch.sum(torch.Tensor(elapsed)), file=sys.stderr)
print("Avg. Translation Time:  %f(s)" % (torch.sum(torch.Tensor(elapsed)) / (1.0 * test_size)), file=sys.stderr)
print("Total Perfections:      %d" % torch.sum(torch.Tensor(perfects)), file=sys.stderr)
print("Perfection Accuracy:    %f" % (torch.sum(torch.Tensor(perfects)) / (1.0 * test_size)), file=sys.stderr)
print("General Accuracy:       %f" % (torch.sum(torch.Tensor(accs)) / (1.0 * test_size)), file=sys.stderr)
print("Avg. Streak:            %f" % (torch.sum(torch.Tensor(streaks)) / (1.0 * test_size)), file=sys.stderr)
print("", file=sys.stderr)
print("By Length:", file=sys.stderr)
for key in sorted(sub_accs):
    print("\t[" + str(key) + "]\tSeq.: " + str(torch.sum(torch.Tensor(sub_accs[key])) / (1.0 * len(sub_accs[key]))) + "\tTok.: " + str(torch.sum(torch.Tensor(sub_char_accs[key])) / (1.0 * len(sub_char_accs[key]))), file=sys.stderr)
