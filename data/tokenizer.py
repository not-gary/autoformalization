import argparse
import subprocess
import torch
import re

parser = argparse.ArgumentParser('LaTeX/Coq Tokenizer')
parser.add_argument('-s', '--source',
                    type=str,
                    default=None,
                    help='Path of source file.')

args = None
args, _ = parser.parse_known_args()

def detokenize_coq(content, subset="even-odd"):
    seq = content.split(' ')
    length = len(seq)

    coq = ""
    hypoth_scope = ""
    hypotheses = []
    for i in range(length):
        tok = seq[i]
        if re.match("<nat:.+", tok) or re.match("<var:.+", tok):
            tok = tok[5:-1]
            if subset == "composites" and tok == "N":
                tok = "N0"
        elif re.match("<def:.+", tok):
            tok = "placeholder_def"
        elif re.match("<genP:.+", tok):
            tok = "placeholder_thm"
        elif re.match("<genH>", tok):
            # assert case. Base name on following consts/vars.
            if i < length - 1 and seq[i + 1] == ":":
                if subset == "even-odd":
                    subseq = seq[i:min(i + 10, length)]
                elif subset == "composites":
                    subseq = seq[i:min(i + 3, length)]
                elif subset == "powers":
                    subseq = seq[i:min(i + 3, length)]
                elif subset == "poly":
                    subseq = seq[i:min(i + 10, length)]
                # Check if sum.
                if "+" in subseq:
                    tok = "Hsum"
                    hypotheses.insert(0, tok)
                    hypoth_scope = "Hsum"
                else:
                    coeff = ""
                    var = ""
                    for subtok in subseq:
                        if re.match("<nat:.+", subtok):
                            coeff = subtok[5:-1]
                            if subset in ["even-odd", "poly"]:
                                hypoth_scope = "H" + coeff if "H" + coeff in hypotheses else ""
                        elif re.match("<var:.+", subtok):
                            var = subtok[5:-1]
                    tok = "H" + coeff + var
                    if subset in ["even-odd", "poly"]:
                        hypotheses.insert(0, tok)
            # rewrite case. Base name on existing hypoths.
            elif i < length - 1 and seq[i + 1] == ".":
                if hypoth_scope == "Hsum":
                    if hypotheses[-1][-1].isalpha():
                        tok = hypotheses.pop()
                    else:
                        tok = hypotheses[-2]
                        hypotheses.remove(hypotheses[-2])

                    hypoth_scope = ""
                    for h in hypotheses:
                        if h[-1].isalpha() and h != "Hsum":
                           hypoth_scope = "Hsum" 
                elif hypoth_scope != "":
                    tok = hypoth_scope
                    hypotheses.remove(hypoth_scope)
                    hypoth_scope = ""
                else:
                    try:
                        tok = hypotheses.pop()
                    except:
                        tok = "<genH>"
        elif i < length - 2 and tok == "exists" and re.match("<nat:.+", seq[i + 1]):
            hypotheses.insert(0, "H" + seq[i + 1][5:-1])

        coq += tok
        dot = i < length - 1 and seq[i + 1] == "."
        lib = (tok == ".") and (i < length - 1 and (re.match(".*even.*", seq[i + 1]) or re.match(".*odd.*", seq[i + 1])))
        if not (dot or lib):
            coq += " "

    return coq

def check_proof(coq, tgt, subset=""):
    # Check that the proof is correct by calling coqtop.
    if subset == "poly":
        proc = subprocess.Popen(['coqtop', '-Q', 'plf/', 'PLF'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE, encoding='utf8')
    else:
        proc = subprocess.Popen(['coqtop'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE, encoding='utf8')
    try:
        valid_proof = ("No more subgoals." in proc.communicate(coq + "\n", timeout=600)[0])
    except:
        valid_proof = False

    # Check that the theorem is correct by direct comparison with target.
    seq = coq.split(' ')
    if "<PAD>" in seq:
        return False, False
    thm_start = seq.index("Theorem")
    thm_end = seq.index("Proof.")
    coq_thm = " ".join(seq[thm_start:thm_end])
    valid_theorem = (coq_thm == tgt)

    return valid_proof, valid_theorem

def validate(filename):
    thm_avgs = {}
    proof_avgs = {}
    total_avgs = {}

    file = open(filename, 'r')
    examples = file.read().split('\n\n')
    for example in examples:
        # Split up the example into name, trans., and target.
        seq = example.split()
        translation = " ".join(seq[seq.index('Translation:') + 1:seq.index('Target:')])
        # translation = " ".join(seq[seq.index("Target:") + 1:])
        target = seq[seq.index("Target:") + 1:]
        target = " ".join(target[target.index("Theorem"):target.index("Proof")])

        # Split by subset and length.
        if "even-odd" in seq[3]:
            subset = "even-odd_" + str(seq[3].count('_') - 1)
            detok_sub = "even-odd"
        elif "powers" in seq[3]:
            subset = "powers"
            detok_sub = "powers"
        elif "composites" in seq[3]:
            subset = "composites_" + seq[3].split('_')[-2]
            detok_sub = "composites"
        elif "poly" in seq[3]:
            subset = "poly_" + str(translation.count(r'%assertion'))
            detok_sub = "poly"

        coq = detokenize_coq(translation, detok_sub)
        tgt = detokenize_coq(target, detok_sub)[:-1]

        proof, thm = check_proof(coq, tgt, detok_sub)

        # Report rolling averages, stats.
        if not subset in thm_avgs:
            thm_avgs[subset] = [1.0 * thm]
            proof_avgs[subset] = [1.0 * proof]
            total_avgs[subset] = [1.0 * (thm and proof)]
        else:
            thm_avgs[subset].append(1.0 * thm)
            proof_avgs[subset].append(1.0 * proof)
            total_avgs[subset].append(1.0 * (thm and proof))

        print("\n" + seq[3])
        print("Translation " + seq[2] + "\tTheorem: " + str(thm) + "\tProof: " + str(proof))
        print("[" + subset + "]\tTheorem: " + str(torch.sum(torch.Tensor(thm_avgs[subset])) / len(thm_avgs[subset])) + "\tProof: " + str(torch.sum(torch.Tensor(proof_avgs[subset])) / len(proof_avgs[subset])) + "\tBoth: " + str(torch.sum(torch.Tensor(total_avgs[subset])) / len(total_avgs[subset])))

        # Print content of misses.
        if not (proof and thm):
            print("\n" + coq + "\n")

    print("\nFinal Stats by Length:")
    for key in sorted(thm_avgs):
        print("[" + key + "]\tTheorem: " + str(torch.sum(torch.Tensor(thm_avgs[key])) / len(thm_avgs[key])) + "\tProof: " + str(torch.sum(torch.Tensor(proof_avgs[key])) / len(proof_avgs[key])) + "\tBoth: " + str(torch.sum(torch.Tensor(total_avgs[key])) / len(total_avgs[key])))

if __name__ == "__main__":
    validate(args.source)