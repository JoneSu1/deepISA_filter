from deepISA.utils import one_hot_encode
from deepISA.score.utils_isa import ablate_motifs
import numpy as np


def test_one_hot_encode():
    seqs = ["ACGT", "NNNN"]
    out = one_hot_encode(seqs)
    assert out.shape == (2, 4, 4)
    # Check A (first channel, first pos)
    assert out[0, 0, 0] == 1.0
    # Check N (all zeros)
    assert np.all(out[1] == 0)

def test_ablate_motifs():
    seq = "ATGCATGC"
    # [start, end) semantics: ablating [2, 3) replaces the single base at 2
    ablated = ablate_motifs(seq, [2], [3])
    assert ablated == "ATNCATGC"
    # Ablating [2, 4) replaces both G and C
    ablated2 = ablate_motifs(seq, [2], [4])
    assert ablated2 == "ATNNATGC"