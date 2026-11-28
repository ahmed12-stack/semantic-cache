from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util

model = SentenceTransformer("all-MiniLM-L6-v2")
checker = CrossEncoder("cross-encoder/stsb-distilroberta-base")

data = load_dataset("nyu-mll/glue", "qqp", split="validation")
data = data.shuffle(seed=42).select(range(2000))
q1, q2, labels = data["question1"], data["question2"], data["label"]

v1 = model.encode(q1, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
v2 = model.encode(q2, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
scores = util.pairwise_cos_sim(v1, v2).tolist()

#the first filter

FIRST_CUTOFF = 0.60

candidates = [i for i, s in enumerate(scores) if s >= FIRST_CUTOFF]
print(f"{len(candidates)} of {len(scores)} pairs passed the fast filter")

#checks on candidates

pairs = [(q1[i], q2[i]) for i in candidates]
check_scores = checker.predict(pairs, batch_size=32, show_progress_bar=True)


careful = [0.0] * len(scores)
for i, s in zip(candidates, check_scores):
    careful[i] = float(s)

def evaluate(decisions):
    true_hit = false_hit = true_miss = false_miss = 0
    for says_hit, label in zip(decisions, labels):
        if says_hit and label == 1:
            true_hit += 1
        elif says_hit and label == 0:
            false_hit += 1
        elif not says_hit and label == 0:
            true_miss += 1
        else:
            false_miss += 1
    hits = true_hit + false_hit
    wrong = false_hit / hits if hits > 0 else 0
    missed = false_miss / (true_hit + false_miss) if (true_hit + false_miss) > 0 else 0
    return hits, wrong, missed
#instead of taking a cutoff, it now takes a list of True/False decisions. That makes one function work for any method: 
# score only, word check, two-stage, or anything else you invent later.

def show(name, decisions):
    hits, wrong, missed = evaluate(decisions)
    print(f"{name:28s} hits {hits:5d}   wrong {wrong*100:5.1f}%   missed {missed*100:5.1f}%")

print("\n--- SCORE ONLY ---")
for c in [0.80, 0.85, 0.90]:
    show(f"score >= {c}", [s >= c for s in scores])

print("\n--- TWO-STAGE (score >= 0.60, then careful check) ---")
for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
    show(f"careful >= {c}", [s >= c for s in careful])