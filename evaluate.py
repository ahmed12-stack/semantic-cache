from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

#loading data and taking a random sample of 2000 pairs

data = load_dataset("nyu-mll/glue", "qqp", split = "validation")
data = data.shuffle(seed=42).select(range(2000))

q1 = data["question1"]
q2 = data["question2"]
labels = data["label"]

# Turning all questions into numbers (in batches, much faster)

v1 = model.encode(q1, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
v2 = model.encode(q2, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

# similarity of each pair: q1[0] vs q2[0], q1[1] vs q2[1], ...
scores = util.pairwise_cos_sim(v1, v2).tolist()

#counting the four outcomes for one cutoff
def evaluate(cutoff):
    true_hit = false_hit = true_miss = false_miss = 0
    for score, label in zip(scores, labels):
        says_hit = score >= cutoff
        if says_hit and label == 1:
            true_hit += 1
        elif says_hit and label == 0:
            false_hit += 1
        elif not says_hit and label == 0:
            true_miss += 1
        else:
            false_miss += 1
    hits = true_hit +false_hit
    wrong_hit_rate = false_hit / hits if hits > 0 else 0
    duplicates = true_hit + false_miss
    missed_rate = false_miss / duplicates if duplicates > 0 else 0
    return hits, wrong_hit_rate, missed_rate

#trying many cutoff and priting table

print("\n cutoff hits wrong-hit % missed-duplicate %")
for cutoff in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
    hits, wrong, missed = evaluate(cutoff)
    print(f"{cutoff:.2f}  {hits:5d}  {wrong*100:5.1f}%   {missed*100:5.1f}%")