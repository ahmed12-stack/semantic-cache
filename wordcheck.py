import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

data = load_dataset("nyu-mll/glue", "qqp", split="validation")
data = data.shuffle(seed=42).select(range(2000))
q1, q2, labels = data["question1"], data["question2"], data["label"]

v1 = model.encode(q1, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
v2 = model.encode(q2, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
scores = util.pairwise_cos_sim(v1, v2).tolist()

def important_words(text):
    words = re.findall(r"[A-Za-z0-9]+", text)
    important = set()
    for i, w in enumerate(words):
        if w.isdigit() or (i > 0 and w[0].isupper() and w != "I"):
            important.add(w.lower())
    return important

words_match = [important_words(a) == important_words(b) for a, b in zip(q1, q2)]#list comprehension, a short way to build a list. It means "for each pair (a, b), check whether the two sets of important words are exactly equal, and collect all the True/False answers into a list."

def evaluate(cutoff, use_word_check):
    true_hit = false_hit = true_miss = false_miss = 0
    for score, label, same_words in zip(scores, labels, words_match):
        says_hit = score >= cutoff and (same_words or not use_word_check)
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

print("\ncutoff |  SCORE ONLY: hits  wrong%  missed%  |  SCORE + WORDS: hits  wrong%  missed%")
for cutoff in [0.60, 0.70, 0.80, 0.85, 0.90]:
    a = evaluate(cutoff, False)
    b = evaluate(cutoff, True)
    print(f"{cutoff:.2f}   |  {a[0]:5d}  {a[1]*100:5.1f}%  {a[2]*100:5.1f}%     |  "
          f"{b[0]:5d}  {b[1]*100:5.1f}%  {b[2]*100:5.1f}%")