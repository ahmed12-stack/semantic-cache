import re #(import re loads Python's built-in regular expressions 
#tool,used for finding patterns in text (here,splitting a sentence into words)
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

#cache

saved_questions = [
    "How do I reset my password?",
    "What is the capital of France?",
    "How can I change my email address?",
    "What are your opening hours?",
]
saved_answers = [
    "Click 'Forgot password' on the login page.",
    "Paris.",
    "Go to Settings > Account > Email.",
    "We are open 9 AM to 6 PM, Monday to Saturday.",
]
saved_vectors = model.encode(saved_questions)

#cutoff

SCORE_CUTOFF = 0.70
MARGIN_CUTOFF = 0.15

#finding important words

def import_words(text):
    """Names and numbers: capitalised words (not the first word) and digits"""
    words = re.findall(r"[A-Za-z0-9]+",text)
    important = set()#empty set, like a list no duplicates 
    for i, w in enumerate(words):#enumerate gives both the position i (0, 1, 2…) and the word w
        if w.isdigit() or (i > 0 and w[0].isupper() and w != "I"):
            important.add(w.lower())
    return important

#main logic
def check_cache(new_question):
    new_vector = model.encode(new_question)
    scores = util.cos_sim(new_vector, saved_vectors)[0].tolist()
   
    order = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)
    best, second = order[0], order[1]
    best_score = scores[best]
    margin = best_score - scores[second]

    if best_score < SCORE_CUTOFF:
        return None, f"MISS score {best_score:.2f} below {SCORE_CUTOFF}"
    if margin < MARGIN_CUTOFF:
        return None, f"MISS too close to call (gap {margin: .2f})"
    missing = import_words(new_question) - import_words(saved_questions[best])
    if missing:
        return None, f"MISS new important words {missing}"
    
    return saved_answers[best], f"HIT   '{saved_questions[best]}' (score {best_score:.2f}, gap {margin:.2f})"


tests = [
    "I forgot my password, how do I make a new one?",
    "When do you open?",
    "How do I reset my email password?",
    "What is the capital of Germany?",
    "What are your opening hours on Sunday?",
    "How do I bake a cake?",
    "I don't want to reset my password",
    "When are you closed?",
    "How do I remove my email address?",
]

for q in tests:
    answer, reason = check_cache(q)
    print(f"{q}\n   -> {reason}\n   -> answer: {answer}\n")