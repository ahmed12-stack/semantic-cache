from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

saved_questions = [
    "How do I reset my password?",
    "What is the capital of France?",
    "How can I change my email address?",
    "What are your opening hours?",
]
new_question = "What is the capital of Germany?"

saved_vectors = model.encode(saved_questions)
new_vector = model.encode(new_question)

scores = util.cos_sim(new_vector, saved_vectors)[0]

for question, score in zip(saved_questions, scores):
    print(f"{score: .2f} {question}")