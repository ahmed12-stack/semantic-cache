from datasets import load_dataset

data = load_dataset("nyu-mll/glue", "qqp", split="validation")

print(data)
print()

for i in range(5):
    row = data[i]
    print(f"Q1: {row['question1']}")
    print(f"Q2: {row['question2']}")
    print(f"Same meaning? {row['label']}")
    print()    