import re
from collections import defaultdict, Counter

# --------------------------------------------------
# 1. Training Corpus
# --------------------------------------------------
corpus = """
Hospital pharmacy manages extensive inventories of drugs and prescriptions.
Take two capsules daily with water after meals.
Dispense fifty tablets of paracetamol five hundred milligrams.
Patient refill requests require checking dosage instructions and rules.
Take twice daily for seven days.
Five hundred milligrams of amoxicillin capsules are dispensed.
Refill requests are processed by pharmacy inventory terminals.
Hospital dispensaries manage drug instructions and patient refills.
Take one tablet daily in the morning.
Dispense thirty capsules of ibuprofen.
"""

# --------------------------------------------------
# 2. Text Preprocessing
# --------------------------------------------------
text = corpus.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
words = text.split()
print("Number of Words:", len(words))

# --------------------------------------------------
# 3. Create Bigram Counts
# --------------------------------------------------
bigram_counts = defaultdict(Counter)
for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    bigram_counts[current_word][next_word] += 1

# --------------------------------------------------
# 4. Next Word Prediction
# --------------------------------------------------
def predict_next_word(word):
    word = word.lower()
    if word not in bigram_counts:
        return "No prediction available"
    next_words = bigram_counts[word]
    predicted_word = next_words.most_common(1)[0][0]
    return predicted_word

# --------------------------------------------------
# 5. Generate Text
# --------------------------------------------------
def generate_text(start_word, length=10):
    current_word = start_word.lower()
    generated = [current_word]
    for i in range(length - 1):
        if current_word not in bigram_counts:
            break
        next_word = bigram_counts[current_word].most_common(1)[0][0]
        generated.append(next_word)
        current_word = next_word
    return " ".join(generated)

# --------------------------------------------------
# 6. Display Vocabulary
# --------------------------------------------------
vocabulary = sorted(set(words))
print("\nVocabulary:")
print(vocabulary)

# --------------------------------------------------
# 7. User Input
# --------------------------------------------------
input_word = input("\nEnter a word for next-word prediction: ")
prediction = predict_next_word(input_word)
print("\nPredicted Next Word:", prediction)

# --------------------------------------------------
# 8. Generate Text
# --------------------------------------------------
generated = generate_text(input_word, 10)
print("\nGenerated Text:")
print(generated)
