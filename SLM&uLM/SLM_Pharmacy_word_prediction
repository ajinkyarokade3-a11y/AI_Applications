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
# 2. Preprocessing
# --------------------------------------------------
text = corpus.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
words = text.split()

# --------------------------------------------------
# 3. Create Trigram Model
# --------------------------------------------------
trigram_model = defaultdict(Counter)
for i in range(len(words) - 2):
    word1 = words[i]
    word2 = words[i + 1]
    word3 = words[i + 2]
    context = (word1, word2)
    trigram_model[context][word3] += 1

# --------------------------------------------------
# 4. Next Word Prediction
# --------------------------------------------------
def predict_next_word(word1, word2):
    context = (word1.lower(), word2.lower())
    if context not in trigram_model:
        return "No prediction available"
    prediction = trigram_model[context].most_common(1)[0][0]
    return prediction

# --------------------------------------------------
# 5. Text Generation
# --------------------------------------------------
def generate_text(word1, word2, length=10):
    generated = [word1.lower(), word2.lower()]
    for i in range(length - 2):
        context = (generated[-2], generated[-1])
        if context not in trigram_model:
            break
        next_word = trigram_model[context].most_common(1)[0][0]
        generated.append(next_word)
    return " ".join(generated)

# --------------------------------------------------
# 6. User Input
# --------------------------------------------------
word1 = input("\nEnter first context word: ")
word2 = input("Enter second context word: ")

# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------
prediction = predict_next_word(word1, word2)
print("\nPredicted Next Word:", prediction)

# --------------------------------------------------
# 8. Generate Text
# --------------------------------------------------
generated_text = generate_text(word1, word2, 10)
print("\nGenerated Text:")
print(generated_text)
