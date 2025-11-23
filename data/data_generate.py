import json
import random
import os
import re
from faker import Faker
from num2words import num2words

# 1. Setup & Configuration
# We use the Indian English locale for names, cities, and phone formats
fake = Faker('en_IN')
Faker.seed(18)
random.seed(18)

# Create the data directory if it doesn't exist [cite: 21]
os.makedirs('data', exist_ok=True)

# ---------------------------------------------------------
# NOISE FUNCTIONS (Simulating STT errors)
# ---------------------------------------------------------
def apply_stt_noise(text, label=None):
    """
    Transforms clean text into noisy STT text:
    - Lowercase
    - Spells out numbers (critical for CREDIT_CARD, PHONE, DATE)
    - Replaces symbols (@ -> ' at ', . -> ' dot ')
    - Removes punctuation
    """
    # 1. Basic normalization
    text = text.lower()
    
    # 2. Handle Symbols (Email/URL specific) [cite: 46]
    if label == "EMAIL":
        text = text.replace("@", " at ").replace(".", " dot ")
        
    # 3. Handle Digits (The most important STT feature) [cite: 46]
    # STT often converts "4242" to "four two four two" or "10" to "ten"
    if label in ["CREDIT_CARD", "PHONE", "DATE"] or any(c.isdigit() for c in text):
        # Specific fix for Indian mobile codes
        text = text.replace("+91", "plus nine one")
        
        words = []
        for char in text:
            if char.isdigit():
                # 80% chance to spell out the digit (noisy), 20% keep as digit
                if random.random() < 0.8:
                    words.append(num2words(int(char)))
                else:
                    words.append(char)
            else:
                words.append(char)
        text = " ".join(words)

    # 4. Remove Punctuation (Standard STT output) [cite: 46]
    # We remove commas, hyphens, periods (unless converted to 'dot' above)
    text = re.sub(r"[.,\-\(\)]", " ", text)
    
    # 5. Collapse multiple spaces
    return " ".join(text.split())

# ---------------------------------------------------------
# DATA GENERATION LOGIC
# ---------------------------------------------------------
def get_entity_template():
    """
    Returns a tuple: (clean_prefix, clean_entity_value, label)
    We separate prefix and entity to calculate offsets accurately after noise.
    """
    # Templates for Indian Context
    choices = [
        ("my credit card number is ", fake.credit_card_number(), "CREDIT_CARD"),
        ("call me on ", fake.phone_number(), "PHONE"),
        ("mobile number is ", fake.phone_number(), "PHONE"),
        ("my email id is ", fake.email(), "EMAIL"),
        ("send mail to ", fake.email(), "EMAIL"),
        ("my name is ", fake.name(), "PERSON_NAME"),
        ("this is ", fake.first_name(), "PERSON_NAME"),
        ("i was born on ", fake.date(), "DATE"),
        ("the date is ", fake.date(), "DATE"),
        ("i live in ", fake.city(), "CITY"),       # PII=False [cite: 8]
        ("office is in ", fake.city(), "CITY"),    # PII=False
        ("traveling to ", fake.state(), "LOCATION") # PII=False
    ]
    return random.choice(choices)

def generate_labeled_entry(id_counter):
    """Generates a single training example with correct character offsets."""
    prefix, entity_val, label = get_entity_template()
    
    # Apply noise separately to prefix and entity to track where the entity starts
    noisy_prefix = apply_stt_noise(prefix)
    noisy_entity = apply_stt_noise(entity_val, label)
    
    # Construct final sentence
    full_text = f"{noisy_prefix} {noisy_entity}".strip()
    
    # Calculate Offsets
    # The entity starts after the prefix + 1 space (if prefix exists)
    start_index = len(noisy_prefix) + 1 if noisy_prefix else 0
    end_index = start_index + len(noisy_entity)
    
    return {
        "id": f"utt_{id_counter:04d}",
        "text": full_text,
        "entities": [
            {
                "start": start_index,
                "end": end_index,
                "label": label
            }
        ]
    }

def generate_test_entry(id_counter):
    """Generates a test example (NO labels, higher complexity)."""
    # For test data, we sometimes combine two sentences to make it harder
    if random.random() < 0.3:
        # Complex Case: "My name is X and my email is Y"
        p1, e1, l1 = get_entity_template()
        p2, e2, l2 = get_entity_template()
        
        # Noise the whole thing at once since we don't need offsets
        clean_text = f"{p1}{e1} and {p2}{e2}"
        noisy_text = apply_stt_noise(clean_text)
    else:
        # Simple Case
        p, e, l = get_entity_template()
        noisy_text = apply_stt_noise(f"{p}{e}", l)
        
    return {
        "id": f"utt_{id_counter:04d}",
        "text": noisy_text
        # No "entities" field in test set [cite: 25]
    }

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    # 1. Generate Train (1000 examples) [cite: 13, 23]
    print("Generating data/train.jsonl (1000 examples)...")
    with open('data/train.jsonl', 'w') as f:
        for i in range(1000):
            entry = generate_labeled_entry(i)
            json.dump(entry, f)
            f.write('\n')

    # 2. Generate Dev (200 examples) [cite: 13, 24]
    print("Generating data/dev.jsonl (200 examples)...")
    with open('data/dev.jsonl', 'w') as f:
        for i in range(200):
            entry = generate_labeled_entry(1000 + i) # ID continues from train
            json.dump(entry, f)
            f.write('\n')

    # 3. Generate Test (200 examples, Unlabeled) [cite: 25]
    print("Generating data/test.jsonl (200 examples)...")
    with open('data/test.jsonl', 'w') as f:
        for i in range(200):
            entry = generate_test_entry(2000 + i) # ID block 2000+
            json.dump(entry, f)
            f.write('\n')
            
    print("\n✅ Success! All datasets created in 'data/' folder.")
    print("Sample Train Entry:")
    with open('data/train.jsonl') as f:
        print(f.readline().strip())

if __name__ == "__main__":
    main()