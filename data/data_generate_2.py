import json
import random
import os
import re
from faker import Faker
from num2words import num2words

# 1. Setup
fake = Faker('en_IN')
Faker.seed(18)
random.seed(18)

os.makedirs('data', exist_ok=True)

# --- NOISE CONFIGURATION ---
FILLERS = ["um", "uh", "like", "actually", "wait", "i mean", "sorry"]
INTROS = [
    "can you note that", "please write down", "i said", "the details are", 
    "it is", "check this", "verify", "identifying as", "record shows",
    "kindly update", "change it to"
]

HOMOPHONES = {
    "mail": "male", "to": "too", "two": "to",
    "for": "four", "eight": "ate", "i": "eye",
    "see": "sea", "meet": "meat", "know": "no",
    "right": "write", "buy": "bye", "hi": "high"
}

# --- NOISE FUNCTIONS ---

def apply_homophone_noise(text):
    """Swaps words for their acoustic homophones."""
    words = text.split()
    new_words = []
    for word in words:
        # 20% chance to swap if a homophone exists
        if word in HOMOPHONES and random.random() < 0.2:
            new_words.append(HOMOPHONES[word])
        else:
            new_words.append(word)
    return " ".join(new_words)

def apply_indian_phonetic_noise(text):
    """Simulates common Indian English pronunciation swaps in STT."""
    words = text.split()
    new_words = []
    for word in words:
        # Only apply noise to words longer than 2 chars to avoid destroying "is", "at"
        if len(word) < 3:
            new_words.append(word)
            continue
            
        # 1. The V/W Swap ("video" -> "wideo")
        if "v" in word and random.random() < 0.15:
            word = word.replace("v", "w")
        elif "w" in word and random.random() < 0.15:
            word = word.replace("w", "v")
            
        # 2. The S/SH Confusion ("sheet" -> "sheet", "same" -> "shame")
        if "sh" in word and random.random() < 0.15:
            word = word.replace("sh", "s")
        elif "s" in word and random.random() < 0.15:
            word = word.replace("s", "sh")
            
        # 3. The F/PH Swap ("photo" -> "foto")
        if "ph" in word and random.random() < 0.2:
            word = word.replace("ph", "f")
            
        # 4. Z/J Swap ("zero" -> "jero")
        if "z" in word and random.random() < 0.15:
            word = word.replace("z", "j")
            
        new_words.append(word)
    return " ".join(new_words)

def apply_stt_noise(text, label=None):
    """Applies the full stack of STT noises."""
    text = text.lower()
    
    # 1. Insert random fillers (hesitation)
    # Don't add fillers inside specific entities (like inside a phone number)
    if label is None and len(text.split()) > 3 and random.random() < 0.25:
        words = text.split()
        insert_idx = random.randint(1, len(words)-1)
        words.insert(insert_idx, random.choice(FILLERS))
        text = " ".join(words)
        
    # 2. Phonetic & Homophone Noise
    text = apply_indian_phonetic_noise(text)
    text = apply_homophone_noise(text)
    
    # 3. Domain Specific Noise (Email symbols)
    if label == "EMAIL":
        text = text.replace("@", " at ").replace(".", " dot ")
        
    # 4. Digit Spelling (Critical for STT)
    # We spell out digits for CARD, PHONE, DATE or if digits exist in text
    if label in ["CREDIT_CARD", "PHONE", "DATE"] or any(c.isdigit() for c in text):
        text = text.replace("+91", "plus nine one")
        words = []
        for char in text:
            if char.isdigit():
                # 70% chance to spell digit (noisy), 30% keep digit (easy)
                if random.random() < 0.7:
                    words.append(num2words(int(char)))
                else:
                    words.append(char)
            else:
                words.append(char)
        text = " ".join(words)

    # 5. Punctuation Removal
    text = re.sub(r"[.,\-\(\)]", " ", text)
    
    return " ".join(text.split())

# --- DATA GENERATOR ---

def get_base_data():
    """Generates raw (Prefix, Entity, Suffix, Label) tuples."""
    label = random.choice(["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY"])
    
    if label == "CREDIT_CARD": entity = fake.credit_card_number()
    elif label == "PHONE": entity = fake.phone_number()
    elif label == "EMAIL": entity = fake.email()
    elif label == "PERSON_NAME": entity = fake.name()
    elif label == "DATE": entity = fake.date()
    elif label == "CITY": entity = fake.city()

    # Create Context (Prefix/Suffix)
    prefix = ""
    suffix = ""
    
    structure_roll = random.random()
    
    # Type A: Prefix + Entity (Most common: "My email is ...")
    if structure_roll < 0.6:
        prefix = random.choice([
            f"my {label.lower().replace('_',' ')} is", 
            f"{random.choice(INTROS)}", 
            "the value is", "it is", "update"
        ])
    # Type B: Entity + Suffix ("... is my number")
    elif structure_roll < 0.8:
        suffix = random.choice([
            "is the correct one", "is my details", "please verify", 
            "is what i said", "right"
        ])
    # Type C: Embedded ("so [Entity] is fine")
    else:
        prefix = random.choice(["so", "and", "well", "okay"])
        suffix = random.choice(["is fine", "looks good", "correct?"])

    return prefix, entity, suffix, label

def generate_entry(id_counter, is_test=False):
    prefix, entity, suffix, label = get_base_data()
    
    # Apply Noise Component-by-Component
    # This ensures we know exactly how long the entity becomes after noise
    
    n_prefix = apply_stt_noise(prefix)
    n_entity = apply_stt_noise(entity, label)
    n_suffix = apply_stt_noise(suffix)
    
    # Construct Full Text
    # Handle spacing carefully. If prefix is empty, don't add leading space.
    parts = []
    if n_prefix: parts.append(n_prefix)
    parts.append(n_entity)
    if n_suffix: parts.append(n_suffix)
    
    full_text = " ".join(parts)
    
    # Calculate Offsets
    # The entity starts after the prefix (plus a space if prefix exists)
    start = len(n_prefix) + 1 if n_prefix else 0
    end = start + len(n_entity)
    
    entry = {
        "id": f"utt_{id_counter:04d}",
        "text": full_text
    }
    
    if not is_test:
        entry["entities"] = [{
            "start": start,
            "end": end,
            "label": label
        }]
        
    return entry

# --- MAIN EXECUTION ---

def main():
    print("Generating robust dataset with Phonetic & Homophone noise...")
    
    # 1. Train (1200 examples)
    with open('data/train.jsonl', 'w') as f:
        for i in range(1200):
            f.write(json.dumps(generate_entry(i)) + "\n")
            
    # 2. Dev (200 examples)
    with open('data/dev.jsonl', 'w') as f:
        for i in range(200):
            f.write(json.dumps(generate_entry(1200+i)) + "\n")
            
    # 3. Test (200 examples, No Labels)
    with open('data/test.jsonl', 'w') as f:
        for i in range(200):
            # Test data can be slightly more complex (combining 2 sentences)
            if random.random() < 0.3:
                e1 = generate_entry(2000+i, is_test=True)
                e2 = generate_entry(3000+i, is_test=True)
                combined_text = f"{e1['text']} and {e2['text']}"
                f.write(json.dumps({"id": f"utt_test_{i:04d}", "text": combined_text}) + "\n")
            else:
                f.write(json.dumps(generate_entry(2000+i, is_test=True)) + "\n")

    print("✅ Done! Files saved to data/train.jsonl, data/dev.jsonl, data/test.jsonl")

if __name__ == "__main__":
    main()