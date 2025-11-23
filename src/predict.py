import json
import argparse
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii

# --- 1. VALIDATION LOGIC (For High Precision) ---
def validate_span(text, label):
    """
    Returns True if the span 'looks' like the label.
    Returns False if it's likely a false positive.
    """
    text = text.lower()
    
    # PHONE: Must contain digits or number-words (one, two, etc.)
    if label == "PHONE":
        if re.search(r'\d|one|two|three|four|five|six|seven|eight|nine|zero|plus', text):
            return True
        return False
        
    # EMAIL: Must imply structure (at, dot, @)
    if label == "EMAIL":
        if " at " in text or "@" in text or " dot " in text:
            return True
        return False
    
    # CREDIT_CARD: Must contain digits or number-words
    if label == "CREDIT_CARD":
        if re.search(r'\d|one|two|three|four|five|six|seven|eight|nine|zero', text):
            return True
        return False
        
    # DATE: Very basic check, usually safe to trust model or add checks for months/days
    if label == "DATE":
        return True

    # For Names/Cities, we trust the model (Regex is too hard for names)
    return True

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0: continue # Skip special tokens
            
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=128) # Reduced to 128 for speed
    ap.add_argument("--device", default="cpu") 
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # --- 2. QUANTIZATION (For Low Latency) ---
    # Compresses model to int8 for CPU speedup
    if args.device == "cpu":
        # print("Quantizing model for CPU inference...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            raw_spans = bio_to_spans(text, offsets, pred_ids)
            
            # --- 3. FILTERING ---
            ents = []
            for s, e, lab in raw_spans:
                span_text = text[s:e]
                # Only keep if it passes validation
                if validate_span(span_text, lab):
                    ents.append(
                        {
                            "start": int(s),
                            "end": int(e),
                            "label": lab,
                            "pii": bool(label_is_pii(lab)),
                        }
                    )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()