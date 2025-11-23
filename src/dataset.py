import json
from typing import List, Dict, Any
from torch.utils.data import Dataset

class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, label_list: List[str], max_length: int = 128, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.is_train = is_train

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])

                # 1. Create Character-Level Tags
                char_tags = ["O"] * len(text)
                for e in entities:
                    s, e_idx, lab = e["start"], e["end"], e["label"]
                    # Safety check for bad indices
                    if s < 0 or e_idx > len(text) or s >= e_idx:
                        continue
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e_idx):
                        char_tags[i] = f"I-{lab}"

                # 2. Tokenize with Offsets
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    padding="max_length", # PAD standardizes batch shapes
                    max_length=self.max_length,
                    return_tensors=None # Return lists, not tensors here
                )
                
                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                # 3. Align Tokens to Labels
                bio_tags = []
                for idx, (start, end) in enumerate(offsets):
                    # Special tokens (CLS, SEP, PAD) usually have (0,0) offset
                    # OR they track to valid text but we should ignore them.
                    # We check if it's a special token ID
                    if start == end or input_ids[idx] in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                         # Label for special tokens is -100 (ignored by PyTorch Loss)
                        bio_tags.append(-100)
                    elif start < len(char_tags):
                        # Standard Logic: Take label of the start char
                        tag_str = char_tags[start]
                        bio_tags.append(self.label2id.get(tag_str, self.label2id["O"]))
                    else:
                        bio_tags.append(self.label2id["O"])

                self.items.append(
                    {
                        "id": obj["id"],
                        "text": text,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": bio_tags, # Direct ID list
                        "offset_mapping": offsets,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

# --- MISSING FUNCTION ADDED BELOW ---

def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    # Since we used padding="max_length" in the dataset, 
    # all items are already the same length (max_length).
    # We just need to stack them. However, sticking to the dynamic padding logic 
    # is safer if you ever change max_length settings.
    
    max_len = max(len(ids) for ids in input_ids_list)

    def pad(seq, pad_value, max_len):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id, max_len) for ids in input_ids_list]
    attention_mask = [pad(am, 0, max_len) for am in attention_list]
    labels = [pad(lab, label_pad_id, max_len) for lab in labels_list]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
    return out