import torch
import torch.nn.functional as F
import re
import unicodedata
import contractions


def clean_text(text):

    combined = f"{text}"
    combined = contractions.fix(combined)
    combined = unicodedata.normalize('NFD', combined)
    combined = ''.join(ch for ch in combined if unicodedata.category(ch) != 'Mn')
    combined = re.sub(r'https?://t\.co/\S+|pic\.twitter\.com/\S+', ' link_twitter ', combined)
    combined = re.sub(r'https?://\S+|www\.\S+', ' link ', combined)
    combined = re.sub(r'©.*$', ' ', combined, flags=re.MULTILINE)
    combined = re.sub(r'#\w+', ' ', combined)
    combined = re.sub(r'@\w+', ' ', combined)
    combined = ''.join(ch for ch in combined if ch.encode('ascii', 'ignore').decode('ascii') == ch)
    combined = re.sub(r'All rights reserved.*$', ' ', combined, flags=re.IGNORECASE|re.MULTILINE)
    combined = re.sub(r'[^A-Za-z\s]', ' ', combined)
    combined = combined.lower()

    return re.sub(r'\s+', ' ', combined).strip()


def BIAS(texto,tokenizador,modelo,encoder):

    full_text = clean_text(texto)

    inputs = tokenizador(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits, dim=1).squeeze().tolist()

    classes = encoder.classes_
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_label = classes[pred_idx]

    result = {
        "prediction": pred_label,
        "probabilities": {cls: round(probs[i], 4) for i, cls in enumerate(classes)}
    }
    
    return result