import torch

def FAKE(text,tokenizer,model):

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
        )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

    classes = ["real", "fake"]
    pred_idx = int(probs.argmax())
    prediction = classes[pred_idx]

    result = {
        "prediction": prediction,
        "probabilities": {"real": float(probs[0]),"fake": float(probs[1])}
        }
    
    return result
