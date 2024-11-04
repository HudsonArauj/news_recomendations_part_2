from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModel.from_pretrained('distilgpt2')

tokenizer.pad_token = tokenizer.eos_token

def get_embedding(text):
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')
    return embeddings