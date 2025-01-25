def load_model(model_name):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_data(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    return inputs

def generate_report(memory_usage, inference_speed, accuracy):
    report = {
        "Memory Usage (MB)": memory_usage,
        "Inference Speed (ms)": inference_speed,
        "Accuracy (%)": accuracy
    }
    return report