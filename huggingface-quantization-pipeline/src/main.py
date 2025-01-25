# This file is the entry point of the application. It initializes the pipeline, allows the user to select a model from Hugging Face, and orchestrates the quantization and evaluation processes.

import yaml
from utils.helpers import load_model, generate_report
from quantization.quantize import Quantizer
from evaluation.memory_usage import measure_memory_usage
from evaluation.inference_speed import measure_inference_speed
from evaluation.accuracy import evaluate_accuracy

def main():
    # Load configuration
    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_name = config['model_name']
    quantization_technique = config['quantization_technique']

    # Load the model
    model = load_model(model_name)

    # Initialize the quantizer
    quantizer = Quantizer(model)

    # Apply the selected quantization technique
    if quantization_technique == 'dynamic':
        quantizer.apply_dynamic_quantization()
    elif quantization_technique == 'static':
        quantizer.apply_static_quantization()
    elif quantization_technique == 'qat':
        quantizer.apply_quantization_aware_training()
    else:
        raise ValueError("Invalid quantization technique selected.")

    # Evaluate the quantized model
    memory_usage = measure_memory_usage(model)
    inference_speed = measure_inference_speed(model)
    accuracy = evaluate_accuracy(model)

    # Generate and save the report
    report = {
        'Memory Usage': memory_usage,
        'Inference Speed': inference_speed,
        'Accuracy': accuracy
    }
    generate_report(report)

if __name__ == "__main__":
    main()