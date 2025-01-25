# Hugging Face Quantization Pipeline

This project provides a pipeline for applying various quantization techniques to models from Hugging Face. Users can select a model, apply different quantization methods, and evaluate the results based on memory usage, inference speed, and accuracy.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Quantization Techniques](#quantization-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd huggingface-quantization-pipeline
pip install -r requirements.txt
```

## Usage

1. Run the main application:

```bash
python src/main.py
```

2. Follow the prompts to select a model from Hugging Face and choose the desired quantization techniques.

3. The pipeline will execute the quantization process and generate a report with the results.

## Quantization Techniques

The following quantization techniques are implemented:

- **Dynamic Quantization**: Reduces the model size and improves inference speed without requiring a full retraining.
- **Static Quantization**: Involves calibrating the model with a representative dataset to optimize performance.
- **Quantization-Aware Training**: Prepares the model for quantization during training, allowing for better accuracy retention.

## Evaluation Metrics

The evaluation process includes measuring:

- **Memory Usage**: The amount of memory consumed by the quantized model.
- **Inference Speed**: The time taken for the model to make predictions.
- **Accuracy**: The performance of the quantized model on a validation dataset.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.