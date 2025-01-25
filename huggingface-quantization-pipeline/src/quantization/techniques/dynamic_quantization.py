def apply_dynamic_quantization(model):
    """
    Applies dynamic quantization to the given model.

    Args:
        model: The model to be quantized.

    Returns:
        The quantized model.
    """
    import torch

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model