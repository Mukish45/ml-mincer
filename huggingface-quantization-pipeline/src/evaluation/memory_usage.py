def measure_memory_usage(model):
    import torch

    # Move the model to the appropriate device
    device = next(model.parameters()).device
    model.eval()

    # Measure memory usage
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input shape
        model(input_tensor)

    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB

    return {
        "memory_allocated_MB": memory_allocated,
        "memory_reserved_MB": memory_reserved
    }