def apply_quantization_aware_training(model, training_data, num_epochs=3):
    from torch.quantization import prepare_qat, convert
    import torch

    # Prepare the model for quantization-aware training
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    prepare_qat(model, inplace=True)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        for data, target in training_data:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    # Convert the model to a quantized version
    model.eval()
    convert(model, inplace=True)

    return model