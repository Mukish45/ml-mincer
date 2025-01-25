def apply_static_quantization(model, calibration_data):
    import torch
    from torch.quantization import quantize_dynamic, prepare, convert

    # Prepare the model for static quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepare(model, inplace=True)

    # Calibrate the model with the calibration data
    with torch.no_grad():
        for data in calibration_data:
            model(data)

    # Convert the model to a quantized version
    quantized_model = convert(model, inplace=False)

    return quantized_model