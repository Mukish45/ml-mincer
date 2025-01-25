def measure_inference_speed(model, input_data, num_runs=100):
    import time
    import torch

    # Warm up the model
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # Measure inference speed
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_data)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time