class Quantizer:
    def __init__(self, model):
        self.model = model

    def apply_dynamic_quantization(self):
        from .techniques.dynamic_quantization import apply_dynamic_quantization
        return apply_dynamic_quantization(self.model)

    def apply_static_quantization(self):
        from .techniques.static_quantization import apply_static_quantization
        return apply_static_quantization(self.model)

    def apply_quantization_aware_training(self):
        from .techniques.quantization_aware_training import apply_quantization_aware_training
        return apply_quantization_aware_training(self.model)