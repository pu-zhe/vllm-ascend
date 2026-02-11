import torch
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader import ShardedStateLoader


class ShardedStateLoader310(ShardedStateLoader):
    """
    A specialized sharded state loader for Ascend 310P platform.

    This class extends the base ShardedStateLoader to provide specific
    functionality for handling quantized models on the 310P platform,
    including compressed model saving and quantization-aware operations.
    """

    # Data types that are considered for quantization
    QUANTIZE_DTYPE_LIST = [torch.int8, torch.int32, torch.int64]

    def __init__(self, load_config: LoadConfig):
        """
        Initialize the ShardedStateLoader310 with the given load configuration.

        Args:
            load_config: Configuration for loading the model
        """
        super().__init__(load_config)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        """
        Save the model to the specified path, with special handling for W8A8S quantization.

        Args:
            model: The PyTorch model to save
            path: Directory path where the model will be saved
            pattern: Filename pattern for sharded checkpoints
            max_size: Maximum shard size in bytes
        """
        quantize_type = model.quant_config.quant_description.get("model_quant_type", "FLOAT")
        if quantize_type == "W8A8S":
            ShardedStateLoader310.save_model_compress(model, path, pattern=pattern)
        else:
            super().save_model(model, path, pattern=pattern, max_size=max_size)

    @staticmethod
    def save_model_compress(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
    ) -> None:
        """
        Save the model using compression techniques specific to 310P platform.

        This method applies pseudo-sparse compression and exports the model
        in safetensors format with quantization information.

        Args:
            model: The PyTorch model to save
            path: Directory path where the model will be saved
            pattern: Filename pattern for the checkpoint file
        """
        if pattern is None:
            pattern = ShardedStateLoader310.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        quant_model_description = ShardedStateLoader310.generate_quant_model_description(model)
        state_dict = ShardedStateLoader310._filter_subtensors(model.state_dict())
        compress_config = CompressConfig(
            do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=path, multiprocess_num=2
        )
        compressor = Compressor(compress_config, weight=state_dict, quant_model_description=quant_model_description)
        compressor.run()
        filename = pattern.format(rank=rank, part=part_idx)
        compressor.export_safetensors(path, safetensors_name=filename)

    @staticmethod
    def generate_module_type_map(model: torch.nn.Module):
        """
        Generate a mapping of parameter names to their corresponding module types.

        This method creates a dictionary that maps each parameter name to the
        type of the module it belongs to, which is useful for identifying
        quantizable layers.

        Args:
            model: The PyTorch model to analyze

        Returns:
            A dictionary mapping parameter names to module type names
        """
        module_type_map = {}
        module_dict = dict(model.named_modules())
        state_dict = model.state_dict()
        for name in state_dict:
            module_path = name.rsplit(".", 1)[0]
            module = module_dict.get(module_path)
            if module:
                module_type_map[name] = type(module).__name__
        return module_type_map

    @staticmethod
    def generate_quant_model_description(model: torch.nn.Module):
        """
        Generate a description of the quantization properties for each parameter.

        This method creates a dictionary that describes the quantization type
        for each parameter in the model, distinguishing between quantized and
        floating-point weights.

        Args:
            model: The PyTorch model to analyze

        Returns:
            A dictionary mapping parameter names to their quantization types
        """
        quant_model_description = {}
        quantize_type = model.quant_config.quant_description.get("model_quant_type", "FLOAT")
        quant_model_description["model_quant_type"] = quantize_type
        quant_model_description["version"] = "1.0.0"
        module_type_map = ShardedStateLoader310.generate_module_type_map(model)
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if "Linear" in module_type_map[name] and tensor.dtype in ShardedStateLoader310.QUANTIZE_DTYPE_LIST:
                quant_model_description[name] = quantize_type
            else:
                quant_model_description[name] = "FLOAT"
        return quant_model_description
