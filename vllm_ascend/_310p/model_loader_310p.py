
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import ShardedStateLoader
from vllm.distributed import get_tensor_model_parallel_rank
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor


class ShardedStateLoader310(ShardedStateLoader):

    QUANTIZE_DTYPE_LIST = [torch.int8, torch.int32, torch.int64]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        if pattern is None:
            pattern = ShardedStateLoader310.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        quant_model_description = ShardedStateLoader310.generate_quant_model_description(model)
        state_dict = ShardedStateLoader310._filter_subtensors(model.state_dict())
        compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True,
                                         record_detail_root=path, multiprocess_num=2)
        compressor = Compressor(compress_config, weight=state_dict, quant_model_description=quant_model_description)
        compressor.run()
        filename = pattern.format(rank=rank, part=part_idx)
        compressor.export_safetensors(path, safetensors_name=filename)
            
    @staticmethod
    def generate_module_type_map(model: torch.nn.Module):
        """Generate module type map of model."""
        module_type_map = {}
        module_dict = dict(model.named_modules())
        state_dict = model.state_dict()
        for name in state_dict.keys():
            module_path = name.rsplit('.', 1)[0]
            module = module_dict.get(module_path)
            if module:
                module_type_map[name] = type(module).__name__
        return module_type_map   

    @staticmethod
    def generate_quant_model_description(model: torch.nn.Module):
        """Generate module description of quant weight."""
        quant_model_description = {}
        quantize_type = model.quant_config.quant_description.get("model_quant_type", "FLOAT")
        quant_model_description['model_quant_type'] = quantize_type
        quant_model_description['version'] = "1.0.0"
        module_type_map = ShardedStateLoader310.generate_module_type_map(model)
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if 'Linear' in module_type_map[name] and tensor.dtype in ShardedStateLoader310.QUANTIZE_DTYPE_LIST:
                quant_model_description[name] = quantize_type
            else:
                quant_model_description[name] = 'FLOAT'
        return quant_model_description
