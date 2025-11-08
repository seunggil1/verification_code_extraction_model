from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache

pytorch_model = gemma3.build_model_270m("./models/gemma3-270m-it-sms-verification_code_extraction")

export_config = ExportConfig()
export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input = True

converter.convert_to_tflite(
    pytorch_model,
    output_path="./models/gemma3-270m-it-sms-verification_code_extraction",
    output_name_prefix="sms_verification_code_extraction",
    prefill_seq_len=2048,
    kv_cache_max_len=4096,
    quantize="fp16",
    export_config=export_config,
)

