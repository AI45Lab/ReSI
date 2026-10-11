"""Model-family contracts shared by critic LoRA training and reload gates."""
from dataclasses import dataclass

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


@dataclass(frozen=True)
class ModelFamilyContract:
    name: str
    model_loader: object
    lora_target: str
    expected_adapter_tensors: int


QWEN36_TARGET = (
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
MINISTRAL3_TARGET = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def resolve_model_family(base, requested="auto"):
    config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    if config.model_type not in {'mistral3','qwen3_5','qwen3_5_text'}:
        raise RuntimeError(f'unsupported model type for bundled trainer: {config.model_type}')
    detected = "ministral3" if config.model_type == "mistral3" else "qwen36"
    family = detected if requested == "auto" else requested
    if family != detected:
        raise RuntimeError(
            f"model-family mismatch: requested={family}, detected={detected}, "
            f"model_type={config.model_type}"
        )
    if family == "ministral3":
        layers = int(config.text_config.num_hidden_layers)
        if layers != 40:
            raise RuntimeError(f"unexpected Ministral text layer count: {layers}")
        return ModelFamilyContract(
            name=family,
            model_loader=AutoModelForImageTextToText,
            lora_target=MINISTRAL3_TARGET,
            expected_adapter_tensors=layers * 7 * 2,
        )
    if family == "qwen36":
        return ModelFamilyContract(
            name=family,
            model_loader=AutoModelForCausalLM,
            lora_target=QWEN36_TARGET,
            expected_adapter_tensors=992,
        )
    raise RuntimeError(f"unsupported model family: {family}")


def load_base_model(base, contract, **kwargs):
    model = contract.model_loader.from_pretrained(
        base, trust_remote_code=True, **kwargs
    )
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = False
    return model


def verify_adapter_tensor_count(state, contract):
    actual = len(state)
    if actual != contract.expected_adapter_tensors:
        raise RuntimeError(
            f"adapter tensor count mismatch for {contract.name}: "
            f"expected={contract.expected_adapter_tensors}, actual={actual}"
        )
