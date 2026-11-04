from vllm import ModelRegistry


def register_model():
    from vllm_ascend.patch.hy4_config import HY_DFLASH_ARCHITECTURES, register_hy4_configs

    register_hy4_configs()
    for architecture in HY_DFLASH_ARCHITECTURES:
        for name in (architecture, f"DFlash{architecture}"):
            ModelRegistry.register_model(name, "vllm_ascend.patch.worker.hy_dflash_model:HY4DflyModels")

    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("HYV4ForCausalLM", "vllm_ascend.models.hy_v4:AscendHYV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
