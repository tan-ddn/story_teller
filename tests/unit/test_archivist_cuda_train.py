import pytest

from training.archivist.cuda_train import expected_lora_layers, validate_smoke


def test_replacement_models_target_final_16_layers():
    assert expected_lora_layers("zai-org/glm-4-9b-chat-hf") == list(range(24, 40))
    assert expected_lora_layers("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") == list(range(32, 48))


def test_full_run_requires_matching_finite_smoke():
    audit = {"train": {"sha256": "train"}, "valid": {"sha256": "valid"}}
    report = {
        "model": "qwen",
        "lora_layers": list(range(48, 64)),
        "train_scene": audit["train"],
        "valid_scene": audit["valid"],
        "train_metrics": {"train_loss": 0.2},
        "eval_metrics": {"eval_loss": 0.3},
    }
    validate_smoke(report, audit, "qwen")
    with pytest.raises(ValueError, match="model"):
        validate_smoke(report, audit, "other")
    report["eval_metrics"]["eval_loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_smoke(report, audit, "qwen")
