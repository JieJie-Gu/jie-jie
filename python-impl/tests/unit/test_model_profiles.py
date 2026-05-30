# 测试不同 LLM 任务 profile 的模型选择和回退逻辑。

from __future__ import annotations

from smart_cs.config import Settings
from smart_cs.infrastructure.model_factory import ModelProfiles, configured_model_profiles


def test_model_profiles_fall_back_to_base_llm_model() -> None:
    settings = Settings(
        llm_model="base-model",
        llm_api_key="test-key",
        llm_base_url="http://localhost/v1",
    )

    profiles = configured_model_profiles(settings)

    assert isinstance(profiles, ModelProfiles)
    assert profiles.agent.model_name == "base-model"
    assert profiles.extraction.model_name == "base-model"
    assert profiles.summary.model_name == "base-model"
    assert profiles.memory.model_name == "base-model"
    assert profiles.rag.model_name == "base-model"
    assert profiles.vision.model_name == "base-model"


def test_model_profiles_use_task_specific_model_when_configured() -> None:
    settings = Settings(
        llm_model="base-model",
        extraction_model="extract-model",
        summary_model="summary-model",
        memory_model="memory-model",
        rag_model="rag-model",
        vision_model="vision-model",
        llm_api_key="test-key",
        llm_base_url="http://localhost/v1",
    )

    profiles = configured_model_profiles(settings)

    assert profiles.agent.model_name == "base-model"
    assert profiles.extraction.model_name == "extract-model"
    assert profiles.summary.model_name == "summary-model"
    assert profiles.memory.model_name == "memory-model"
    assert profiles.rag.model_name == "rag-model"
    assert profiles.vision.model_name == "vision-model"
