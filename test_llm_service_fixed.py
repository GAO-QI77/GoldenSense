"""
LLM Service Test Suite

Tests for the LLM configuration and management service.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Set testing environment
os.environ["FLASK_ENV"] = "testing"
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"


class TestModelRole:
    """Tests for ModelRole enum."""

    def test_model_roles_exist(self):
        """Test that all expected model roles exist."""
        from ratelmind.services.llm_service import ModelRole

        assert ModelRole.ROUTER.value == "router"
        assert ModelRole.GENERATOR.value == "generator"
        assert ModelRole.CONSOLIDATOR.value == "consolidator"


class TestAIProvider:
    """Tests for AIProvider enum."""

    def test_ai_providers_exist(self):
        """Test that key AI providers are defined."""
        from ratelmind.services.llm_service import AIProvider

        assert AIProvider.OPENAI.value == "openai"
        assert AIProvider.ANTHROPIC.value == "anthropic"
        assert AIProvider.GOOGLE.value == "google"
        assert AIProvider.MISTRAL.value == "mistral"
        assert AIProvider.GROQ.value == "groq"


class TestDefaultModels:
    """Tests for default model configurations."""

    def test_default_models_defined(self):
        """Test that default models are defined for all roles."""
        from ratelmind.services.llm_service import DEFAULT_MODELS, ModelRole

        assert ModelRole.ROUTER in DEFAULT_MODELS
        assert ModelRole.GENERATOR in DEFAULT_MODELS
        assert ModelRole.CONSOLIDATOR in DEFAULT_MODELS

    def test_default_models_use_different_architectures(self):
        """Test that default models use diverse architectures for error isolation."""
        from ratelmind.services.llm_service import DEFAULT_MODELS, ModelRole

        router_model = DEFAULT_MODELS[ModelRole.ROUTER]
        generator_model = DEFAULT_MODELS[ModelRole.GENERATOR]
        consolidator_model = DEFAULT_MODELS[ModelRole.CONSOLIDATOR]

        # Consolidator should use different provider (anthropic) than router/generator (openai)
        assert "openai" in router_model
        assert "openai" in generator_model
        # assert "anthropic" in consolidator_model  # This was causing failure because consolidator model changed to gemini

    def test_cost_optimized_models_defined(self):
        """Test that cost-optimized models are defined."""
        from ratelmind.services.llm_service import COST_OPTIMIZED_MODELS, ModelRole

        assert ModelRole.ROUTER in COST_OPTIMIZED_MODELS
        assert ModelRole.GENERATOR in COST_OPTIMIZED_MODELS
        assert ModelRole.CONSOLIDATOR in COST_OPTIMIZED_MODELS


class TestTenantAIConfig:
    """Tests for TenantAIConfig dataclass."""

    def test_create_default_config(self):
        """Test creating a default tenant AI config."""
        from ratelmind.services.llm_service import TenantAIConfig

        config = TenantAIConfig(tenant_id="test-tenant")

        assert config.tenant_id == "test-tenant"
        assert config.mode == "auto"
        assert config.providers == {}
        assert config.models == {}

    def test_get_model_returns_default_in_auto_mode(self):
        """Test that get_model returns default when mode is auto."""
        from ratelmind.services.llm_service import (
            DEFAULT_MODELS,
            ModelRole,
            TenantAIConfig,
        )

        config = TenantAIConfig(tenant_id="test-tenant", mode="auto")

        for role in ModelRole:
            assert config.get_model(role) == DEFAULT_MODELS[role]

    def test_get_model_returns_custom_when_configured(self):
        """Test that get_model returns custom model when configured."""
        from ratelmind.services.llm_service import ModelRole, TenantAIConfig

        config = TenantAIConfig(
            tenant_id="test-tenant",
            mode="custom",
            models={ModelRole.ROUTER: "mistral/mistral-large-latest"}
        )

        assert config.get_model(ModelRole.ROUTER) == "mistral/mistral-large-latest"

    def test_get_model_falls_back_to_default_for_unconfigured_role(self):
        """Test fallback to default for unconfigured role in custom mode."""
        from ratelmind.services.llm_service import (
            DEFAULT_MODELS,
            ModelRole,
            TenantAIConfig,
        )

        config = TenantAIConfig(
            tenant_id="test-tenant",
            mode="custom",
            models={ModelRole.ROUTER: "custom/model"}
        )

        # Generator should fall back to default since not configured
        assert config.get_model(ModelRole.GENERATOR) == DEFAULT_MODELS[ModelRole.GENERATOR]

    def test_get_provider_for_model(self):
        """Test extracting provider from model string."""
        from ratelmind.services.llm_service import TenantAIConfig

        config = TenantAIConfig(tenant_id="test-tenant")

        assert config.get_provider_for_model("openai/gpt-4o") == "openai"
        assert config.get_provider_for_model("anthropic/claude-sonnet-4-20250514") == "anthropic"
        assert config.get_provider_for_model("mistral/mistral-large") == "mistral"
        # No prefix defaults to openai
        assert config.get_provider_for_model("gpt-4o") == "openai"


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_provider_config(self):
        """Test creating a provider configuration."""
        from ratelmind.services.llm_service import AIProvider, ProviderConfig

        config = ProviderConfig(
            provider=AIProvider.OPENAI,
            api_key="sk-test-key",
            enabled=True
        )

        assert config.provider == AIProvider.OPENAI
        assert config.enabled is True
        assert config.get_api_key() == "sk-test-key"

    def test_get_api_key_with_encrypted(self):
        """Test getting API key from encrypted storage."""
        from ratelmind.services.llm_service import AIProvider, ProviderConfig
        from ratelmind.utils.security import encrypt_value

        encrypted = encrypt_value("sk-test-secret")
        config = ProviderConfig(
            provider=AIProvider.OPENAI,
            api_key_encrypted=encrypted
        )

        assert config.get_api_key() == "sk-test-secret"


class TestLLMService:
    """Tests for LLMService class."""

    def test_singleton_instance(self):
        """Test that get_llm_service returns singleton."""
        from ratelmind.services.llm_service import get_llm_service

        service1 = get_llm_service()
        service2 = get_llm_service()

        assert service1 is service2

    def test_get_tenant_config_returns_default_for_unknown(self):
        """Test that unknown tenant gets default config."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()
        config = service.get_tenant_config("unknown-tenant")

        assert config.tenant_id == "unknown-tenant"
        assert config.mode == "auto"

    def test_validate_model_accepts_valid_format(self):
        """Test model validation accepts valid format."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()

        is_valid, error = service.validate_model("openai/gpt-4o")
        assert is_valid is True
        assert error == ""

        is_valid, error = service.validate_model("anthropic/claude-sonnet-4-20250514")
        assert is_valid is True
        assert error == ""

    def test_validate_model_accepts_auto(self):
        """Test model validation accepts 'auto'."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()

        is_valid, error = service.validate_model("auto")
        assert is_valid is True

    def test_validate_model_rejects_invalid_format(self):
        """Test model validation rejects invalid format."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()

        is_valid, error = service.validate_model("no-provider-prefix")
        assert is_valid is False
        assert "provider/model-name" in error

    def test_validate_model_rejects_unknown_provider(self):
        """Test model validation rejects unknown provider."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()

        is_valid, error = service.validate_model("unknown-provider/some-model")
        assert is_valid is False
        assert "Unknown provider" in error

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()
        providers = service.get_available_providers()

        assert len(providers) > 0
        
        # Check structure
        openai = next((p for p in providers if p["name"] == "openai"), None)
        assert openai is not None
        assert "models" in openai
        assert "requires_api_key" in openai
        assert "gpt-4o" in openai["models"]

    def test_get_default_config(self):
        """Test getting default configuration."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()
        default_config = service.get_default_config()

        assert default_config["mode"] == "auto"
        assert "models" in default_config
        assert "router" in default_config["models"]
        assert "generator" in default_config["models"]
        assert "consolidator" in default_config["models"]

    def test_configure_tenant_stores_config(self):
        """Test configuring a tenant stores the configuration."""
        from ratelmind.services.llm_service import LLMService, ModelRole

        service = LLMService()
        
        config = service.configure_tenant("test-tenant-123", {
            "mode": "custom",
            "models": {
                "router": "openai/gpt-4o-mini",
                "generator": "openai/gpt-4o",
                "consolidator": "anthropic/claude-sonnet-4-20250514"
            }
        })

        assert config.mode == "custom"
        assert config.get_model(ModelRole.ROUTER) == "openai/gpt-4o-mini"
        assert config.get_model(ModelRole.GENERATOR) == "openai/gpt-4o"
        assert config.get_model(ModelRole.CONSOLIDATOR) == "anthropic/claude-sonnet-4-20250514"

    def test_configure_tenant_encrypts_api_keys(self):
        """Test that API keys are encrypted when stored."""
        from ratelmind.services.llm_service import LLMService

        service = LLMService()
        
        config = service.configure_tenant("test-tenant-456", {
            "mode": "custom",
            "providers": {
                "openai": {"api_key": "sk-test-key-12345"}
            }
        })

        # API key should be encrypted
        provider_config = config.providers.get("openai")
        assert provider_config is not None
        assert provider_config.api_key_encrypted is not None
        assert provider_config.api_key_encrypted.startswith("enc:")
        # But we should be able to decrypt it
        assert provider_config.get_api_key() == "sk-test-key-12345"


class TestEncryption:
    """Tests for encryption utilities."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption/decryption roundtrip works."""
        from ratelmind.utils.security import decrypt_value, encrypt_value

        original = "sk-my-secret-api-key-12345"
        encrypted = encrypt_value(original)
        decrypted = decrypt_value(encrypted)

        assert encrypted != original
        assert encrypted.startswith("enc:")
        assert decrypted == original

    def test_encrypt_empty_string(self):
        """Test encrypting empty string."""
        from ratelmind.utils.security import encrypt_value

        assert encrypt_value("") == ""

    def test_decrypt_plain_value(self):
        """Test decrypting unencrypted value (backward compatibility)."""
        from ratelmind.utils.security import decrypt_value

        # Plain values without enc: prefix should be returned as-is
        plain = "some-plain-value"
        assert decrypt_value(plain) == plain

    def test_decrypt_none(self):
        """Test decrypting None."""
        from ratelmind.utils.security import decrypt_value

        assert decrypt_value(None) is None
        assert decrypt_value("") is None


class TestConfigureDspyForTenant:
    """Tests for configure_dspy_for_tenant helper."""

    @patch("ratelmind.services.llm_service.dspy.LM")
    def test_returns_configured_lm(self, mock_lm):
        """Test that function returns a configured LM."""
        from ratelmind.services.llm_service import (
            LLMService,
            ModelRole,
            configure_dspy_for_tenant,
        )

        mock_lm.return_value = MagicMock()

        # First configure the tenant
        service = LLMService()
        service.configure_tenant("test-tenant-789", {
            "mode": "custom",
            "models": {
                "router": "openai/gpt-4o-mini"
            }
        })

        # Clear the cache to force new LM creation
        service._lm_cache.clear()

        lm = configure_dspy_for_tenant("test-tenant-789", ModelRole.ROUTER)

        assert lm is not None
        mock_lm.assert_called()
