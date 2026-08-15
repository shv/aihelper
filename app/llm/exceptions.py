class LLMError(Exception):
    """Базовая ошибка LLM-слоя."""


class LLMTimeoutError(LLMError):
    """Провайдер не ответил вовремя."""


class LLMRateLimitError(LLMError):
    """Провайдер ограничил частоту запросов."""


class LLMUnavailableError(LLMError):
    """Провайдер временно недоступен."""


class LLMAuthenticationError(LLMError):
    """Ошибка ключа или доступа к провайдеру."""


class LLMInvalidResponseError(LLMError):
    """Провайдер вернул неполный или некорректный ответ."""
