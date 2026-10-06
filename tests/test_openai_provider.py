from unittest.mock import MagicMock

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

from app.models.exceptions import TransientProviderError, PermanentProviderError
from app.models.openai_provider import OpenAIProvider


def test_openai_timeout_becomes_transient_provider_error():

    mock_client = MagicMock()
    mock_request_client = MagicMock()

    # Inject a fake client so no API key or network access is needed
    provider = OpenAIProvider("test-model", client=mock_client)

    # with_options(...) returns the client used for this request
    mock_client.with_options.return_value = mock_request_client

    # Simulate OpenAI timing out
    mock_request_client.responses.create.side_effect = APITimeoutError(
        request=MagicMock()
    )

    with pytest.raises(TransientProviderError):
        provider.generate(
            prompt="hello",
            timeout_ms=2000,
        )


def test_openai_permanent_provider_error():

    mock_client = MagicMock()
    mock_request_client = MagicMock()

    # Inject a fake client so no API key or network access is needed
    provider = OpenAIProvider("test-model", client=mock_client)

    # with_options(...) returns the client used for this request
    mock_client.with_options.return_value = mock_request_client

    # Simulate OpenAI rejecting the API key
    mock_request_client.responses.create.side_effect = AuthenticationError(
        "Invalid API key",
        response=httpx.Response(
            401,
            request=httpx.Request(
                "POST", "https://api.openai.com/v1/responses"),
        ),
        body=None,
    )

    with pytest.raises(PermanentProviderError):
        provider.generate(
            prompt="hello",
            timeout_ms=2000,
        )
