from types import SimpleNamespace

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


def _mk_loop() -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop._TOOL_RESULT_MAX_CHARS = AgentLoop._TOOL_RESULT_MAX_CHARS
    return loop


def test_save_turn_skips_multimodal_user_when_only_runtime_context() -> None:
    loop = _mk_loop()
    session = Session(key="test:runtime-only")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
        skip=0,
    )
    assert session.messages == []


def test_save_turn_keeps_image_placeholder_after_runtime_strip() -> None:
    loop = _mk_loop()
    session = Session(key="test:image")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        skip=0,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]


def test_save_turn_uses_image_summary_when_provided() -> None:
    loop = _mk_loop()
    session = Session(key="test:image-summary")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        skip=0,
        image_summary="screenshot of a login page with an invalid password error",
    )

    assert session.messages[0]["content"] == [
        {
            "type": "text",
            "text": "[image summary: screenshot of a login page with an invalid password error]",
        }
    ]


def test_save_turn_inserts_only_one_summary_for_multiple_images() -> None:
    loop = _mk_loop()
    session = Session(key="test:multi-image-summary")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
                ],
            }
        ],
        skip=0,
        image_summary="two screenshots related to a Python error",
    )

    assert session.messages[0]["content"] == [
        {
            "type": "text",
            "text": "[image summary: two screenshots related to a Python error]",
        }
    ]


def test_normalize_image_summary_truncates_long_content() -> None:
    loop = _mk_loop()
    summary = "x" * 200

    normalized = loop._normalize_image_summary(summary)

    assert normalized is not None
    assert normalized.endswith("...")
    assert len(normalized) == loop._IMAGE_SUMMARY_MAX_CHARS + 3


def test_normalize_image_summary_strips_marker_prefix() -> None:
    loop = _mk_loop()

    normalized = loop._normalize_image_summary(
        "[image summary: a receipt showing merchant name and total amount]"
    )

    assert normalized == "a receipt showing merchant name and total amount"


def test_normalize_image_summary_strips_bad_prefixes() -> None:
    loop = _mk_loop()

    normalized = loop._normalize_image_summary(
        "This image shows a login page with an invalid password error"
    )

    assert normalized == "a login page with an invalid password error"


def test_save_turn_keeps_tool_results_under_16k() -> None:
    loop = _mk_loop()
    session = Session(key="test:tool-result")
    content = "x" * 12_000

    loop._save_turn(
        session,
        [{"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": content}],
        skip=0,
    )

    assert session.messages[0]["content"] == content


@pytest.mark.asyncio
async def test_summarize_image_inputs_returns_none_without_media(tmp_path) -> None:
    loop = _mk_loop()

    result = await loop._summarize_image_inputs("hello", None)

    assert result is None


@pytest.mark.asyncio
async def test_summarize_image_inputs_uses_provider_response(tmp_path) -> None:
    loop = _mk_loop()
    loop.context = ContextBuilder(tmp_path)
    loop.model = "test-model"

    image_path = tmp_path / "img.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01"
        b"\x0b\x0e-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    async def _fake_chat_with_retry(**kwargs):
        return LLMResponse(content="a login page screenshot showing an invalid password error")

    loop.provider = SimpleNamespace(chat_with_retry=_fake_chat_with_retry)

    result = await loop._summarize_image_inputs("what is this?", [str(image_path)])

    assert result == "a login page screenshot showing an invalid password error"
