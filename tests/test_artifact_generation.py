"""Exercise the generation tools through the SDK with offline RPC responses."""

import json
from unittest.mock import Mock
from urllib.parse import parse_qs

import httpx
import pytest
from notebooklm import AuthTokens, NotebookLMClient
from notebooklm.rpc import ArtifactStatus, ArtifactTypeCode, RPCMethod
from notebooklm.types import QuizQuantity

from notebooklm_mcp_server import (
    generate_audio_overview,
    generate_infographic,
    generate_quiz,
    generate_report,
    generate_slide_deck,
    generate_video,
)


@pytest.fixture
async def artifact_rpc(mock_client, monkeypatch, tmp_path):
    """Record artifact RPC parameters and supply the responses for each test."""
    rpc = Mock()

    async def handle_request(_transport, request):
        method = RPCMethod(request.url.params["rpcids"])
        if method == RPCMethod.GET_NOTEBOOK:
            result = [["Notebook", [[["src-1"]]]]]
        else:
            envelope = json.loads(parse_qs(request.content.decode())["f.req"][0])
            result = rpc(method, json.loads(envelope[0][0][1]))
        frame = json.dumps([["wrb.fr", method.value, json.dumps(result)]])
        return httpx.Response(200, text=f")]}}'\n{len(frame)}\n{frame}")

    monkeypatch.setenv("NOTEBOOKLM_HOME", str(tmp_path))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)
    async with NotebookLMClient(AuthTokens(cookies={}, csrf_token="test", session_id="test")) as client:
        mock_client.artifacts = client.artifacts
        yield rpc


@pytest.mark.parametrize(
    ("options", "format_code", "length_code"),
    [
        ({}, 1, 2),
        ({"audio_format": "brief", "length": "short"}, 2, 1),
        ({"audio_format": "critique", "length": "long"}, 3, 3),
        ({"audio_format": "debate"}, 4, 2),
    ],
)
async def test_audio_request_options(options, format_code, length_code, mock_ctx, artifact_rpc):
    artifact_rpc.return_value = [["task-1", "Audio", ArtifactTypeCode.AUDIO, None, ArtifactStatus.FAILED]]

    result = await generate_audio_overview("nb-1", ctx=mock_ctx, **options)

    assert result["status"] == "failed"
    assert result["task_id"] == "task-1"
    method, params = artifact_rpc.call_args.args
    assert method == RPCMethod.CREATE_ARTIFACT
    audio_options = params[2][6][1]
    assert (audio_options[6], audio_options[1]) == (format_code, length_code)


@pytest.mark.parametrize(
    ("options", "codes"),
    [
        ({}, [2, 2]),
        ({"quantity": "few", "difficulty": "hard"}, [1, 3]),
        ({"quantity": "more", "difficulty": "easy"}, [QuizQuantity.MORE.value, 1]),
    ],
)
async def test_quiz_request_options(options, codes, mock_ctx, artifact_rpc):
    artifact_rpc.return_value = [["task-1", "Quiz", ArtifactTypeCode.QUIZ, None, ArtifactStatus.FAILED]]

    result = await generate_quiz("nb-1", ctx=mock_ctx, **options)

    assert result["status"] == "failed"
    assert result["task_id"] == "task-1"
    method, params = artifact_rpc.call_args.args
    assert method == RPCMethod.CREATE_ARTIFACT
    assert params[2][9][1][7] == codes


async def test_report_returns_error_from_failed_poll(mock_ctx, artifact_rpc):
    pending = ["report-1", "Report", ArtifactTypeCode.REPORT, None, ArtifactStatus.PENDING]
    failed = ["report-1", "Report", ArtifactTypeCode.REPORT, "No sources available", ArtifactStatus.FAILED]
    artifact_rpc.side_effect = [[pending], [[failed]]]

    result = await generate_report("nb-1", ctx=mock_ctx)

    assert result == {"task_id": "report-1", "status": "failed", "error": "No sources available"}


@pytest.mark.parametrize(
    ("tool", "artifact_type"),
    [
        (generate_audio_overview, ArtifactTypeCode.AUDIO),
        (generate_video, ArtifactTypeCode.VIDEO),
        (generate_infographic, ArtifactTypeCode.INFOGRAPHIC),
        (generate_slide_deck, ArtifactTypeCode.SLIDE_DECK),
    ],
)
async def test_completed_submission_without_media_waits_for_sdk_outcome(tool, artifact_type, mock_ctx, artifact_rpc):
    # The first poll still has no media URL. The second poll reports failure.
    missing_media = ["task-1", "Artifact", artifact_type, None, ArtifactStatus.COMPLETED]
    failed = ["task-1", "Artifact", artifact_type, None, ArtifactStatus.FAILED]
    artifact_rpc.side_effect = [[missing_media], [[missing_media]], [[failed]]]

    result = await tool("nb-1", ctx=mock_ctx)

    assert result["status"] == "failed"
    assert result["error"]
    assert all(call.args[0] != 1.0 for call in mock_ctx.report_progress.await_args_list)
