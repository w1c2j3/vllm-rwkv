from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.helicopter_attestation import attach_router


def _contract() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "served_name": "rwkv-test",
            "checkpoint_sha256": "a" * 64,
            "tokenizer_revision": "tok-v1",
            "chat_template_revision": "chat-v1",
        },
        "provider": {
            "server_revision": "server-v1",
            "wkv_mode": "fp32io16",
            "precision": "fp16-io-fp32-state",
            "gemm_policy": "fp32-accumulation",
            "launch_contract": "launch-v1",
        },
        "capabilities": [
            "openai-chat",
            "output-token-ids",
            "terminal-reason",
            "prompt-evidence",
        ],
    }


def test_attestation_route_is_absent_without_explicit_contract() -> None:
    app = FastAPI()
    attach_router(app, None)
    assert TestClient(app).get("/v1/helicopter/attestation").status_code == 404


def test_attestation_route_round_trips_validated_identity() -> None:
    app = FastAPI()
    contract = _contract()
    attach_router(app, json.dumps(contract))
    response = TestClient(app).get("/v1/helicopter/attestation")
    assert response.status_code == 200
    assert response.json() == contract


def test_attestation_rejects_unknown_or_incomplete_capabilities() -> None:
    contract = _contract()
    contract["capabilities"] = ["openai-chat"]
    with pytest.raises(ValueError, match="exact evaluation capabilities"):
        attach_router(FastAPI(), json.dumps(contract))
