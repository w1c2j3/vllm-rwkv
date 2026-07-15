from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelAttestation(_StrictModel):
    served_name: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_revision: str
    chat_template_revision: str


class ProviderAttestation(_StrictModel):
    server_revision: str
    wkv_mode: str
    precision: str
    gemm_policy: str
    launch_contract: str


class HelicopterAttestation(_StrictModel):
    schema_version: Literal[1] = 1
    model: ModelAttestation
    provider: ProviderAttestation
    capabilities: tuple[
        Literal[
            "openai-chat", "output-token-ids", "terminal-reason", "prompt-evidence"
        ],
        ...,
    ]


def attach_router(app: FastAPI, raw_contract: str | None) -> None:
    if raw_contract is None:
        return
    try:
        contract = HelicopterAttestation.model_validate(json.loads(raw_contract))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "--helicopter-attestation-json is not a valid signed contract"
        ) from error
    required = {"openai-chat", "output-token-ids", "terminal-reason", "prompt-evidence"}
    if set(contract.capabilities) != required:
        raise ValueError(
            "helicopter attestation must declare the exact evaluation capabilities"
        )
    router = APIRouter()

    @router.get("/v1/helicopter/attestation", response_model=HelicopterAttestation)
    async def show_attestation() -> HelicopterAttestation:
        return contract

    app.include_router(router)
