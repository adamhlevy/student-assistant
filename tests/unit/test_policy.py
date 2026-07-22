# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from app.agent import (
    AcademicPolicyPlugin,
    academic_input_guardrail,
    pii_output_guardrail,
)


@pytest.mark.asyncio
async def test_academic_input_guardrail_violation() -> None:
    """Test that input guardrail blocks coursework cheating requests."""
    mock_ctx = MagicMock()
    mock_event = MagicMock()
    mock_event.content.role = "user"

    mock_part = MagicMock()
    mock_part.text = "Can you write my essay for me on Shakespeare?"
    mock_event.content.parts = [mock_part]

    mock_ctx.session.events = [mock_event]
    mock_req = MagicMock(spec=LlmRequest)

    response = await academic_input_guardrail(mock_ctx, mock_req)
    assert response is not None
    assert response.content is not None
    assert response.content.parts is not None
    text = response.content.parts[0].text
    assert text is not None
    assert "Academic Guardrail" in text


@pytest.mark.asyncio
async def test_academic_input_guardrail_pass() -> None:
    """Test that normal academic assistant requests pass input guardrail."""
    mock_ctx = MagicMock()
    mock_event = MagicMock()
    mock_event.content.role = "user"

    mock_part = MagicMock()
    mock_part.text = "When does Calculus I start?"
    mock_event.content.parts = [mock_part]

    mock_ctx.session.events = [mock_event]
    mock_req = MagicMock(spec=LlmRequest)

    response = await academic_input_guardrail(mock_ctx, mock_req)
    assert response is None


@pytest.mark.asyncio
async def test_pii_output_guardrail() -> None:
    """Test that PII guardrail successfully redacts SSN and Student IDs."""
    mock_ctx = MagicMock()
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="The student's SSN is 123-45-6789 and their ID is STUDENT-54321."
                )
            ],
        )
    )

    cleaned_response = await pii_output_guardrail(mock_ctx, response)
    assert cleaned_response is not None
    assert cleaned_response.content is not None
    assert cleaned_response.content.parts is not None
    text = cleaned_response.content.parts[0].text
    assert text is not None
    assert "123-45-6789" not in text
    assert "STUDENT-54321" not in text
    assert "[SSN REDACTED]" in text
    assert "[STUDENT ID REDACTED]" in text


@pytest.mark.asyncio
async def test_academic_policy_plugin_credit_hours_cap() -> None:
    """Test that AcademicPolicyPlugin blocks enrollment exceeding 15 total hours."""
    plugin = AcademicPolicyPlugin()
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "enroll_in_class"

    # Mock ToolContext state containing enrolled classes of total 14 hours
    mock_tool_context = MagicMock(spec=ToolContext)
    mock_tool_context.state = {
        "enrolled_classes": [
            {
                "name": "Organic Chemistry",
                "start_times": ["13:00, 17:00"],
                "duration_hrs": 3,
            },
            {
                "name": "Biochemistry",
                "start_times": ["10:00, 15:00"],
                "duration_hrs": 3,
            },
            {
                "name": "General Physics I",
                "start_times": ["11:00, 16:00"],
                "duration_hrs": 3,
            },
            {"name": "Ecology", "start_times": ["14:00, 18:00"], "duration_hrs": 3},
            {
                "name": "Linear Algebra",
                "start_times": ["14:00, 18:00"],
                "duration_hrs": 2,
            },
        ]
    }

    # Attempting to enroll in a 2-hour class (total will be 16)
    args = {"class_name": "Database Systems"}  # 2 hours
    res = await plugin.before_tool_callback(
        tool=mock_tool, tool_args=args, tool_context=mock_tool_context
    )

    assert res is not None
    assert res["status"] == "error"
    assert "Academic Policy Violation" in res["message"]
    assert "exceed your credit limit of 15 hours" in res["message"]


@pytest.mark.asyncio
async def test_academic_policy_plugin_schedule_conflict() -> None:
    """Test that AcademicPolicyPlugin blocks enrollment causing schedule conflicts."""
    plugin = AcademicPolicyPlugin()
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "enroll_in_class"

    # Mock ToolContext state containing already enrolled class at 09:00
    mock_tool_context = MagicMock(spec=ToolContext)
    mock_tool_context.state = {
        "enrolled_classes": [
            {
                "name": "Introduction to Computer Science",
                "start_times": ["09:00, 14:00"],
                "duration_hrs": 1,
            }
        ]
    }

    # Attempting to enroll in Calculus II (09:00, 14:00)
    args = {"class_name": "Calculus II"}
    res = await plugin.before_tool_callback(
        tool=mock_tool, tool_args=args, tool_context=mock_tool_context
    )

    assert res is not None
    assert res["status"] == "error"
    assert "Academic Policy Violation" in res["message"]
    assert "conflicts with your already enrolled class" in res["message"]


@pytest.mark.asyncio
async def test_academic_policy_plugin_after_tool_callback() -> None:
    """Test that AcademicPolicyPlugin correctly updates state after successful enrollment."""
    plugin = AcademicPolicyPlugin()
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "enroll_in_class"

    mock_tool_context = MagicMock(spec=ToolContext)
    mock_tool_context.state = {"enrolled_classes": []}

    args = {"class_name": "Calculus I"}
    tool_response = {
        "status": "success",
        "course": {
            "name": "Calculus I",
            "start_times": ["13:00, 17:00"],
            "duration_hrs": 1,
        },
    }

    await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=args,
        tool_context=mock_tool_context,
        result=tool_response,
    )

    assert len(mock_tool_context.state["enrolled_classes"]) == 1
    assert mock_tool_context.state["enrolled_classes"][0]["name"] == "Calculus I"
