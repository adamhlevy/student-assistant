# ruff: noqa
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

import asyncio
import contextvars
import datetime
import json
import logging
import os
import re

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models import Gemini
from google.genai import types
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, List, Any

from google.adk.tools import FunctionTool, ToolContext, BaseTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

# Context variables to track active session identifiers
active_session_id = contextvars.ContextVar("active_session_id", default=None)
active_user_id = contextvars.ContextVar("active_user_id", default=None)
active_app_name = contextvars.ContextVar("active_app_name", default=None)


class AsyncLlmEventSummarizer(LlmEventSummarizer):
    """An LLM-based event summarizer that executes compaction in an asynchronous background task.

    This avoids blocking the main conversation thread on the latent LLM summarization call.
    """

    async def maybe_summarize_events(self, *, events: List[Event]) -> Optional[Event]:
        """Summarizes the events in a background task and appends the result to the session."""
        session_id = active_session_id.get()
        user_id = active_user_id.get()
        app_name = active_app_name.get()

        # If session context is missing, run synchronously as a fallback
        if not (session_id and user_id and app_name):
            return await super().maybe_summarize_events(events=events)

        async def run_compaction():
            try:
                # 1. Run the LLM summarization (latent call)
                compacted_event = await super(
                    AsyncLlmEventSummarizer, self
                ).maybe_summarize_events(events=events)
                if compacted_event:
                    # 2. Append the compacted event asynchronously back to the session history
                    from app.app_utils.services import get_session_service

                    session_service = get_session_service()
                    session = await session_service.get_session(
                        app_name=app_name, user_id=user_id, session_id=session_id
                    )
                    if session:
                        await session_service.append_event(
                            session=session, event=compacted_event
                        )
            except Exception as e:
                import logging

                logging.getLogger("AsyncLlmEventSummarizer").error(
                    f"Failed to run background event compaction: {e}", exc_info=True
                )

        # Dispatch compaction to background event loop task
        asyncio.create_task(run_compaction())

        # Return None immediately to unblock the current user transaction
        return None


async def populate_session_context(callback_context: CallbackContext) -> None:
    """Populates context variables with session identifiers before agent runs."""
    active_session_id.set(callback_context.session.id)
    active_user_id.set(callback_context.session.user_id)
    active_app_name.set(callback_context.session.app_name)


class Course(BaseModel):
    name: str = Field(description="The unique name of the college class.")
    start_times: List[str] = Field(
        description="The available start times of the class."
    )
    duration_hrs: int = Field(description="The duration of the class in hours.")
    professor: str = Field(description="The professor teaching the class.")
    max_students: int = Field(
        description="The maximum number of students allowed in the class."
    )
    enrolled_students: int = Field(
        description="The number of currently enrolled students."
    )


class CoursesCatalog(BaseModel):
    courses: List[Course] = Field(
        description="The list of all available college courses."
    )


class EnrollInput(BaseModel):
    class_name: str = Field(description="The name of the class to enroll in.")


class EnrollResult(BaseModel):
    status: str = Field(
        description="The status of the enrollment, e.g., 'success' or 'error'."
    )
    message: str = Field(
        description="Detailed message explaining the enrollment outcome."
    )
    course: Optional[Course] = Field(
        default=None,
        description="The updated course details, if enrollment is successful.",
    )


def get_available_classes() -> dict:
    """Get the list of all available college classes.

    Returns:
        dict: A dictionary containing available classes.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    courses_file_path = os.path.join(current_dir, "courses.json")
    with open(courses_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Strict validation of catalog layout using Pydantic
    validated_catalog = CoursesCatalog.model_validate(data)
    return validated_catalog.model_dump()


def enroll_in_class(class_name: str) -> dict:
    """Enroll a student in a class.

    This function updates the enrollment count for the specified class in the course catalog.
    Students can only enroll in classes where the current number of enrolled students is less than the max students.

    Args:
        class_name: The name of the class to enroll in.

    Returns:
        dict: A dictionary containing the status, message, and updated course details.
    """
    # Strict validation of input parameters
    validated_input = EnrollInput(class_name=class_name)
    class_name = validated_input.class_name

    current_dir = os.path.dirname(os.path.abspath(__file__))
    courses_file_path = os.path.join(current_dir, "courses.json")
    with open(courses_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Strictly validate raw json catalog data on load
    catalog = CoursesCatalog.model_validate(data)

    found_course = None
    for course in catalog.courses:
        if course.name == class_name:
            found_course = course
            break

    if not found_course:
        # Case-insensitive fallback for robust matching
        for course in catalog.courses:
            if course.name.strip().lower() == class_name.strip().lower():
                found_course = course
                break

    if not found_course:
        result = EnrollResult(
            status="error",
            message=f"Class '{class_name}' was not found in the course catalog.",
        )
        return result.model_dump()

    enrolled = found_course.enrolled_students
    max_students = found_course.max_students

    if enrolled >= max_students:
        result = EnrollResult(
            status="error",
            message=f"Cannot enroll in '{found_course.name}'. The class is full ({enrolled}/{max_students} students).",
        )
        return result.model_dump()

    found_course.enrolled_students = enrolled + 1

    # Save the updated catalog data back to file
    with open(courses_file_path, "w", encoding="utf-8") as f:
        json.dump(catalog.model_dump(), f, indent=4, ensure_ascii=False)

    result = EnrollResult(
        status="success",
        message=f"Successfully enrolled in '{found_course.name}'!",
        course=found_course,
    )
    return result.model_dump()


# =====================================================================
# 1. RUNTIME GUARDRAILS (Callbacks)
# =====================================================================


async def academic_input_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Guardrail to intercept and block non-academic, toxic or out-of-domain requests."""
    events = callback_context.session.events
    if not events:
        return None

    last_user_message = None
    for event in reversed(events):
        if event.content and event.content.role == "user":
            parts = event.content.parts
            if parts and hasattr(parts[0], "text") and parts[0].text:
                last_user_message = parts[0].text
                break

    if not last_user_message:
        return None

    text = last_user_message.lower().strip()
    academic_violation_keywords = [
        "write my essay",
        "do my homework",
        "hack university",
        "cheat on exam",
        "bypass grade",
    ]

    if any(kw in text for kw in academic_violation_keywords):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Academic Guardrail: I can only assist with class schedule queries and official enrollment actions. I am strictly prohibited from assisting with coursework, assignments, or any actions that violate academic integrity."
                    )
                ],
            )
        )

    return None


async def pii_output_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Guardrail to redact sensitive PII (Student IDs, SSNs) from model output."""
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return None

    part = llm_response.content.parts[0]
    if not hasattr(part, "text") or not part.text:
        return None

    text = part.text
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    student_id_pattern = r"\bSTUDENT-\d{5,8}\b"

    cleaned_text = re.sub(ssn_pattern, "[SSN REDACTED]", text)
    cleaned_text = re.sub(student_id_pattern, "[STUDENT ID REDACTED]", cleaned_text)

    if cleaned_text != text:
        part.text = cleaned_text
        return llm_response

    return None

    return None


# =====================================================================
# 2. CAMPUS POLICY PLUGINS
# =====================================================================


class AcademicPolicyPlugin(BasePlugin):
    """Enforces official university policies on credit limits and scheduling conflicts."""

    def __init__(self):
        super().__init__(name="academic_policy_plugin")

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        if tool.name != "enroll_in_class":
            return None

        class_name = tool_args.get("class_name")
        if not class_name:
            return None

        catalog = get_available_classes()
        courses = catalog.get("courses", [])

        target_course = None
        for course in courses:
            if course["name"].strip().lower() == class_name.strip().lower():
                target_course = course
                break

        if not target_course:
            return None

        # Fetch currently enrolled classes from session state
        enrolled_classes = tool_context.state.get("enrolled_classes", [])

        # Policy 1: Credit Hours Cap (Maximum 15 hours)
        current_total_hours = sum(c.get("duration_hrs", 0) for c in enrolled_classes)
        target_hours = target_course.get("duration_hrs", 0)

        if current_total_hours + target_hours > 15:
            return {
                "status": "error",
                "message": f"Academic Policy Violation: Enrolling in '{target_course['name']}' ({target_hours} hrs) would exceed your credit limit of 15 hours per session. Currently enrolled: {current_total_hours} hours.",
            }

        # Policy 2: Schedule Conflict Prevention
        target_start_times = target_course.get("start_times", [])
        for enrolled in enrolled_classes:
            enrolled_times = enrolled.get("start_times", [])
            enrolled_times_set = {
                t.strip() for time_str in enrolled_times for t in time_str.split(",")
            }
            target_times_set = {
                t.strip()
                for time_str in target_start_times
                for t in time_str.split(",")
            }

            common_times = enrolled_times_set.intersection(target_times_set)
            if common_times:
                overlap_time = list(common_times)[0]
                return {
                    "status": "error",
                    "message": f"Academic Policy Violation: '{target_course['name']}' conflicts with your already enrolled class '{enrolled['name']}' at {overlap_time}.",
                }

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if tool.name == "enroll_in_class" and result.get("status") == "success":
            course_details = result.get("course")
            if course_details:
                enrolled_classes = tool_context.state.get("enrolled_classes", [])
                if not any(
                    c["name"] == course_details["name"] for c in enrolled_classes
                ):
                    enrolled_classes.append(course_details)
                    tool_context.state["enrolled_classes"] = enrolled_classes
        return None


class ActionOutcomeLoggingPlugin(BasePlugin):
    """Global ADK plugin that explicitly logs intended actions versus execution outcomes."""

    def __init__(self):
        super().__init__(name="action_outcome_logging")
        self.logger = logging.getLogger("student_assistant.action_outcome_logger")

    async def before_agent_callback(
        self, *, agent: "BaseAgent", callback_context: "CallbackContext"
    ) -> Optional[types.Content]:
        session_id = "unknown"
        if callback_context and getattr(callback_context, "session", None):
            session_id = getattr(callback_context.session, "id", "unknown")
        agent_name = getattr(agent, "name", "unknown")
        self.logger.info(
            f"[INTENDED ACTION] Running agent: '{agent_name}' (Session ID: {session_id})"
        )
        return None

    async def after_agent_callback(
        self, *, agent: "BaseAgent", callback_context: "CallbackContext"
    ) -> Optional[types.Content]:
        session_id = "unknown"
        if callback_context and getattr(callback_context, "session", None):
            session_id = getattr(callback_context.session, "id", "unknown")
        agent_name = getattr(agent, "name", "unknown")
        self.logger.info(
            f"[EXECUTION OUTCOME] Completed agent: '{agent_name}' (Session ID: {session_id})"
        )
        return None

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> Optional[LlmResponse]:
        agent_name = getattr(callback_context, "agent_name", "unknown")
        self.logger.info(f"[INTENDED ACTION] Querying LLM for agent: '{agent_name}'")
        return None

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: "LlmResponse"
    ) -> Optional[LlmResponse]:
        agent_name = getattr(callback_context, "agent_name", "unknown")

        # Log LLM's response or chosen action (e.g., tool calling or delegation)
        parts_text = []
        function_calls = []
        if llm_response.content and llm_response.content.parts:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(
                        f"{part.function_call.name}({part.function_call.args})"
                    )

        outcome_desc = ""
        if function_calls:
            outcome_desc += (
                f" [Intended Actions / Tool Calls: {', '.join(function_calls)}]"
            )
        if parts_text:
            outcome_desc += f" [Text Response: {repr(' '.join(parts_text)[:150])}...]"

        self.logger.info(
            f"[EXECUTION OUTCOME] LLM responded for agent '{agent_name}':{outcome_desc}"
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
    ) -> Optional[dict[str, Any]]:
        self.logger.info(
            f"[INTENDED ACTION] Executing tool: '{tool.name}' with arguments: {tool_args}"
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        self.logger.info(
            f"[EXECUTION OUTCOME] Tool '{tool.name}' completed successfully. Result: {result}"
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
        error: Exception,
    ) -> Optional[dict[str, Any]]:
        self.logger.error(
            f"[EXECUTION OUTCOME] Tool '{tool.name}' failed with error: {error}",
            exc_info=True,
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
        error: Exception,
    ) -> Optional[LlmResponse]:
        agent_name = getattr(callback_context, "agent_name", "unknown")
        self.logger.error(
            f"[EXECUTION OUTCOME] LLM query for agent '{agent_name}' failed with error: {error}",
            exc_info=True,
        )
        return None


# =====================================================================
# 3. SPECIALIST AGENTS & STRATEGIC ROUTING
# =====================================================================

catalog_agent = Agent(
    name="catalog_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Specialist agent for listing, searching, looking up, and retrieving information about available college classes, professor details, and course schedules in the catalog.",
    instruction="""You are a course catalog specialist.
Your primary directives are:
1. Use the `get_available_classes` tool to retrieve all course information.
2. Base all answers strictly on the class information returned by the tools. Do NOT hallucinate, assume, or fabricate any details (such as professors, schedules, duration, or availability) not explicitly present in the data.
3. If a student asks about a class, professor, or schedule that does not exist in the retrieved classes, state clearly and politely that the class/information is not available in the courses catalog.
4. Format class details cleanly using bullet points:
   - **Course Name**: [name]
   - **Professor**: [professor]
   - **Start Times**: [start_times]
   - **Duration**: [duration_hrs] hour(s)
   - **Enrollment**: [enrolled_students] / [max_students] students enrolled""",
    tools=[get_available_classes],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
    ),
)


enrollment_agent = Agent(
    name="enrollment_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Specialist agent for enrolling a student in a class. This agent modifies enrollment records and requires explicit student confirmation.",
    instruction="""You are an enrollment specialist.
Your primary directives are:
1. Use the `enroll_in_class` tool to enroll a student in a specified class when requested.
2. Always report the exact success or error message returned by the tool directly.
3. Students can only enroll in classes where enrolled_students < max_students.
4. Use the tool exactly as provided.""",
    tools=[FunctionTool(enroll_in_class, require_confirmation=True)],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
    ),
)


root_agent_instruction = """You are a helpful college student assistant dedicated to helping students search, check, schedule, and enroll in classes.

Your primary directives are:
1. Welcome students and identify how you can help them.
2. Delegate course listing, searching, looking up, and retrieving class details/schedules to the `catalog_agent`.
3. Delegate enrollment requests to the `enrollment_agent`.

CRITICAL RULES FOR RESPONDING:
1. Maintain a professional, polite, and helpful student-assistant tone.
2. Do not attempt to look up classes or perform enrollments yourself. Always delegate to the respective specialist agents (`catalog_agent` or `enrollment_agent`).
3. Never reference any external real-world universities or scheduling rules outside of our local catalog."""


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="A helpful college student assistant coordinator that helps with scheduling and enrolling in classes.",
    instruction=root_agent_instruction,
    sub_agents=[catalog_agent, enrollment_agent],
    before_agent_callback=populate_session_context,
    before_model_callback=academic_input_guardrail,
    after_model_callback=pii_output_guardrail,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.4,
    ),
)


# =====================================================================
# 4. APP WITH RESUMABILITY AND PLUGINS
# =====================================================================

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[AcademicPolicyPlugin(), ActionOutcomeLoggingPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=20,  # summarize every 20 events
        overlap_size=3,  # include last 3 events in next window for continuity
        summarizer=AsyncLlmEventSummarizer(llm=Gemini(model="gemini-flash-latest")),
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,  # only cache if context exceeds this
        ttl_seconds=1800,  # cache lifetime (default 1800)
        cache_intervals=10,  # re-cache every N invocations
    ),
)
