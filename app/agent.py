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
import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models import Gemini
from google.genai import types
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, List

# Context variables to track active session identifiers
active_session_id = contextvars.ContextVar("active_session_id", default=None)
active_user_id = contextvars.ContextVar("active_user_id", default=None)
active_app_name = contextvars.ContextVar("active_app_name", default=None)


class AsyncLlmEventSummarizer(LlmEventSummarizer):
    """An LLM-based event summarizer that executes compaction in an asynchronous background task.

    This avoids blocking the main conversation thread on the latent LLM summarization call.
    """

    async def maybe_summarize_events(
        self, *, events: List[Event]
    ) -> Optional[Event]:
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
                compacted_event = await super(AsyncLlmEventSummarizer, self).maybe_summarize_events(events=events)
                if compacted_event:
                    # 2. Append the compacted event asynchronously back to the session history
                    from app.app_utils.services import get_session_service
                    session_service = get_session_service()
                    session = await session_service.get_session(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id
                    )
                    if session:
                        await session_service.append_event(session=session, event=compacted_event)
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


root_agent_instruction = """You are a helpful college student assistant dedicated to helping students search, check, schedule, and enroll in classes.

Your primary directives are:
1. Use the `get_available_classes` tool to retrieve all course information.
2. Use the `enroll_in_class` tool when a student explicitly requests to enroll in a class.

CRITICAL RULES FOR RESPONDING:
1. STRICT GROUNDING: You MUST base your answers on the class information returned by the tools. Do NOT hallucinate, assume, or fabricate any details (such as professors, schedules, duration, or availability) not explicitly present in the data.
2. DATA-ONLY ANSWERS: If a student asks about or wants to enroll in a class, professor, or schedule that does not exist in the retrieved classes, state clearly and politely that the class/information is not available in the courses catalog.
3. CLEAR FORMATTING: When presenting course details, format them cleanly using bullet points or structured markdown:
   - **Course Name**: [name]
   - **Professor**: [professor]
   - **Start Times**: [start_times]
   - **Duration**: [duration_hrs] hour(s)
   - **Enrollment**: [enrolled_students] / [max_students] students enrolled
4. NO EXTERNAL KNOWLEDGE: Never reference any external real-world universities, courses, or scheduling rules outside of the provided tool data.
5. ENROLLMENT LIMITS: Students can only enroll in classes where the current number of enrolled students is less than the max students. Use the `enroll_in_class` tool to process enrollment, and report the success or error message returned by the tool directly to the student."""


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="A helpful college student assistant that helps with scheduling and enrolling in classes.",
    instruction=root_agent_instruction,
    tools=[get_available_classes, enroll_in_class],
    before_agent_callback=populate_session_context,
)


app = App(
    root_agent=root_agent,
    name="app",
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
