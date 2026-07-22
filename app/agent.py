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

import datetime
import json
import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, List


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
)


app = App(
    root_agent=root_agent,
    name="app",
)
