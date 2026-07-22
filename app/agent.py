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


def get_available_classes() -> dict:
    """Get the list of all available college classes.

    Returns:
        dict: A dictionary containing available classes.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    courses_file_path = os.path.join(current_dir, "courses.json")
    with open(courses_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def enroll_in_class(class_name: str) -> dict:
    """Enroll a student in a class.

    This function updates the enrollment count for the specified class in the course catalog.
    Students can only enroll in classes where the current number of enrolled students is less than the max students.

    Args:
        class_name: The name of the class to enroll in.

    Returns:
        dict: A dictionary containing the status, message, and updated course details.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    courses_file_path = os.path.join(current_dir, "courses.json")
    with open(courses_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    courses = data.get("courses", [])
    found_course = None
    for course in courses:
        if course.get("name") == class_name:
            found_course = course
            break

    if not found_course:
        # Case-insensitive fallback for robust matching
        for course in courses:
            if course.get("name", "").strip().lower() == class_name.strip().lower():
                found_course = course
                break

    if not found_course:
        return {
            "status": "error",
            "message": f"Class '{class_name}' was not found in the course catalog.",
        }

    enrolled = found_course.get("enrolled_students", 0)
    max_students = found_course.get("max_students", 0)

    if enrolled >= max_students:
        return {
            "status": "error",
            "message": f"Cannot enroll in '{found_course.get('name')}'. The class is full ({enrolled}/{max_students} students).",
        }

    found_course["enrolled_students"] = enrolled + 1

    with open(courses_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return {
        "status": "success",
        "message": f"Successfully enrolled in '{found_course.get('name')}'!",
        "course": found_course,
    }


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
