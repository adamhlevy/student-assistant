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

import json
import os
import shutil

import pytest

import app.agent
from app.agent import enroll_in_class, get_available_classes


@pytest.fixture
def restore_courses():
    """Fixture to backup and restore the courses.json file after each test."""
    current_dir = os.path.dirname(app.agent.__file__)
    courses_path = os.path.join(current_dir, "courses.json")
    backup_path = courses_path + ".backup"

    # Make a backup
    shutil.copy(courses_path, backup_path)
    yield courses_path
    # Restore the backup
    if os.path.exists(backup_path):
        shutil.move(backup_path, courses_path)


def test_successful_enrollment(restore_courses) -> None:
    """Test enrolling in a class with available seats succeeds."""
    # Find an available class
    classes = get_available_classes()
    courses = classes.get("courses", [])

    available_course = None
    for course in courses:
        if course.get("enrolled_students", 0) < course.get("max_students", 0):
            available_course = course
            break

    assert available_course is not None, "No available course found for testing"

    course_name = available_course["name"]
    initial_enrollment = available_course["enrolled_students"]

    # Perform enrollment
    res = enroll_in_class(course_name)
    assert res["status"] == "success"
    assert "Successfully enrolled" in res["message"]
    assert res["course"]["enrolled_students"] == initial_enrollment + 1

    # Verify persistence
    updated_classes = get_available_classes()
    updated_courses = updated_classes.get("courses", [])
    updated_course = next(c for c in updated_courses if c["name"] == course_name)
    assert updated_course["enrolled_students"] == initial_enrollment + 1


def test_enrollment_class_not_found(restore_courses) -> None:
    """Test enrolling in a non-existent class returns an error."""
    res = enroll_in_class("Intro to Quantum Supercomputing with Cats")
    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_enrollment_class_full(restore_courses) -> None:
    """Test enrolling in a full class returns an error."""
    courses_path = restore_courses

    # Load and force a class to be full for testing
    with open(courses_path, encoding="utf-8") as f:
        data = json.load(f)

    courses = data.get("courses", [])
    assert len(courses) > 0, "No courses found to set full"

    # Mark the first course as full
    test_course = courses[0]
    test_course["enrolled_students"] = test_course["max_students"]

    with open(courses_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Attempt enrollment
    res = enroll_in_class(test_course["name"])
    assert res["status"] == "error"
    assert "full" in res["message"]

    # Ensure enrollment count did not change
    assert test_course["enrolled_students"] == test_course["max_students"]
