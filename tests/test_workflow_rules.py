import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

from backend.main import (
    _apply_contextual_memory,
    _intent_from_task_choice,
    _multiple_task_intent,
    _needs_clarification,
    _normalize_intent,
    _prepare_session_input,
)
from fastapi import HTTPException

from backend.services import add_token_metrics, optimize_prompt_tokens


def test_multiple_unrelated_tasks_trigger_clarification():
    intent = _multiple_task_intent(
        "Create a plan for developing a gym mobile application and a pasta recipe"
    )

    assert intent["intent"] == "clarify_multiple_tasks"
    assert intent["confidence_score"] == 0.5
    assert "Pasta Recipe" in intent["detected_tasks"]
    assert "Gym App Development" in intent["detected_tasks"]
    assert _needs_clarification(intent, intent["confidence_score"])


def test_same_session_choice_becomes_clear_intent():
    intent = _intent_from_task_choice("Pasta Recipe")

    assert intent["task"] == "Provide a pasta recipe"
    assert intent["domain"] == "culinary"
    assert intent["confidence_score"] == 0.9


def test_pasta_recipe_intent_is_normalized():
    intent = _normalize_intent(
        {
            "intent": "provide_recipe",
            "task": "Provide a pasta recipe",
            "domain": "general",
            "constraints": [],
            "output_format": "text",
            "audience": "home_cooks",
            "language_detected": "english",
            "confidence_score": 0.9,
        }
    )

    assert intent["domain"] == "culinary"


def test_gym_marketing_example_is_deterministic():
    result = optimize_prompt_tokens(
        {
            "intent": "create_marketing_plan",
            "task": "Create a marketing plan for a gym mobile application",
            "domain": "business",
            "constraints": ["quick_turnaround"],
            "output_format": "bullet_points",
            "audience": "gym_app_stakeholders",
            "language_detected": "hinglish",
            "confidence_score": 0.85,
        },
        "Ek marketing plan bana do for gym app.",
    )

    assert result["optimized_prompt"] == (
        "Create a 3-step marketing plan for a gym app.\n"
        "Output: bullet points\n"
        "Constraint: under 100 words"
    )


def test_short_inputs_show_clarified_token_change():
    result = add_token_metrics(
        {"optimized_prompt": "Provide a pasta recipe for home cooks.\nOutput: text"},
        "Provide a pasta recipe",
    )

    assert result["optimized_tokens"] > result["original_tokens"]
    assert result["token_reduction"].startswith("clarified")


def test_contextual_memory_resolves_unit_test_followup():
    import backend.main as main

    main.recent_completed_intent = {
        "task": "Write a Python function to calculate the area of a circle"
    }
    intent = _apply_contextual_memory(
        "ab iska unit test bhi likh do",
        {
            "intent": "clarify_task",
            "task": "ab iska unit test bhi likh do",
            "domain": "general",
            "constraints": [],
            "output_format": "text",
            "audience": "general_users",
            "language_detected": "hinglish",
            "confidence_score": 0.45,
        },
    )

    assert intent["task"] == "Write unit tests for the Python function that calculates the area of a circle"
    assert intent["domain"] == "technical"
    assert intent["output_format"] == "code"


def test_unsafe_requests_are_rejected_before_intent_confirmation():
    try:
        _prepare_session_input("How to bypass a computer password?")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "Unauthorized or harmful request rejected" in exc.detail
    else:
        raise AssertionError("Unsafe request was not rejected")


def test_python_draw_circle_keeps_code_context():
    result = optimize_prompt_tokens(
        {
            "intent": "generate_code",
            "task": "Create a Python script to draw a circle",
            "domain": "technical",
            "constraints": [],
            "output_format": "code",
            "audience": "developers",
            "language_detected": "english",
            "confidence_score": 0.95,
        },
        "Write a Python code to draw a circle.",
    )

    assert result["optimized_prompt"] == "Python circle drawing | Code"
    assert 30 <= result["reduction_pct"] <= 50


def test_beginner_pasta_prompt_hits_reduction_goal():
    result = optimize_prompt_tokens(
        {
            "intent": "provide_recipe",
            "task": "Provide a simple pasta recipe for a beginner",
            "domain": "culinary",
            "constraints": ["beginner_level"],
            "output_format": "text",
            "audience": "home_cook",
            "language_detected": "english",
            "confidence_score": 0.9,
        },
        "Provide me a pasta recipe, I am a beginner",
    )

    assert result["optimized_prompt"] == "Beginner pasta | Text"
    assert 30 <= result["reduction_pct"] <= 50


def test_python_file_merger_hits_reduction_goal():
    result = optimize_prompt_tokens(
        {
            "intent": "merge_files",
            "task": "Merge files with Python",
            "domain": "technical",
            "constraints": [],
            "output_format": "code",
            "audience": "developers",
            "language_detected": "english",
            "confidence_score": 0.9,
        },
        "Write a Python code to merge files.",
    )

    assert result["optimized_prompt"] == "Python file merger | Code"
    assert 30 <= result["reduction_pct"] <= 50
