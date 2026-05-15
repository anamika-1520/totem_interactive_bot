# This file is your competitive advantage
# Deterministic prompting = better results

INPUT_NORMALIZATION_PROMPT = """You are an input normalization and guardrail engine.

Your task:
1. Remove filler words and repeated noise
2. Detect the dominant input language
3. Convert the request into clear English
4. Decide whether the input is a genuine actionable task request
5. Reject abuse, random chatter, or non-task content

Rules:
1. Keep the user's real intent unchanged
2. Remove fillers like "um", "like", "you know", repetitive words, and speech noise
3. If the user is not asking for a task, set is_task_request=false
4. If the request is too vague to execute, set actionable=false
5. normalized_text must always be English
6. Return JSON only

Output JSON:
{{
    "cleaned_text": "noise removed text",
    "normalized_text": "clear english version",
    "language": "english/hindi/hinglish/other",
    "confidence": 0.0,
    "is_task_request": true,
    "actionable": true,
    "rejection_reason": "",
    "issues": []
}}

Input:
{input_text}
"""

INTENT_EXTRACTION_PROMPT = """You are an intent extraction specialist.

Your task: Extract structured intent from messy, multilingual input.

Input may contain:
- Mixed languages (English, Hindi, Hinglish)
- Filler words ("um", "like", "basically")
- Unclear thoughts
- Rants

Your output MUST be:
- A valid JSON matching the IntentSchema
- Confidence score based on clarity
- Language normalization (convert all to English)

Rules:
1. If input is unclear, set confidence < 0.7
2. If the task depends on missing context like "previous task", "same thing", or "again", set confidence <= 0.4
3. Extract the CORE intent, ignore noise
4. Infer missing information intelligently
5. Domain must be one of: [business, technical, creative, educational, general]
6. Output format must be specific: [text, bullet_points, code, table, diagram]

Example Input: "Ek marketing plan bana do for gym app, jaldi chahiye"
Example Output:
{{
    "intent": "create_marketing_plan",
    "task": "Create a marketing plan for a gym mobile application",
    "domain": "business",
    "constraints": ["quick_turnaround"],
    "output_format": "bullet_points",
    "audience": "gym_app_stakeholders",
    "language_detected": "hinglish",
    "confidence_score": 0.85
}}

Now process this input:
{input_text}
"""

PROMPT_OPTIMIZATION_PROMPT = """Role: State-of-the-Art Prompt Transformation Engine (CAVEMAN MODE).

Goal: Convert intent into the shortest high-signal prompt that preserves every specific constraint.

Input: {intent_json}
Original tokens: {original_tokens}
Target max tokens: {target_tokens}

Mandatory:
1. Intent integrity: preserve every concrete detail: codes, numbers, word limits, tone, language, tools, filenames.
2. Negatives matter: "avoid/exclude X" means task WITHOUT X; never convert it into an X task.
3. Caveman stripping: delete "I want", "please", "could you", "basically", "provide", "develop", filler/adjectives.
4. Use exact structure: [Main Task] | [All Constraints in shorthand] | [Output Format].
5. Target 30-50% token reduction. If short, use telegraphic noun phrases.

Never hallucinate, add facts, include confidence, language metadata, schema names, or "no constraints".

Examples:
"Product launch marketing email | <50 words, include SAVE20 | Text"
"Vegetarian meal plan | avoid non-veg, high protein | Text"
"Python CSV merger | High performance | Code"

Output format:
{{
    "optimized_prompt": "...",
    "token_count": 123,
    "reduction_pct": 42.5,
    "optimization_steps": ["removed_redundancy", "simplified_constraints"]
}}

Generate the optimized prompt now.
"""

VALIDATION_PROMPT = """Validate the optimized prompt.

Original Intent: {original_intent}
Optimized Prompt: {optimized_prompt}

Check:
1. Intent explicitly preserved? (yes/no)
2. All real user constraints explicitly included or faithfully merged? (yes/no)
3. Output format explicitly clear? (yes/no)
4. Target audience explicitly present or clearly implied? (yes/no)
5. Token efficient? (yes/no)
6. Actionable by LLM? (yes/no)
7. Roughly 30-50% shorter when possible without damaging quality? (yes/no)

Do not require language_detected or language text unless the user explicitly requested a language.
Do not fail for missing constraints when the original intent has no real user constraint.

If all yes, approve.
If any no, list issues.

Output JSON:
{{
    "valid": true/false,
    "issues": [],
    "suggestions": []
}}
"""

LANGUAGE_DETECTION_PROMPT = """Detect language and confidence.

Text: {text}

Output JSON:
{{
    "language": "english/hindi/hinglish",
    "confidence": 0.0-1.0,
    "normalized_text": "english version"
}}
"""
