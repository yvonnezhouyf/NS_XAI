import json
import os
import re
from typing import Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Valid evaluation levels
VALID_LEVELS = {'MICRO', 'SEQUENCE', 'TREND', 'MACRO', 'HUGS'}

# Module-level singleton client
_sync_client = None

def _get_sync_client():
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI()
    return _sync_client


def classify_query_with_openai(query: str, prompt_id: str) -> Tuple[int, Optional[str]]:
    """
    Classifies the user's query using a managed prompt on the OpenAI platform.

    The LLM is expected to return JSON format:
    {"type_id": int, "level": "MICRO"|"SEQUENCE"}

    Args:
        query: The user's natural language query
        prompt_id: The OpenAI prompt ID from the use case config

    Returns:
        Tuple of (type_id, level) where:
        - type_id: Integer label for the classification category (-1 if fails)
        - level: Evaluation level string ('MICRO', 'SEQUENCE') or None
    """
    try:
        client = _get_sync_client()

        # Try the managed prompt API
        response = client.responses.create(
            prompt={
                "id": prompt_id
            },
            input=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "input_text",
                    "text": query
                    }
                ]
                }
            ],
            text={
                "format": {
                "type": "text"
                }
            },
            reasoning={},
            max_output_tokens=2048
        )

        # Extract the text from the response object
        if response.output and response.output[0].content:
            classification_text = response.output[0].content[0].text.strip()

            # Parse as JSON
            type_id, level = _parse_json_response(classification_text)

            if type_id is not None:
                return type_id, level

        return -1, None  # Return -1, None if parsing fails

    except Exception as e:
        print(f"Error during classification: {e}")
        return -1, None


def _parse_json_response(text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse the response as JSON format.

    Expected format:
    {"type_id": 2, "level": "MICRO"}

    Args:
        text: Response text from LLM

    Returns:
        Tuple of (type_id, level) or (None, None) if not valid JSON
    """
    try:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)

            type_id = data.get('type_id')
            level = data.get('level')

            # Validate type_id is an integer
            if type_id is not None:
                type_id = int(type_id)

            # Validate level is a known value
            if level is not None:
                level = level.upper()
                if level not in VALID_LEVELS:
                    level = None

            return type_id, level

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None, None
