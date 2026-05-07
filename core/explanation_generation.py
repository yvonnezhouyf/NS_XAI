import asyncio
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Module-level singleton clients
_sync_client = None
_async_client = None

def _get_sync_client():
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI()
    return _sync_client

def _get_async_client():
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client


async def generate_explanation_async(llm_input: str, prompt_id: str) -> str:
    """Async version of generate_explanation_with_openai."""
    import time as _t
    _start = _t.time()
    print(f"  [async] Starting LLM call for prompt {prompt_id[-8:]}...", flush=True)
    try:
        client = _get_async_client()
        response = await client.responses.create(
            prompt={"id": prompt_id},
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": llm_input}]
            }],
            text={"format": {"type": "text"}},
            reasoning={}
        )

        print(f"  [async] LLM call for prompt {prompt_id[-8:]} done in {_t.time()-_start:.1f}s", flush=True)

        if response and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    text = item.content[0].text
                    if response.status == 'incomplete':
                        print(f"⚠️  Warning: Response was truncated (status: incomplete)")
                        if text and len(text) > 50:
                            return text + " [truncated]"
                    return text
            return "Unable to generate explanation - no valid response from API."
        else:
            return "Unable to generate explanation - no valid response from API."

    except Exception as e:
        print(f"Error generating explanation (async): {e}")
        return f"Unable to generate explanation due to error: {str(e)}"


def generate_explanation_with_openai(llm_input: str, prompt_id: str) -> str:
    """
    Generates natural language explanations using OpenAI GPT.

    Args:
        llm_input: The formatted input containing query and PCTL analysis results
        prompt_id: The OpenAI prompt ID from the use case config

    Returns:
        Generated natural language explanation string
    """
    try:
        client = _get_sync_client()

        # Use chat completions for explanation generation
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
                    "text": llm_input
                    }
                ]
                }
            ],
            text={
                "format": {
                "type": "text"
                }
            },
            reasoning={}
        )

        # Extract the text content from the response object
        # output[0] is reasoning, output[1] is the actual message
        if response and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    # Found the message with content
                    text = item.content[0].text

                    # Check if response was incomplete (truncated)
                    if response.status == 'incomplete':
                        print(f"⚠️  Warning: Response was truncated (status: incomplete)")
                        print(f"   Reason: {response.incomplete_details.reason if hasattr(response, 'incomplete_details') else 'unknown'}")
                        # Return partial text anyway - it might still be useful
                        if text and len(text) > 50:  # If we got at least some meaningful text
                            return text + " [truncated]"

                    return text
            return "Unable to generate explanation - no valid response from API."
        else:
            return "Unable to generate explanation - no valid response from API."

    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"Unable to generate explanation due to error: {str(e)}"


def generate_explanation_streaming(llm_input: str, prompt_id: str):
    """Stream explanation tokens from OpenAI Responses API.

    Yields tuples of (event_type, data):
        ("delta", text_chunk)   — incremental text
        ("done", full_text)     — final assembled text
        ("error", error_msg)    — on failure
    """
    try:
        client = _get_sync_client()
        stream = client.responses.create(
            stream=True,
            prompt={"id": prompt_id},
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": llm_input}]
            }],
            text={"format": {"type": "text"}},
            reasoning={},
        )

        full_text = []
        for event in stream:
            if event.type == "response.output_text.delta":
                full_text.append(event.delta)
                yield ("delta", event.delta)
            elif event.type == "response.completed":
                break

        assembled = "".join(full_text)
        yield ("done", assembled)

    except Exception as e:
        print(f"Error streaming explanation: {e}")
        yield ("error", str(e))


