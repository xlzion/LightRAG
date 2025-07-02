# src/llm_handler.py
import httpx # Import httpx
import json
from config import LLM_API_BASE_URL, LLM_MODEL_IDENTIFIER

def generate_llm_response(prompt_context: str, original_query: str, llm_model_identifier: str = LLM_MODEL_IDENTIFIER) -> str:
    user_content = f"Given the context: {prompt_context}, please answer the question: {original_query}"

    data_payload = {
        "model": llm_model_identifier,
        "messages": [{"role": "user", "content": user_content}],
        "stream": False,
        "max_tokens": 300, # Ensure this is appropriate for your RAG responses
        "temperature": 0.5
    }
    
    json_payload_to_send = json.dumps(data_payload)

    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
    }

    target_url = f"{LLM_API_BASE_URL}/chat/completions"

    try:
        print(f"\nAttempting to send RAG request to LLM (using httpx) at: {target_url}")
        print(f"Using Model: {llm_model_identifier}")
        print("--- Payload Snippet Being Sent (User Content - first 500 chars) ---")
        print(f"  User Content: {user_content[:500]}...")
        print("--------------------------------------------------------------------")

        # Use httpx.Client, ensuring trust_env=False to ignore system proxies for localhost
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                target_url,
                headers=headers,
                content=json_payload_to_send, # httpx uses 'content' for raw body or 'json' for dict
                timeout=90
            )
        
        print(f"\n--- LLM Server Response (httpx) ---")
        print(f"Status Code: {response.status_code}")
        print("Response Body (Raw Text - first 500 chars):")
        print(response.text[:500])
        print("--- End LLM Server Response ---")

        response.raise_for_status() 
        
        response_data = response.json()
        
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message")
            if message and "content" in message:
                assistant_message = message["content"]
                print("\nSUCCESS: Parsed assistant message from LLM.")
                return assistant_message.strip()
            else:
                print(f"\nWarning: 'message' or 'content' missing in LLM response choice: {json.dumps(response_data, indent=2)}")
                return "Error: LLM response structure malformed (missing content)."
        else:
            print(f"\nWarning: LLM response JSON structure unexpected (no choices): {json.dumps(response_data, indent=2)}")
            return "Error: Could not parse LLM response structure (no choices)."

    except httpx.HTTPStatusError as http_err: # Specific httpx exception for HTTP errors
        print(f"\nCRITICAL HTTPStatusError (httpx): {http_err}")
        print(f"Response body from LLM: {http_err.response.text[:500]}")
        return f"Error: HTTP error ({http_err.response.status_code}) connecting to LLM."
    except httpx.RequestError as req_err: # Specific httpx exception for request issues
        print(f"\nCRITICAL RequestError (httpx): {req_err}")
        return "Error: Could not connect/send request to the Language Model (httpx)."
    except json.JSONDecodeError as json_err:
        print(f"\nCRITICAL JSON DECODE ERROR: {json_err}")
        return "Error: Failed to decode LLM response (not valid JSON)."
    except Exception as e:
        print(f"\nCRITICAL UNEXPECTED ERROR in LLM Handler: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return "Error: An unexpected critical issue occurred with the LLM interaction."