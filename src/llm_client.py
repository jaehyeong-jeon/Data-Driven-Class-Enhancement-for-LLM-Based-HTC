import time
import asyncio
from openai import OpenAI, AsyncOpenAI

class LLMClient:
    """
    Unified LLM Client for both Sync and Async calls.
    Handles retry logic, stats recording, and flexible message formatting.
    """
    def __init__(self, model, stats=None):
        self.model = model
        self.stats = stats
        try:
            self.client = OpenAI()
            self.aclient = AsyncOpenAI()
        except Exception:
            self.client = None
            self.aclient = None
            print("\n[WARNING] OpenAI API Key not found. Falling back to Mock Response for testing.\n")

    def _prepare_messages(self, prompt, input_text=None, messages=None):
        if messages:
            return messages
        
        msgs = [{"role": "user", "content": prompt}]
        if input_text:
            msgs.append({"role": "user", "content": input_text})
        return msgs

    def call(self, prompt=None, input_text=None, messages=None, verbose=False, max_tokens=1000):
        """Synchronous GPT call."""
        msgs = self._prepare_messages(prompt, input_text, messages)
        
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    n=1,
                    messages=msgs
                )
                if getattr(r.choices[0].message, "content", None):
                    if self.stats:
                        self.stats.record(r.usage)
                    return r.choices[0].message.content
                
                if r.choices[0].finish_reason == "content_filter":
                    return "none_class"
            except Exception as e:
                if attempt == 4:
                    print(f"\n[sync call] FAILED after 5 attempts: {e}")
            if attempt < 4:
                time.sleep(1)
        return ""

    async def acall(self, prompt=None, input_text=None, messages=None, verbose=False, max_tokens=1000):
        """Asynchronous GPT call."""
        msgs = self._prepare_messages(prompt, input_text, messages)
        
        for attempt in range(5):
            try:
                resp = await self.aclient.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    n=1,
                    messages=msgs
                )
                if getattr(resp.choices[0].message, "content", None):
                    if self.stats:
                        self.stats.record(resp.usage)
                    return resp.choices[0].message.content
                
                if resp.choices[0].finish_reason == "content_filter":
                    return "none_class"
            except Exception as e:
                wait = 0.5 * (attempt + 1)
                if attempt == 4:
                    print(f"\n[async call] FAILED after 5 attempts: {e}")
                    break
                await asyncio.sleep(wait)
        return ""
