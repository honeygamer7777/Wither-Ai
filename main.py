import os
import discord
from discord.ext import commands
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# Load environment variables from .env file (for local/external hosting)
load_dotenv()

# Initializing environment variables for Replit AI Integrations
# OpenAI (GPT-5)
AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")

# Anthropic (Claude 4.0 Sonnet)
AI_INTEGRATIONS_ANTHROPIC_API_KEY = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
AI_INTEGRATIONS_ANTHROPIC_BASE_URL = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")

# Gemini (Gemini 3 Pro)
AI_INTEGRATIONS_GEMINI_API_KEY = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
AI_INTEGRATIONS_GEMINI_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")

# Groq (Backup Brain)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Model Names
OPENAI_MODEL = "gpt-5"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-3-pro-preview"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize clients
openai_client = OpenAI(
    api_key=AI_INTEGRATIONS_OPENAI_API_KEY,
    base_url=AI_INTEGRATIONS_OPENAI_BASE_URL
)

anthropic_client = Anthropic(
    api_key=AI_INTEGRATIONS_ANTHROPIC_API_KEY,
    base_url=AI_INTEGRATIONS_ANTHROPIC_BASE_URL
)

gemini_client = genai.Client(
    api_key=AI_INTEGRATIONS_GEMINI_API_KEY,
    http_options={
        'api_version': '',
        'base_url': AI_INTEGRATIONS_GEMINI_BASE_URL
    }
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def is_rate_limit_error(exception: BaseException) -> bool:
    """Check if the exception is a rate limit or quota violation error."""
    error_msg = str(exception)
    return (
        "429" in error_msg
        or "RATELIMIT_EXCEEDED" in error_msg
        or "quota" in error_msg.lower()
        or "rate limit" in error_msg.lower()
        or (hasattr(exception, "status_code") and getattr(exception, "status_code", None) == 429)
        or (hasattr(exception, "status") and getattr(exception, "status", None) == 429)
    )

def get_backup_response(prompt: str) -> str:
    """Get response from Groq (Llama-3) if main models fail."""
    if not groq_client:
        return "Backup brain not configured."
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a backup AI coding assistant. The primary models are offline, so you are helping now. Provide high-quality code snippets."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content or "Backup brain failed."
    except Exception as e:
        return f"Backup brain error: {str(e)}"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_gpt5_response(prompt: str) -> str:
    """GPT-5 for general coding and responses."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert AI coding agent powered by GPT-5. Always provide high-quality, production-ready code snippets."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=8192
    )
    return response.choices[0].message.content or "Error with GPT-5"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_claude_response(prompt: str) -> str:
    """Claude 4.0 Sonnet for logic and fixing code."""
    message = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        system="You are an expert at code logic and debugging powered by Claude 4.0 Sonnet. Focus on fixing errors and improving structural integrity.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text if message.content[0].type == "text" else "Error with Claude"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_gemini_response(prompt: str) -> str:
    """Gemini 3 for design and UI features."""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"system_instruction": "You are a design and UI expert powered by Gemini 3. Provide beautiful CSS, UI components, and design guidance."}
    )
    return response.text or "Error with Gemini"

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    if bot.user:
        print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        print('------')

@bot.command(name="code")
async def code(ctx, *, prompt: str):
    """Routing logic to different AI models based on prompt keywords."""
    async with ctx.typing():
        try:
            p_lower = prompt.lower()
            
            # Route based on keywords
            if any(k in p_lower for k in ["design", "ui", "css", "html", "style", "frontend"]):
                try:
                    response = get_gemini_response(prompt)
                    provider = "Gemini 3"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else: raise e
            elif any(k in p_lower for k in ["fix", "error", "bug", "logic", "reasoning", "refactor"]):
                try:
                    response = get_claude_response(prompt)
                    provider = "Claude 4.0 Sonnet"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else: raise e
            else:
                try:
                    response = get_gpt5_response(prompt)
                    provider = "GPT-5"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else: raise e

            full_response = f"**[AI: {provider}]**\n{response}"

            # Handle Discord's 2000 character limit
            if len(full_response) > 2000:
                for i in range(0, len(full_response), 2000):
                    await ctx.send(full_response[i:i+2000])
            else:
                await ctx.send(full_response)

        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")
            print(f"Error: {e}")

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if token:
        bot.run(token)
    else:
        print("Error: DISCORD_TOKEN not set.")


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if token:
        bot.run(token)
    else:
        print("Error: DISCORD_TOKEN not set.")
