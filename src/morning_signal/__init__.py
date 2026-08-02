"""Morning Signal — auto-generated daily briefing podcast.

Generates a podcast script via Claude with web search, converts it to audio
with a TTS engine (Amazon Polly or Google Chirp3 HD), publishes the MP3 + RSS
feed to S3, and optionally sends a success/failure notification via Telegram.
"""

__version__ = "0.2.1"

__all__ = ["__version__"]
