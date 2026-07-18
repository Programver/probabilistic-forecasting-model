"""AI-assisted insight layer: deterministic driver attribution, offline
template narratives, and an optional LLM polish step (Gemini / OpenAI /
Anthropic) that degrades gracefully when no key or network is available.

Nothing in this package is imported by the offline scored pipeline
(generate_features.py / predict.py); it powers the app, docs and demo only.
"""
