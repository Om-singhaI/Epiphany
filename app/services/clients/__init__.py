"""External service clients for the Epiphany agent loop.

Each client wraps one integration (Fivetran, Elastic, Vertex AI/Gemini, GitLab)
behind a small async API. Every client is constructed from :class:`app.config.Settings`
and transparently falls back to deterministic **simulation mode** when its
credentials are not configured, so the orchestrator runs end-to-end regardless.
"""
