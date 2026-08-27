"""Backward-compatible ASGI entry for uvicorn main:app."""

from anabelle.app import app

__all__ = ["app"]
