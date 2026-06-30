"""Repeated-play experiments over the core mechanism.

Orchestration only. These modules drive the core auction over many rounds with optional
communication and history, and emit traces for the eval layer. The core never imports them;
import-linter enforces that direction.
"""
