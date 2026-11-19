"""The online system (specs 016-018; plan phase 6): database, streaming engine, API.

``engine``, ``ingest``, ``db`` and ``schemas`` are torch-free; ``bundle`` (the real model) needs
torch and is imported only when the service loads a model.
"""
