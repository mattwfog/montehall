"""The brain layer: tokenized observation streams for the game-state model.

Contract: docs/cv-brain-token-contract.md. Vision is a
primitive; every sensor is an observation channel writing into one token
schema (store.schemas.TOKENS_SCHEMA). Model verdicts never enter the
stream.
"""
