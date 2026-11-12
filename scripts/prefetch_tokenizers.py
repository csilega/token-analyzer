# scripts/prefetch_tokenizers.py
"""Build-time step: load every supported tokenizer once, so its data is baked into the image."""
from token_analyzer.registry import SUPPORTED_MODELS, get_adapter

for model in SUPPORTED_MODELS:
    adapter = get_adapter(model)   # same loader production uses
    adapter.encode("warm-up")      # force any lazy loading to finish now
    print(f"prefetched {model}  version={adapter.version}")