"""rag_eval
"""

# Workaround: langchain-community 0.4.x eliminou `chat_models.vertexai` /
# `llms.vertexai`, pero ragas 0.3.9 impórtaos ao cargar. Como non usamos
# VertexAI, abonda con rexistrar módulos stub antes de que ragas se importe.
import sys as _sys
import types as _types

for _mod, _attr in (("langchain_community.chat_models.vertexai", "ChatVertexAI"),
                    ("langchain_community.llms.vertexai", "VertexAI")):
    if _mod not in _sys.modules:
        _stub = _types.ModuleType(_mod)
        setattr(_stub, _attr, None)
        _sys.modules[_mod] = _stub

__version__ = "0.1"
