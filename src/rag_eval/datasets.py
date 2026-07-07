from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class JSONLDataset(AbstractDataset):
    """Le/escribe ficheiros JSONL como lista de cadeas (unha por liña), UTF-8."""

    def __init__(self, filepath: str, **kwargs: Any) -> None:
        self._filepath = Path(filepath)

    def _load(self) -> list[str]:
        with self._filepath.open(encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def _save(self, data: list[str]) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._filepath.open("w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}


class KnowledgeGraphDataset(AbstractDataset):
    """Le/escribe un KnowledgeGraph de RAGAS dende/cara a un ficheiro JSON."""

    def __init__(self, filepath: str, **kwargs: Any) -> None:
        self._filepath = Path(filepath)

    def _load(self):
        from ragas.testset.graph import KnowledgeGraph
        return KnowledgeGraph.load(self._filepath)

    def _save(self, data) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data.save(self._filepath)

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}


class TextDataset(AbstractDataset):
    """Le/escribe unha única cadea de texto plano, UTF-8. Úsase para rutas escalares (p.ex. chroma_path)."""

    def __init__(self, filepath: str, **kwargs: Any) -> None:
        self._filepath = Path(filepath)

    def _load(self) -> str:
        return self._filepath.read_text(encoding="utf-8").strip()

    def _save(self, data: str) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filepath.write_text(data, encoding="utf-8")

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}
