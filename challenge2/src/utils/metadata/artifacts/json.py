import json
from typing import Any
from challenge2.src.utils.metadata.artifacts.base import Artifact
from tempfile import NamedTemporaryFile


class JsonArtifact(Artifact):
    def __init__(
        self,
        name: str,
        uri: str | None = None,
        metadata: dict[str, Any] = {}
    ):
        if uri is None:
            uri = NamedTemporaryFile().name 
        super().__init__(name=name, metadata=metadata, uri=uri)

    def content(self):
        with open(self.uri, 'r') as f:
            return json.load(f)
