from typing import Any
from challenge2.src.utils.metadata.store import store
from challenge2.src.utils.metadata.base import MetadataType, MetadataDefinition, MetadataObject
from ml_metadata.proto import metadata_store_pb2


class ArtifactType(MetadataType, metadata_store_pb2.ArtifactType):
    def register(self) -> int:
        return store.put_artifact_type(self)


class ArtifactDefinition(MetadataDefinition, metadata_store_pb2.Artifact):
    def register(self) -> int:
        return store.put_artifacts([self])[0]


class Artifact(MetadataObject):
    def __init__(self, name: str, uri: str, metadata: dict[str, Any]):
        super().__init__(
            name=name,
            metadata=metadata,
            type_object=ArtifactType,
            def_object=ArtifactDefinition,
            uri=uri
        )
        self.uri = uri
