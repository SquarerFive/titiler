import attr
from quantized_mesh_encoder.extensions import ExtensionBase, ExtensionId
from quantized_mesh_encoder.constants import EXTENSION_HEADER
from typing import Union, Dict
from struct import pack
import json

@attr.s(kw_only=True)
class MetadataExtension(ExtensionBase):
    id: ExtensionId = attr.ib(
        ExtensionId.METADATA,
        validator=attr.validators.instance_of(ExtensionId))
    data: Union[Dict, bytes] = attr.ib(
        validator=attr.validators.instance_of((dict, bytes)))

    def encode(self) -> bytes:
        encoded: bytes
        if isinstance(self.data, dict):
            # Minify output
            encoded = json.dumps(self.data, separators=(',', ':')).encode()
        elif isinstance(self.data, bytes):
            encoded = self.data

        buf = b''
        buf += pack(EXTENSION_HEADER['extensionId'], self.id.value)
        buf += pack(EXTENSION_HEADER['extensionLength'], len(encoded)+4)
        buf += pack('<I', len(encoded))
        buf += encoded

        return buf