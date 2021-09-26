from typing import Optional, Tuple
import quantized_mesh_encoder
from quantized_mesh_encoder.extensions import ExtensionId, VertexNormalsExtension, WaterMaskExtension
import pymartini
from pymartini.util import rescale_positions
from io import BytesIO
import numpy as np
from titiler.core.resources.extensions import MetadataExtension
from morecantile import TileMatrixSet, Tile

from titiler.core.utils import build_availability, resize_array


def encode_quantized_mesh_tile(
    tile: Tile,
    grid: TileMatrixSet,
    elevation: np.ndarray,
    watermask: Optional[np.ndarray] = None,
    metadata: Optional[dict] = {},
    max_error: Optional[float] = 100.0,
    max_zoom: Optional[int] = 5,
    source_raster_size: Optional[Tuple[int, int]] = (32768, 16384),
    source_bounds: Optional[Tuple[float, float, float, float]] = (-180, -90, 180, 90)
    ) -> BytesIO:
    bounds = grid.bounds(tile)
    source_area = (
        abs(source_bounds[0] - source_bounds[2])*
        abs(source_bounds[1] - source_bounds[3])
    )

    current_area = (
        abs(bounds[0] - bounds[2])*
        abs(bounds[1] - bounds[3])
    )

    area_ratio = current_area / min(source_area, 0.0001)

    metadata.update({
        'available': [a for a in build_availability(bounds, max_zoom, grid, tile.z+1)],
        'geometricerror': (source_raster_size[0]*source_raster_size[1])/elevation.shape[0],
        'surfacearea': (source_raster_size[0]*source_raster_size[1])*area_ratio
    })
    
    elevation = resize_array(elevation, (elevation.shape[0]+1, elevation.shape[1]+1))

    martini = pymartini.martini.Martini(elevation.shape[0])
    tin = martini.create_tile(elevation)
    
    vertices, triangles = tin.get_mesh(max_error)
    vertices = rescale_positions(vertices, elevation, bounds = bounds, flip_y=True)

    extensions = [
        VertexNormalsExtension(positions=vertices, indices=triangles),
        MetadataExtension(data = metadata)]
    if isinstance(watermask, np.ndarray):
        extensions.append(
            WaterMaskExtension(data = watermask))

    content = BytesIO()
    quantized_mesh_encoder.encode(content, vertices, triangles, extensions=extensions)

    return content