"""TiTiler utility functions."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geojson_pydantic.features import Feature

from numba import njit

from PIL import Image

from morecantile import TileMatrixSet

# This code is copied from marblecutter
#  https://github.com/mojodna/marblecutter/blob/master/marblecutter/stats.py
# License:
# Original work Copyright 2016 Stamen Design
# Modified work Copyright 2016-2017 Seth Fitzsimmons
# Modified work Copyright 2016 American Red Cross
# Modified work Copyright 2016-2017 Humanitarian OpenStreetMap Team
# Modified work Copyright 2017 Mapzen
class Timer(object):
    """Time a code block."""

    def __enter__(self):
        """Starts timer."""
        self.start = time.time()
        return self

    def __exit__(self, ty, val, tb):
        """Stops timer."""
        self.end = time.time()
        self.elapsed = self.end - self.start

    @property
    def from_start(self):
        """Return time elapsed from start."""
        return time.time() - self.start


def bbox_to_feature(
    bbox: Tuple[float, float, float, float], properties: Optional[Dict] = None,
) -> Feature:
    """Create a GeoJSON feature polygon from a bounding box."""
    return Feature(
        **{
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                    ]
                ],
            },
            "properties": {} or properties,
            "type": "Feature",
        }
    )


def data_stats(
    data: np.ma.array,
    categorical: bool = False,
    categories: Optional[List[float]] = None,
    percentiles: List[int] = [2, 98],
) -> List[Dict[Any, Any]]:
    """Returns statistics."""
    output: List[Dict[Any, Any]] = []
    percentiles_names = [f"percentile_{int(p)}" for p in percentiles]
    for b in range(data.shape[0]):
        keys, counts = np.unique(data[b].compressed(), return_counts=True)

        valid_pixels = float(np.ma.count(data[b]))
        masked_pixels = float(np.ma.count_masked(data[b]))
        valid_percent = round((valid_pixels / data[b].size) * 100, 2)
        info_px = {
            "valid_pixels": valid_pixels,
            "masked_pixels": masked_pixels,
            "valid_percent": valid_percent,
        }

        if categorical:
            # if input categories we make sure to use the same type as the data
            out_keys = (
                np.array(categories).astype(keys.dtype) if categories else keys
            )
            out_dict = dict(zip(keys.tolist(), counts.tolist()))
            output.append(
                {
                    "categories": {k: out_dict.get(k, 0) for k in out_keys.tolist()},
                    **info_px,
                },
            )
        else:
            percentiles_values = np.percentile(
                data[b].compressed(), percentiles
            ).tolist()

            output.append(
                {
                    "min": float(data[b].min()),
                    "max": float(data[b].max()),
                    "mean": float(data[b].mean()),
                    "count": float(data[b].count()),
                    "sum": float(data[b].sum()),
                    "std": float(data[b].std()),
                    "median": float(np.ma.median(data[b])),
                    "majority": float(
                        keys[counts.tolist().index(counts.max())].tolist()
                    ),
                    "minority": float(
                        keys[counts.tolist().index(counts.min())].tolist()
                    ),
                    "unique": float(counts.size),
                    **dict(zip(percentiles_names, percentiles_values)),
                    **info_px,
                }
            )

    return output


@njit()
def get_tile_children(zoom: int, x: int, y: int):
    return [
        (zoom + 1, x * 2, y * 2),
        (zoom + 1, x * 2+1, y * 2),
        (zoom + 1, x * 2+1, y * 2+1),
        (zoom + 1, x * 2, y * 2+1)
    ]

def build_availability(bounds: Tuple, max_zoom: int, tms: TileMatrixSet, min_zoom: int = 0):
    tiles = tms.tiles(*bounds, zooms=[*list(range(min_zoom, max_zoom))])
    availability: Dict[int, Dict[str, int]] = {}
    # print(list(tiles))
    for t in tiles:
        if not availability.get(t.z):
            availability[t.z] = {
                'startX': t.x,
                'startY': t.y,
                'endX': t.x,
                'endY': t.y
            }
        else:
            availability[t.z] = {
                'startX': t.x if t.x < availability[t.z]['startX'] else availability[t.z]['startX'],
                'startY': t.y if t.y < availability[t.z]['startY'] else availability[t.z]['startY'],
                'endX': t.x if t.x > availability[t.z]['endX'] else availability[t.z]['endX'],
                'endY': t.y if t.y > availability[t.z]['endY'] else availability[t.z]['endY'],
            }
    return [[availability[k]] for k in availability.keys()]


def resize_array(in_array: np.ndarray, new_size: Tuple[int, int]):
    return np.array(Image.fromarray(in_array).resize(new_size))