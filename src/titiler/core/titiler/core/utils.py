"""TiTiler utility functions."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geojson_pydantic.features import Feature

from numba import njit

from PIL import Image

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

def get_tile_availability(tile: tuple, max_depth = 5):
    tiles = get_tile_children(*tile)

    min_tile = min(tiles)
    max_tile = max(tiles)

    end_x = max_tile[0]
    end_y = max_tile[1]

    start_x = min_tile[0]
    start_y = min_tile[1]

    results = []
    results.append({
        'endX': end_x,
        'endY': end_y,
        'startX': start_x,
        'startY': start_y
    })

    zoom = tile[0]+1

    for level in range(max_depth):
        zoom += 1

        start_x = start_x * 2
        start_y = start_y * 2
        end_x = end_x * 2 + 1
        end_y = end_y * 2 + 1

        results.append({
            'endX': end_x,
            'endY': end_y,
            'startX': start_x,
            'startY': start_y
        })
    
    return results

def resize_array(in_array: np.ndarray, new_size: Tuple[int, int]):
    return np.array(Image.fromarray(in_array).resize(new_size))