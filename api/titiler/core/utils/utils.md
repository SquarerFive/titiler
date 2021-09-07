# Module titiler.core.utils

TiTiler utility functions.

None

## Functions

    
### bbox_to_feature

```python3
def bbox_to_feature(
    bbox: Tuple[float, float, float, float],
    properties: Union[Dict, NoneType] = None
) -> geojson_pydantic.features.Feature
```

    
Create a GeoJSON feature polygon from a bounding box.

    
### data_stats

```python3
def data_stats(
    data: <function array at 0x7f229e2d7310>,
    categorical: bool = False,
    categories: Union[List[float], NoneType] = None,
    percentiles: List[int] = [2, 98]
) -> List[Dict[Any, Any]]
```

    
Returns statistics.

## Classes

### Timer

```python3
class Timer(
    /,
    *args,
    **kwargs
)
```

#### Instance variables

```python3
from_start
```

Return time elapsed from start.