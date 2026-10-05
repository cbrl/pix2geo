# pix2geo

pix2geo finds the geographic coordinate of the terrain that a camera pixel sees. It casts a ray
from the pixel into a digital elevation model (DEM) and returns the first intersection as WGS84
latitude, longitude, and height.

## Install

```bash
pip install -e .
pip install -e .[numba]   # multi-core CPU backend
pip install -e .[cuda13]  # GPU backend for CUDA 13 (cupy-cuda13x)
pip install -e .[cuda12]  # GPU backend for CUDA 12 (cupy-cuda12x)
pip install -e .[test]
```

Numba and CuPy are optional. The library falls back to the next best backend when one is missing.

CuPy has one wheel for each CUDA major version. Choose the extra for the CUDA Toolkit on the
system, or manually install one of the `cupy-cudaNNx` packages. Without a CUDA Toolkit on the
system, install `cupy-cuda13x[ctk]` or `cupy-cuda12x[ctk]`. This extra gets the CUDA libraries from
PyPI, so only the GPU driver is necessary. 

## Quick start

```python
from pix2geo import CameraIntrinsics, CameraPose, Terrain, Geolocator

terrain = Terrain.from_file("n40_w106_1arc_v3.tif", geoid="egm96")   # SRTM heights are MSL
cam = CameraIntrinsics.from_fov(1920, 1080, hfov=60)
pose = CameraPose.from_euler(40.0, -105.3, 2500.0, yaw=45, pitch=-30, roll=0)

geo = Geolocator(terrain, backend="auto")
res = geo.pixel_to_geo(cam, pose, (960, 540))
print(float(res.lat), float(res.lon), float(res.alt), bool(res.hit))

grid = geo.image_to_geo(cam, pose, step=8)      # every 8th pixel of the frame
u, v, visible = geo.geo_to_pixel(cam, pose, res.lat, res.lon, res.alt, check_occlusion=True)
```

`pixel_to_geo` accepts `(2,)` or `(..., 2)` arrays. Every field of the `GeoResult` has the input
batch shape:

| Field | Meaning |
|---|---|
| `lat`, `lon` | WGS84 geodetic degrees. NaN on a miss. |
| `alt` | Height above the WGS84 ellipsoid (m). |
| `alt_msl` | Height above the geoid (m), when the terrain has a geoid. |
| `ecef` | ECEF position (m), shape `(..., 3)`. |
| `range` | Slant range from the camera (m). |
| `hit` | True when the ray hit the DEM. |

Rays that miss the DEM result in a NaN value. With `fallback_height=h`, they hit the ellipsoid
grown by `h` meters instead, and `hit` stays False.

## Conventions

These conventions are the usual cause of wrong answers, so read them before you use the library.

### Pixels

`u` is the column and `v` is the row. The origin is the top-left of the image. Integer values are
pixel centers, like in OpenCV, so the top-left image corner is `(-0.5, -0.5)`.

### Camera frame

The camera frame is the same optical frame as OpenCV: +X right, +Y down, +Z forward.

### Orientation

`CameraPose.from_euler` takes aerospace yaw, pitch and roll. These are intrinsic Z-Y'-X'' rotations
from local NED to a body frame (forward, right, down).

- Yaw is the heading, clockwise from north.
- Positive pitch raises the nose.
- Positive roll lowers the right side.
- Without a `mount`, the optical axis is the body forward axis. Thus `(yaw, 0, 0)` looks at the horizon, and `pitch=-90` looks straight down.

To combine a platform attitude with gimbal angles, pass
`mount=euler_rotation(g_yaw, g_pitch, g_roll)`. This is the rotation from the camera head to the
platform body.

`CameraPose.from_rotation(lat, lon, alt, R, frame="NED" | "ENU" | "ECEF")` takes any rotation that
maps camera optical-frame vectors into that frame. `R` can be a SciPy `Rotation`, a 3x3 matrix or
an `(x, y, z, w)` quaternion.

`CameraPose.look_at(...)` points the optical axis at a target.

### Altitude

Multiple altitude references are supported via `CameraPose`'s `alt_ref` parameter.

| `alt_ref=` | Meaning |
|---|---|
| `"ellipsoid"` (default) | Height above the WGS84 ellipsoid. |
| `"geoid"` | Height above MSL. Requires a terrain geoid. |
| `"agl"` | Height above the DEM below the camera. |

### Heights and geoids

The library works in WGS84 ellipsoidal heights. Most DEM products (DTED, SRTM, Copernicus GLO-30)
store MSL heights, so give the `Terrain` a geoid:

| `geoid=` | Model |
|---|---|
| `None` | The DEM heights are already ellipsoidal. |
| `"egm96"`, `"egm2008"`, `"EPSG:5773"` | PROJ geoid grid. PROJ must have the grid (`projsync`, or `PyprojGeoid(..., network=True)`). |
| `"us_nga_egm96_15.tif"` | A geoid grid file, read with rasterio (`RasterGeoid`). |
| `-17.3` | A constant undulation. This is good enough for a small area. |

The loader adds the undulation to the DEM once at load time. A DTED tile with MSL heights and no
geoid results in a warning.

## Elevation models

```python
Terrain.from_file("tile.dt2", geoid="egm96")                       # DTED
Terrain.from_file("big.tif", bounds=(w, s, e, n), margin=0.01)     # read only a window
Terrain.from_files(["n40w106.dt1", "n40w105.dt1"], geoid="egm96")  # mosaic of tiles
Terrain.from_array(heights, affine, "EPSG:32613", nodata=-9999)    # in memory, any CRS
Terrain.from_file("dem.tif", fill_nodata="interpolate")            # fill holes
Terrain.from_file(WarpedVRT(src, crs="EPSG:4326"))                 # an open rasterio dataset
```

The surface is the bilinear interpolation of the elevation posts at the pixel centers. Cells with a
nodata post are treated as holes, and rays go through them. Use `fill_nodata` to close holes.

### Nodata

These values are considered nodata:
- NaN
- Infinity
- The masked entries of a NumPy masked array
- The `nodata=` value
- The raster nodata mask of a file

### Open datasets

`from_file` and `from_files` accept an open rasterio dataset in place of a path. The caller owns
that dataset, so the loader does not close it.

## How it works

1. The intrinsics turn each pixel into a unit ray in the camera frame. The pose rotates the ray
  into ECEF.
2. The kernel clips each ray to the height band of the DEM with two ray / ellipsoid intersections.
  Empty space above the terrain costs nothing, even from orbit.
3. The kernel cuts the clipped ray into segments of at most `max_segment_length` (500 m by
  default). For each segment end, it converts ECEF to geodetic (Bowring's method, two iterations)
  and then to grid coordinates `(column, row, height)`.
   - A WGS84 geographic raster (DTED, SRTM, Copernicus) maps with its exact affine transform.
   - A projected raster (UTM and others) maps through a lon/lat lookup table that pyproj fills. The
     table spacing is about 4 raster pixels.
4. For each grid-space segment, the kernel walks a max quadtree (i.e. maximum mipmap) of the DEM,
  depth-first and near-to-far:
   - A slab test clips the segment to the node footprint.
   - If the lowest point of the clipped segment is above the node maximum, the kernel skips the
     whole node.
   - At a leaf cell, the kernel solves the exact segment / bilinear-patch intersection. This is a
     quadratic equation in the segment parameter.
   - The near-to-far order makes the first leaf hit the nearest hit, so the ray stops there.
5. The kernel maps the hit back onto the exact ECEF ray, and it converts the hit point to latitude,
  longitude and height.

Steps 2 to 5 run in the kernel, one thread per ray. The host only makes the rays (step 1) and
collects the results.

## Backends

| Name | Requires | Notes |
|---|---|---|
| `numba` | Numba | Multi-threaded. The code compiles on first use and goes into a disk cache. |
| `cupy` | CuPy, NVIDIA GPU | One CUDA thread per ray. `cupyx.jit` compiles the kernel to CUDA. Uses 32-bit floats for some math. |
| `python` | - | The same algorithm without compilation. Use it for a few rays or as a reference. |
| `auto` | varies | Batches of 20,000 or more rays use CuPy (if available). Other batches use Numba, then CuPy, then Python. |

`pix2geo.available_backends()` lists the backends that work on the system. Set
`PIX2GEO_DISABLE_NUMBA=1` or `PIX2GEO_DISABLE_CUPY=1` to hide a backend.

A backend that is requested by name and cannot run raises `BackendUnavailableError`. Its
`__cause__` is the original error, such as `ImportError` from a Numba that does not match NumPy.
The CuPy backend compiles its kernel when the backend is requested, so a missing CUDA compiler
notifies immediately. `auto` falls back to the first available backend, so itnever raises an error.

All three backends run the same Python kernel source (`backends/_kernel.py`). Numba compiles it for
the CPU and `cupyx.jit` compiles it to CUDA.

Due to the significantly lower 64-bit performance on many GPUs (often 1/64 of 32-bit performance),
the CuPy backend walks the quadtree in float32 and int32, with grid coordinates relative to each
segment. The geodetic math stays float64, as ECEF coordinates in float32 have poor precision.

## Accuracy

- Straight segments in grid space approximate the slightly curved image of a straight ECEF line.
  The height error is at most `L**2 / (8 R)`, which comes out to 5 mm for `L = 500 m`. The range
  error is the height error divided by the sine of the grazing angle. Make `max_segment_length`
  smaller for very shallow rays.
- The geodetic conversion (Bowring's method, in the kernel and on the host) is within 1e-10 degrees
  and 1 µm in height of the exact coordinates, from 500 m below the surface to 1000 km above it (tested).
- The conversion from geodetic coordinates to ECEF is within 1e-8 m of PROJ (tested).
- The projected-CRS lookup table agrees with pyproj to better than 0.001 pixel (tested).
- The grown-ellipsoid clipping errs by less than 3 cm for heights below 20 km. The clip band has a
  1 m margin.
- The CuPy backend walks the quadtree in float32, so its hits differ slightly from the CPU
  backends. In the benchmark views, the median difference is 0.03 to 0.08 mm, and 99.9 % of rays
  differ by less than 1.5 mm, with no ray changing between hit and miss. DEM accuracy remains the
  largest error source by an extreme factor.

The DEM is almost always the largest error source. For example, SRTM and Copernicus have vertical
errors on the order of 1 m.

## Performance

These results come from `python benchmarks/bench_backends.py` with the following configuration:

- **Terrain:** 4096 x 4096 DEM (13 quadtree levels)
- **Image Size:** 1920 x 1080 (2,073,600 rays)
- **CPU:** AMD Ryzen 7 7700x
- **GPU:** RTX 5070 Ti

| View | numba | cupy | python |
|---|---|---|---|
| nadir | 0.25 s (8.2 M rays/s) | 0.11 s (19.1 M rays/s) | 11.5 k rays/s |
| oblique (pitch -20°, 30 km max range) | 0.93 s (2.2 M rays/s) | 0.12 s (17.4 M rays/s) | 4.7 k rays/s |

## Command line

```bash
pix2geo --dem n40_w106_1arc_v3.tif --geoid egm96 --radius 30 \
    --lat 40.0 --lon -105.3 --alt 2500 --yaw 45 --pitch -30 --roll 0 \
    --size 1920 1080 --hfov 60 --pixel 960 540 --pixel 0 0 \
	--json
```

`--radius` reads only the DEM within that many kilometers of the camera. Run `pix2geo --help` to
see all the options.
