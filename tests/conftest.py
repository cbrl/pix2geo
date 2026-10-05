import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from pix2geo import Terrain, available_backends

LAT0 = 40.0
LON0 = -105.0
UTM = "EPSG:32613"  # UTM zone 13N, which contains (LAT0, LON0)

# PROJ conversions. The tests use them as references that are independent
# of the code under test.
_GEOCENTRIC = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

#: The largest range difference (m) of each backend from the Python backend.
#: The CuPy backend walks the quadtree in float32, so its hits move slightly.
BACKEND_ATOL = {"python": 1e-6, "numba": 1e-6, "cupy": 1e-2}


def ecef_reference(lat, lon, h):
    """WGS84 latitude, longitude (degrees) and height (m) to ECEF ``(..., 3)``, by PROJ."""
    x, y, z = _GEOCENTRIC.transform(*np.broadcast_arrays(lon, lat, h))
    return np.stack([x, y, z], axis=-1)


def geodetic_reference(points):
    """ECEF points ``(..., 3)`` to latitude, longitude and height, by PROJ.

    PROJ uses one Bowring step. Its height error is below 1 µm up to 10 km,
    but 8 mm at 1000 km. Thus use it only for points near the surface.
    """
    p = np.asarray(points, float)
    lon, lat, h = _GEOCENTRIC.transform(p[..., 0], p[..., 1], p[..., 2], direction="INVERSE")
    return lat, lon, h


def enu_axes_reference(lat, lon):
    """The east, north and up unit vectors at a point, in ECEF, as rows, by PROJ."""
    topo = Transformer.from_pipeline(
        f"+proj=topocentric +ellps=WGS84 +lat_0={lat:.17g} +lon_0={lon:.17g} +h_0=0"
    )

    # The inverse maps ENU points to ECEF. Far points keep the rounding of the
    # ECEF origin small against the length of each axis.
    far = 1e9 * np.eye(3)
    x, y, z = topo.transform(far[:, 0], far[:, 1], far[:, 2], direction="INVERSE")
    x0, y0, z0 = topo.transform(0.0, 0.0, 0.0, direction="INVERSE")
    return (np.stack([x, y, z], axis=-1) - [x0, y0, z0]) / 1e9


def hills(
    rows=600,
    cols=600,
    res=1.0 / 1200,
    lat_top=LAT0 + 0.25,
    lon_left=LON0 - 0.25,
    base=1500.0,
    amp=150.0,
):
    """A smooth synthetic DEM in EPSG:4326 around (LAT0, LON0)."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    h = base + amp * np.sin(xx / 37.0) * np.cos(yy / 53.0) + 0.4 * amp * np.sin((xx + yy) / 11.0)
    transform = Affine(res, 0.0, lon_left, 0.0, -res, lat_top)
    return h.astype(np.float32), transform


def utm_transform(rows, cols, spacing=30.0):
    """Affine transform of a UTM raster centered on (LAT0, LON0)."""
    x0, y0 = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(LON0, LAT0)
    return Affine(spacing, 0.0, x0 - cols * spacing / 2, 0.0, -spacing, y0 + rows * spacing / 2)


def utm_terrain(n):
    """A smooth synthetic n x n DEM in UTM zone 13N around (LAT0, LON0)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    h = (900 + 60 * np.sin(xx / 30) * np.cos(yy / 45)).astype(np.float32)
    return Terrain.from_array(h, utm_transform(n, n), UTM)


@pytest.fixture(scope="session")
def hills_terrain():
    h, tr = hills()
    return Terrain.from_array(h, tr, "EPSG:4326")


@pytest.fixture(scope="session")
def backends():
    return available_backends()


def pytest_generate_tests(metafunc):
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", available_backends())


def write_geotiff(path, heights, transform, crs="EPSG:4326", nodata=None):
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=heights.shape[0],
        width=heights.shape[1],
        count=1,
        dtype=heights.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(heights, 1)
    return path
