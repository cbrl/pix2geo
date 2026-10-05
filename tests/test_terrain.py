import numpy as np
import pytest
from affine import Affine

from conftest import LAT0, LON0, UTM, hills, utm_transform, write_geotiff
from pix2geo import ConstantGeoid, HeightPyramid, RasterGeoid, Terrain


def test_pyramid_is_max_quadtree():
    rng = np.random.default_rng(3)
    h = rng.uniform(0, 100, size=(37, 53)).astype(np.float32)
    h[10, 10] = np.nan
    pyr = HeightPyramid(h)
    cell = pyr.levels[0]
    assert cell.shape == (36, 52)
    # A cell with a nodata corner is a hole.
    assert np.isneginf(cell[9:11, 9:11]).all()
    ref = np.nanmax(np.stack([h[:-1, :-1], h[:-1, 1:], h[1:, :-1], h[1:, 1:]]), axis=0)
    ok = np.isfinite(cell)
    np.testing.assert_array_equal(cell[ok], ref[ok])
    for k in range(1, pyr.num_levels):
        lv, below = pyr.levels[k], pyr.levels[k - 1]
        for i in range(lv.shape[0]):
            for j in range(lv.shape[1]):
                block = below[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                assert lv[i, j] == block.max()
    assert pyr.levels[-1].shape == (1, 1)
    assert pyr.levels[-1][0, 0] == np.nanmax(h)
    # Flat storage matches the levels.
    for k, lv in enumerate(pyr.levels):
        o = pyr.offsets[k]
        np.testing.assert_array_equal(pyr.data[o : o + lv.size], lv.ravel())


def test_height_at_matches_posts(hills_terrain):
    t = hills_terrain
    lon, lat = t.grid_to_lonlat(np.array([10.0, 250.0]), np.array([20.0, 300.0]))
    np.testing.assert_allclose(
        t.height_at(lat, lon), [t.heights[20, 10], t.heights[300, 250]], rtol=1e-6
    )
    # Half way between two posts is the mean.
    lon, lat = t.grid_to_lonlat(10.5, 20.0)
    assert t.height_at(lat, lon) == pytest.approx(t.heights[20, 10:12].mean(), rel=1e-6)
    assert np.isnan(t.height_at(LAT0 + 5, LON0))


def test_from_file_and_bounds(tmp_path):
    h, tr = hills()
    path = write_geotiff(tmp_path / "dem.tif", h, tr)
    full = Terrain.from_file(path)
    np.testing.assert_array_equal(full.heights, h)
    part = Terrain.from_file(path, bounds=(LON0 - 0.05, LAT0 - 0.05, LON0 + 0.05, LAT0 + 0.05))
    assert part.shape[0] < h.shape[0] and part.shape[1] < h.shape[1]
    lat = np.linspace(LAT0 - 0.04, LAT0 + 0.04, 7)
    lon = np.linspace(LON0 - 0.04, LON0 + 0.04, 7)
    np.testing.assert_allclose(part.height_at(lat, lon), full.height_at(lat, lon), rtol=1e-6)


def test_nodata_and_fill(tmp_path):
    h, tr = hills(rows=50, cols=50)
    h[20:25, 20:25] = -9999
    path = write_geotiff(tmp_path / "holes.tif", h, tr, nodata=-9999)
    t = Terrain.from_file(path)
    assert np.isnan(t.heights[22, 22])
    t2 = Terrain.from_file(path, fill_nodata=1234.0)
    assert t2.heights[22, 22] == 1234.0
    t3 = Terrain.from_file(path, fill_nodata="interpolate")
    assert np.isfinite(t3.heights).all()


def test_from_files_mosaic(tmp_path):
    h, tr = hills(rows=200, cols=300)
    left = write_geotiff(tmp_path / "a.tif", h[:, :150].copy(), tr)
    right = write_geotiff(tmp_path / "b.tif", h[:, 150:].copy(), tr * Affine.translation(150, 0))
    t = Terrain.from_files([left, right])
    np.testing.assert_array_equal(t.heights, h)


def test_constant_geoid_shifts_heights(hills_terrain):
    h, tr = hills()
    t = Terrain.from_array(h, tr, "EPSG:4326", geoid=-17.0)
    assert isinstance(t.geoid, ConstantGeoid)
    np.testing.assert_allclose(t.heights, h - 17.0, atol=1e-3)
    assert t.orthometric_height_at(LAT0, LON0) == pytest.approx(
        hills_terrain.height_at(LAT0, LON0), abs=1e-3
    )


def test_raster_geoid(tmp_path):
    # A global 1-degree geoid grid with N = lat + lon / 10.
    lon_c = np.arange(-180, 180) + 0.5
    lat_c = 90 - (np.arange(180) + 0.5)
    N = (lat_c[:, None] + lon_c[None, :] / 10.0).astype(np.float32)
    path = write_geotiff(tmp_path / "geoid.tif", N, Affine(1, 0, -180, 0, -1, 90))
    g = RasterGeoid(path)
    assert g.undulation(10.5, 20.5) == pytest.approx(10.5 + 2.05)
    assert g.undulation(-30.25, 100.75) == pytest.approx(-30.25 + 10.075)
    # Longitude wraps across the antimeridian: between -179.5 and 179.5.
    assert np.isfinite(g.undulation(0.5, 179.9))
    t = Terrain.from_array(*hills(), "EPSG:4326", geoid=str(path))
    raw = hills()[0]
    lon, lat = t.grid_to_lonlat(5.0, 7.0)
    assert t.heights[7, 5] - raw[7, 5] == pytest.approx(g.undulation(lat, lon), abs=1e-3)


def test_projected_crs():
    # A ramp from 800 m in UTM zone 13N around (LAT0, LON0).
    h = np.full((400, 400), 800.0, np.float32) + np.arange(400, dtype=np.float32)[None, :] * 0.5
    t = Terrain.from_array(h, utm_transform(400, 400), UTM)
    x, y = t.lonlat_to_grid(LON0, LAT0)
    assert x == pytest.approx(199.5, abs=1e-6) and y == pytest.approx(199.5, abs=1e-6)
    assert t.height_at(LAT0, LON0) == pytest.approx(800 + 199.5 * 0.5, abs=1e-4)
    w, s, e, n = t.bounds_wgs84()
    assert w < LON0 < e and s < LAT0 < n


def test_dted_tile(tmp_path):
    import rasterio
    import rasterio.shutil

    res = 1.0 / 120  # DTED level 0 below 50 degrees latitude
    h = (1500 + 10 * np.arange(121)[None, :] + np.arange(121)[:, None]).astype(np.int16)
    tr = Affine(res, 0, -105 - res / 2, 0, -res, 41 + res / 2)
    src = write_geotiff(tmp_path / "src.tif", h, tr)
    dted = tmp_path / "n40w105.dt0"
    rasterio.shutil.copy(src, dted, driver="DTED")
    with pytest.warns(UserWarning, match="MSL"):
        Terrain.from_file(dted)
    t = Terrain.from_file(dted, geoid=-15.0)
    lon, lat = t.grid_to_lonlat(30.0, 60.0)
    assert lon == pytest.approx(-105 + 30 * res) and lat == pytest.approx(41 - 60 * res)
    assert t.orthometric_height_at(lat, lon) == pytest.approx(1500 + 300 + 60, abs=1e-3)
    assert t.height_at(lat, lon) == pytest.approx(1860 - 15, abs=1e-3)


def test_nodata_sources():
    h, tr = hills(rows=50, cols=50)

    # np.array drops the mask of a masked array. The masked -9999 must not leak.
    masked = np.ma.array(h.copy(), mask=np.zeros(h.shape, bool))
    masked.data[5, 5] = -9999
    masked.mask[5, 5] = True
    t = Terrain.from_array(masked, tr)
    assert np.isnan(t.heights[5, 5])
    assert t.h_min > 0

    # Infinity and the nodata value are nodata too.
    raw = h.copy()
    raw[6, 6] = np.inf
    raw[7, 7] = -9999
    t = Terrain.from_array(raw, tr, nodata=-9999)
    assert np.isnan(t.heights[6, 6]) and np.isnan(t.heights[7, 7])
    assert np.isfinite(t.h_max)
    assert t.pyramid.levels[-1][0, 0] == t.h_max
