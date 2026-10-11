import base64
import http.client
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from conftest import LAT0, LON0, ecef_reference, enu_axes_reference
from pix2geo import CameraIntrinsics, CameraPose, Geolocator
from pix2geo.explorer import Session, make_server
from pix2geo.explorer._geometry import image_border, pixel_grid
from pix2geo.explorer.session import _camera_json, _near_miss, _pose_json

CAM = CameraIntrinsics.from_fov(640, 480, hfov=60)
POSE = CameraPose.from_euler(LAT0, LON0, 2500.0, yaw=30, pitch=-40, roll=5)
LEVEL = CameraPose.from_euler(LAT0, LON0, 2500.0, yaw=0, pitch=0, roll=0)
PIXELS = [[320.0, 240.0], [100.0, 400.0], [600.0, 50.0]]


def view(pose=POSE, camera=CAM, **extra):
    """The camera and pose part of a request, as the page sends it."""
    return {"camera": _camera_json(camera), "pose": _pose_json(pose), **extra}


def f32(text):
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


@pytest.fixture(scope="module")
def session(hills_terrain):
    return Session(Geolocator(hills_terrain, backend="python"), CAM, POSE)


@pytest.fixture(scope="module")
def geo(hills_terrain):
    return Geolocator(hills_terrain, backend="python")


def test_info(session, hills_terrain):
    info = session.info()
    json.dumps(info, allow_nan=False)

    assert info["terrain"]["shape"] == list(hills_terrain.shape)
    assert info["origin"] == {"lat": LAT0, "lon": LON0}
    assert info["backends"][0] == "auto"
    assert info["pose"]["yaw"] == pytest.approx(30.0)
    assert info["camera"]["dist"] == [0.0] * 8


def test_terrain_mesh(session, hills_terrain):
    mesh = session.terrain_mesh({"size": 100})
    rows, cols = mesh["rows"], mesh["cols"]
    assert max(rows, cols) <= 100

    # The corner posts of the whole DEM, in the scene frame.
    pos = f32(mesh["positions"]).reshape(rows, cols, 3)
    r, c = (n - 1 for n in hills_terrain.shape)
    lon, lat = hills_terrain.grid_to_lonlat(c, r)
    post = ecef_reference(lat, lon, float(hills_terrain.heights[r, c]))
    np.testing.assert_allclose(pos[-1, -1], session.to_enu(post), atol=0.01)
    assert f32(mesh["height"]).shape == (rows * cols,)

    # Bounds of the whole DEM give the corner posts too, and far bounds fail.
    whole = session.terrain_mesh(
        {"extent": "bounds", "bounds": list(hills_terrain.bounds_wgs84()), "size": 100}
    )
    np.testing.assert_allclose(f32(whole["positions"])[-3:], pos[-1, -1], atol=0.01)
    with pytest.raises(ValueError, match="do not overlap"):
        session.terrain_mesh({"extent": "bounds", "bounds": [10.0, 10.0, 11.0, 11.0]})

    # The area around the view is smaller than the whole DEM.
    near = session.terrain_mesh(view(extent="view", size=100))
    w, s, e, n = near["bounds"]
    W, S, E, N = mesh["bounds"]
    assert (e - w) * (n - s) < (E - W) * (N - S)


def test_cast_matches_the_geolocator(session, geo):
    res = session.cast(view(pixels=PIXELS, ids=[7, 8, 9], grid=50, edge_points=6))
    json.dumps(res, allow_nan=False)

    expected = geo.pixel_to_geo(CAM, POSE, PIXELS)
    assert [p["id"] for p in res["picks"]] == [7, 8, 9]
    for k, p in enumerate(res["picks"]):
        assert p["status"] == "hit"
        assert p["lat"] == pytest.approx(expected.lat[k], abs=1e-9)
        assert p["range"] == pytest.approx(expected.range[k], abs=1e-3)
        np.testing.assert_allclose(p["end"], session.to_enu(expected.ecef[k]), atol=2e-3)

    # The camera in the scene frame, with the ENU axes of PROJ. The pose is at
    # the frame origin, so the camera is 2500 m straight up.
    enu = enu_axes_reference(LAT0, LON0)
    np.testing.assert_allclose(res["camera"]["position"], [0.0, 0.0, 2500.0], atol=1e-3)
    np.testing.assert_allclose(res["camera"]["axes"], POSE.camera_to_ecef().T @ enu.T, atol=1e-8)

    assert 40 <= len(res["grid"]["pixels"]) <= 50
    assert len(res["border"]["ends"]) == 4 * 6 + 1
    assert res["border"]["area"] > 0


def test_border_and_grid():
    n = 8
    border, corners = image_border(CAM, n)

    # A closed loop, clockwise from the top-left image corner.
    assert len(border) == 4 * n + 1
    np.testing.assert_array_equal(border[0], border[-1])
    np.testing.assert_array_equal(
        border[corners],
        [[-0.5, -0.5], [639.5, -0.5], [639.5, 479.5], [-0.5, 479.5]],
    )

    # The rays through the middles of the right and left edges span the hfov.
    right, left = CAM.pixel_to_ray(*border[[n + n // 2, 3 * n + n // 2]].T)
    assert np.degrees(np.arccos(right @ left)) == pytest.approx(60.0)

    # A regular grid of at most 50 pixels, centered on the image.
    grid = pixel_grid(CAM, 50)
    assert 40 <= len(grid) <= 50
    np.testing.assert_allclose(grid.mean(axis=0), [CAM.cx, CAM.cy])


def test_cast_detail(session):
    res = session.cast(view(pixels=PIXELS, grid=0))
    pick = res["picks"][0]

    assert pick["utm"]["zone"] == "13N"
    assert all(g > 0 for g in pick["gsd"])
    assert 0 < pick["grazing"] < 90
    assert pick["sensitivity"] > 0
    assert pick["depression"] == pytest.approx(40.0, abs=5.0)

    # The image center, and the geodesic path through the three picks.
    assert res["center"]["u"] == CAM.cx
    m = res["measure"]
    assert len(m["segments"]) == 2
    assert m["length"] == pytest.approx(sum(s["distance"] for s in m["segments"]))
    assert m["area"] > 0


def test_locate_finds_the_pixel(session):
    pick = session.cast(view(pixels=[[200.0, 300.0]]))["picks"][0]

    # A ray straight down onto the hit of the pick.
    above = np.add(pick["end"], [0.0, 0.0, 3000.0])
    res = session.locate(view(origin=above.tolist(), direction=[0.0, 0.0, -1.0]))
    assert res["hit"] and res["visible"] and res["in_image"]
    np.testing.assert_allclose(res["pixel"], [200.0, 300.0], atol=0.05)

    # A ray behind the camera.
    cam = session.cast(view())["camera"]
    back = -np.asarray(cam["axes"][2])
    ground = np.asarray(cam["position"]) + 3000.0 * back
    ground[2] += 3000.0
    res = session.locate(view(origin=ground.tolist(), direction=[0.0, 0.0, -1.0]))
    assert res["hit"] and not res["in_front"] and not res["visible"]


def test_profile(session):
    pick = session.cast(view(pixels=[[320.0, 240.0]]))["picks"][0]
    res = session.profile(view(pixel=[320.0, 240.0]))
    json.dumps(res, allow_nan=False)

    assert res["hit"]["range"] == pytest.approx(pick["range"], abs=1e-3)
    assert res["hit"]["height"] == pytest.approx(pick["alt"], abs=0.01)

    # The ray starts at the camera, above the terrain, and ends past the hit.
    assert res["t"][0] == 0 and res["t"][-1] > pick["range"]
    assert res["ray"][0] > res["terrain"][0]


def test_image_layers(session, geo):
    res = session.image(view(step=16, layers=["range", "status", "gsd", "shade", "sensitivity"]))
    rows, cols = res["rows"], res["cols"]
    assert (rows, cols) == (480 // 16, 640 // 16)

    expected = geo.image_to_geo(CAM, POSE, 16)
    np.testing.assert_allclose(
        f32(res["layers"]["range"]["data"]).reshape(rows, cols),
        expected.range,
        rtol=1e-6,
    )
    status = np.frombuffer(base64.b64decode(res["layers"]["status"]["data"]), np.uint8)
    np.testing.assert_array_equal(status, expected.hit.ravel().astype(np.uint8))
    assert np.nanmin(f32(res["layers"]["gsd"]["data"])) > 0
    shade = f32(res["layers"]["shade"]["data"])
    assert np.nanmin(shade) >= 0 and np.nanmax(shade) <= 1

    with pytest.raises(ValueError, match="unknown layers"):
        session.image(view(layers=["nope"]))


def test_viewshed(session, geo):
    mesh = session.terrain_mesh({"size": 80})
    res = session.viewshed(view(mesh=mesh["id"]))
    visible = np.frombuffer(base64.b64decode(res["visible"]), np.uint8).astype(bool)
    uv = f32(res["uv"]).reshape(-1, 2)

    assert 0 < res["count"] == visible.sum() < visible.size
    assert res["area"] > 0
    assert CAM.contains(uv[visible, 0], uv[visible, 1]).all()

    with pytest.raises(ValueError, match="unknown terrain mesh"):
        session.viewshed(view(mesh="nope"))


def test_look_at_centers_the_target(session):
    pick = session.cast(view(pixels=[[80.0, 420.0]]))["picks"][0]
    angles = session.look_at(view(target=[pick["lat"], pick["lon"], pick["alt"]]))
    spec = {**_pose_json(POSE), **angles}

    res = session.locate(
        view(
            CameraPose.from_euler(LAT0, LON0, 2500.0, spec["yaw"], spec["pitch"], spec["roll"]),
            origin=np.add(pick["end"], [0.0, 0.0, 2000.0]).tolist(),
            direction=[0.0, 0.0, -1.0],
        )
    )
    np.testing.assert_allclose(res["pixel"], [CAM.cx, CAM.cy], atol=0.05)


def test_settings(hills_terrain):
    session = Session(hills_terrain, CAM, POSE)
    out = session.settings({"backend": "python", "max_range": 1000.0})
    assert out == {"backend": "python", "max_range": 1000.0, "max_segment_length": 500.0}
    assert session.cast(view(pixels=PIXELS))["picks"][0]["status"] == "miss"

    with pytest.raises(ValueError):
        session.settings({"backend": "nope"})


def test_http_server(session):
    server = make_server(session, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = server.url

    def get(path):
        with urllib.request.urlopen(url + path) as r:
            return r.status, r.headers["Content-Type"], r.read()

    def post(path, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, method="POST")
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())

    def status(fn, *args):
        with pytest.raises(urllib.error.HTTPError) as err:
            fn(*args)
        return err.value.code

    try:
        code, kind, page = get("")
        assert code == 200 and kind.startswith("text/html") and b"main.js" in page
        assert get("static/main.js")[1].startswith("text/javascript")
        assert json.loads(get("api/session")[2])["origin"]["lat"] == LAT0

        code, res = post("api/cast", view(pixels=PIXELS))
        assert code == 200 and len(res["picks"]) == 3

        assert status(get, "static/..%2Fsession.py") == 404
        assert status(get, "nope") == 404
        assert status(post, "api/nope", {}) == 404
        assert status(post, "api/cast", b"not json") == 400
        assert status(post, "api/cast", {"pixels": [[1, 2, 3]]}) == 400

        # A 404 reads the body, so the next request on the connection works.
        conn = http.client.HTTPConnection(*server.server_address[:2])
        conn.request("POST", "/api/nope", body=json.dumps(view()))
        response = conn.getresponse()
        assert response.status == 404 and response.read()
        conn.request("POST", "/api/cast", body=json.dumps(view(pixels=PIXELS)))
        response = conn.getresponse()
        assert response.status == 200 and len(json.loads(response.read())["picks"]) == 3
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
