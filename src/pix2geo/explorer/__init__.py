"""Interactive explorer: the terrain, the camera and the rays, in 3D and from the camera.

The explorer is a small web app. A local HTTP server runs pix2geo for each ray
cast, and a browser page draws the scene with three.js. The page shows:

- a 3D view of the terrain, the camera, its frustum and a fan of rays;
- the view from the camera, with the exact intrinsics and lens distortion;
- the picked pixels, their ray casts, and a terrain profile along each ray.

Start it from the command line::

    python -m pix2geo.explorer dem.tif --geoid egm96
    python -m pix2geo.explorer               # a synthetic DEM

or from Python::

    from pix2geo.explorer import serve

    serve(geo, cam, pose)                   # blocks until Ctrl+C
    server = serve(geo, cam, pose, block=False)
    server.shutdown()

The page loads three.js from a CDN (jsDelivr), so the browser needs a network
connection the first time. The server needs only the standard library.
"""

from .server import ExplorerServer, make_server, serve
from .session import Session

__all__ = ["ExplorerServer", "Session", "make_server", "serve"]
