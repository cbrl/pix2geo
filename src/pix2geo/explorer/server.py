"""The HTTP server of the explorer: the page, its files and the JSON API.

The server uses only the standard library. ``GET /`` gives the page, ``GET
/static/<name>`` its files, and ``GET /api/session`` the start state. Each
``POST /api/<name>`` calls one method of :class:`~.session.Session` with the
JSON body. A lock runs one computation at a time, because the Numba
backend must not run two parallel kernels at once.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from ..backends import BackendUnavailableError
from ..camera import CameraIntrinsics
from ..geolocator import Geolocator
from ..pose import CameraPose
from ..terrain import Terrain
from .session import JSON, Session

__all__ = ["ExplorerServer", "make_server", "serve"]

#: The POST routes: ``/api/<name>`` calls the session method.
_ROUTES: dict[str, Callable[[Session, JSON], JSON]] = {
    "terrain": Session.terrain_mesh,
    "cast": Session.cast,
    "locate": Session.locate,
    "profile": Session.profile,
    "image": Session.image,
    "viewshed": Session.viewshed,
    "look_at": Session.look_at,
    "settings": Session.settings,
    "benchmark": Session.benchmark,
}

#: The content types of the page files.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}

#: The errors that come from a bad request. They give status 400, not 500.
_CLIENT_ERRORS = (ValueError, KeyError, TypeError, BackendUnavailableError)

#: The largest request body, in bytes.
_MAX_BODY = 1 << 24


class ExplorerServer(ThreadingHTTPServer):
    """A threaded HTTP server for one explorer session."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], session: Session) -> None:
        super().__init__(address, _Handler)
        self.session = session
        self.lock = threading.Lock()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Ignore a browser that closes its connection. Print the other errors."""
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)

    @property
    def url(self) -> str:
        """The address of the page."""
        host, port = self.server_address[:2]
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"

        return f"http://{host!s}:{port}/"


class _Handler(BaseHTTPRequestHandler):
    """Serves the page files and the JSON API of one :class:`ExplorerServer`."""

    protocol_version = "HTTP/1.1"

    @property
    def _server(self) -> ExplorerServer:
        assert isinstance(self.server, ExplorerServer)
        return self.server

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/session":
            with self._server.lock:
                self._send_json(self._server.session.info())
            return

        if path in ("/", "/index.html"):
            name = "index.html"
        elif path.startswith("/static/"):
            name = path[len("/static/") :]
        else:
            self._send_error(HTTPStatus.NOT_FOUND, f"no page at {path}")
            return

        data = _static_file(name)
        if data is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no file {name}")
            return

        suffix = name[name.rfind(".") :]
        self._send(HTTPStatus.OK, data, _CONTENT_TYPES.get(suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        # Read the body first. An unread body on a kept-alive connection would
        # become the start of the next request. A body that is too large
        # closes the connection instead.
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = -1
        if not 0 <= size <= _MAX_BODY:
            self.close_connection = True
            self._send_error(HTTPStatus.BAD_REQUEST, f"the body must have 0 to {_MAX_BODY} bytes")
            return
        data = self.rfile.read(size)

        path = urlsplit(self.path).path
        route = _ROUTES.get(path[len("/api/") :]) if path.startswith("/api/") else None
        if route is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no API at {path}")
            return

        try:
            body = json.loads(data or b"{}")
            if not isinstance(body, dict):
                raise TypeError("the request body must be a JSON object")
        except (ValueError, TypeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, f"bad request: {exc}")
            return

        # Run the request. A bad request gives 400, a bug gives 500.
        try:
            with self._server.lock:
                result = route(self._server.session, body)
        except _CLIENT_ERRORS as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, _message(exc))
            return
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, _message(exc))
            return

        self._send_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        """Do not log each request. The errors go to stderr."""

    # ---- Responses ----------------------------------------------------------

    def _send_json(self, value: JSON) -> None:
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self._send(HTTPStatus.OK, data, "application/json")

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        data = json.dumps({"error": message}).encode()
        self._send(status, data, "application/json")

    def _send(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def _static_file(name: str) -> bytes | None:
    """The bytes of a page file, or None. Only plain names in ``static/`` work."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None

    path = resources.files(__package__) / "static" / name
    return path.read_bytes() if path.is_file() else None


def _message(exc: BaseException) -> str:
    """The text of an error for the page."""
    text = str(exc) or type(exc).__name__
    return text.strip("'\"") if isinstance(exc, KeyError) else text


def make_server(session: Session, host: str = "127.0.0.1", port: int = 8765) -> ExplorerServer:
    """Make the server of a session. Port 0 picks a free port.

    When the port is in use, the server uses a free port instead.
    """
    try:
        return ExplorerServer((host, port), session)
    except OSError:
        if port == 0:
            raise
        return ExplorerServer((host, 0), session)


def serve(
    geolocator: Geolocator | Terrain,
    camera: CameraIntrinsics,
    pose: CameraPose,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    block: bool = True,
    **session_options: Any,
) -> ExplorerServer:
    """Start the explorer for a geolocator, a camera and a pose.

    Parameters
    ----------
    geolocator:
        The geolocator, or a terrain for a default geolocator.
    camera, pose:
        The view that the page starts with.
    host, port:
        The server address. Port 0 picks a free port.
    open_browser:
        Open the page in the default browser.
    block:
        Serve until Ctrl+C, then stop. With ``block=False``, the server runs
        in a background thread, and the function returns at once. Call
        ``server.shutdown()`` to stop it. This suits a notebook.
    session_options:
        Other arguments of :class:`~.session.Session`, for example
        ``fallback_height`` or ``max_rays``.
    """
    session = Session(geolocator, camera, pose, **session_options)
    server = make_server(session, host, port)
    print(f"pix2geo explorer: {server.url}")

    if open_browser:
        webbrowser.open(server.url)

    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return server
