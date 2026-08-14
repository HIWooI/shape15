"""Serve frames as MJPEG over HTTP, to watch the demo from another machine.

X11 to a remote display needs an X server listening on TCP, which most desktops disable.
A browser needs nothing: open http://<host>:<port>/ and the stream appears.

    stream = Streamer(8080)
    stream.put(frame_bgr)
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

def _page(peer=None, button=False):
    """One tab, both views: the panels live in separate processes, so the page joins them.

    The second panel's host is whatever the browser used to reach us, which only the
    browser knows — so its src is filled in client-side rather than guessed here.
    """
    imgs = '<img src="/stream">'
    if peer:
        imgs += (f'<img id="p"><script>document.getElementById("p").src='
                 f'location.protocol+"//"+location.hostname+":{peer}/stream"</script>')
    if button:
        imgs += ('<button onclick="fetch(\'/calibrate\')">Calibrate</button>'
                 "<style>button{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);"
                 "font:600 18px sans-serif;padding:14px 32px;border:0;border-radius:8px;"
                 "background:#2b8;color:#fff;cursor:pointer}</style>")
    return (
        '<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">'
        "<style>body{margin:0;background:#111;display:flex;gap:4px;justify-content:center;"
        "align-items:center;height:100vh}img{max-width:49%;max-height:100vh}</style>" + imgs
    ).encode()


class Streamer:
    def __init__(self, port, quality=80, peer=None, on_calibrate=None):
        # A condition, not a spin: polling `frame is None` in a tight loop holds the GIL
        # hard enough to stop the producer from ever setting one — the stream stays empty
        # and the demo's own frame rate collapses.
        self.frame, self.seq, self.quality = None, 0, quality
        page = _page(peer, button=on_calibrate is not None)
        self.cond = threading.Condition()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass  # a request line per frame would drown the console

            def do_GET(self):
                if self.path == "/calibrate" and on_calibrate is not None:
                    on_calibrate()  # sets a flag; the demo loop does the work
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=f"
                )
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with streamer.cond:
                            streamer.cond.wait_for(lambda: streamer.seq != seen, timeout=5.0)
                            buf, seen = streamer.frame, streamer.seq
                        if buf is None:
                            continue
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                        self.wfile.write(buf)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed the tab

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def put(self, frame_bgr):
        ok, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.frame, self.seq = enc.tobytes(), self.seq + 1
                self.cond.notify_all()
