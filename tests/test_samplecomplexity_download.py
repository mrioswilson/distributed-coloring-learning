"""Exercise recovery against real local HTTP disconnects and range responses."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import runpy
import threading

import pytest

download=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/fetch-sample-complexity.py"))["download_archive"]


@contextmanager
def serve(mode):
    payload=bytes(range(256))*97
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            start,end=map(int,re.fullmatch(r"bytes=(\d+)-(\d+)",self.headers["Range"]).groups())
            seen.append((start,end))
            body=payload if mode=="ignore" else payload[start:end+1]
            self.send_response(200 if mode=="ignore" else 206)
            if mode!="ignore":
                offset=start+1 if mode=="wrong" else start
                self.send_header("Content-Range",f"bytes {offset}-{end}/{len(payload)}")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Connection","close");self.end_headers()
            if mode=="empty": body=b""
            elif mode=="disconnect" and len(seen)==1: body=body[:1100]
            self.wfile.write(body)
            self.close_connection=True
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield f"http://127.0.0.1:{server.server_port}/archive",payload,seen
    finally: server.shutdown();server.server_close();thread.join()


def test_resume_retains_prefix_and_recovers_after_disconnect(tmp_path):
    with serve("disconnect") as (url,payload,seen):
        cache=tmp_path/"archive.part";cache.write_bytes(payload[:3000])
        download(url,cache,len(payload),chunk_bytes=4096,timeout=1)
        assert cache.read_bytes()==payload
        assert seen[0][0]==3000 and seen[1][0]>3000


def test_server_ignoring_range_restarts_without_appending(tmp_path):
    with serve("ignore") as (url,payload,seen):
        cache=tmp_path/"archive.part";cache.write_bytes(payload[:3000])
        download(url,cache,len(payload),chunk_bytes=4096,timeout=1)
        assert cache.read_bytes()==payload and len(seen)==1


def test_wrong_range_preserves_cached_bytes(tmp_path):
    with serve("wrong") as (url,payload,seen):
        cache=tmp_path/"archive.part";cache.write_bytes(payload[:3000])
        with pytest.raises(RuntimeError,match="Unexpected HTTP range"):
            download(url,cache,len(payload),chunk_bytes=4096,timeout=1)
        assert cache.read_bytes()==payload[:3000]


def test_retries_are_bounded_and_partial_cache_is_retained(tmp_path):
    with serve("empty") as (url,payload,seen):
        cache=tmp_path/"archive.part";cache.write_bytes(payload[:3000])
        with pytest.raises(RuntimeError,match="retries exhausted"):
            download(url,cache,len(payload),chunk_bytes=4096,timeout=1,attempts=2)
        assert len(seen)==2 and cache.read_bytes()==payload[:3000]
