"""--stream (live TCP SBS-1 feed) tests.

Added v2.1.0. Covers the pieces of the --stream path that don't require
an actual long-running socket loop: the per-row SBS-1 ingestion (shared
with the file-based parser), the line-buffering socket reader, the
target-spec parser, and the dirty-set flush/upload step.

Run: python -m unittest tests/test_stream_tcp.py
"""
from __future__ import annotations
import argparse
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import muninn  # noqa: E402

REAL_SBS1_CAPTURE = ROOT / "examples" / "sbs1_real.txt"


MSG3 = ("MSG,3,1,1,A8A5DD,1,2026/05/09,12:00:00.000,2026/05/09,12:00:00.000,"
        ",30000,,,42.123,-81.456,,,,,,0")
MSG4 = ("MSG,4,1,1,A8A5DD,1,2026/05/09,12:00:01.000,2026/05/09,12:00:01.000,"
        ",,420,270,,,,,,,,0")
MSG1 = ("MSG,1,1,1,A8A5DD,1,2026/05/09,12:00:02.000,2026/05/09,12:00:02.000,"
        "TEST123 ,,,,,,,,,,,0")


def _fake_stream_args(**overrides):
    base = dict(
        upload=False, dry_run=False, stdout=False, no_save=False,
        out_dir=None, api_url=muninn.DEFAULT_API_URL, batch_size=1000,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class IngestSbs1RowTests(unittest.TestCase):

    def test_position_row_sets_lat_lon(self):
        rows: dict = {}
        icao = muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG3])), rows)
        self.assertEqual(icao, "A8A5DD")
        self.assertEqual(rows["A8A5DD"]["lat"], 42.123)
        self.assertEqual(rows["A8A5DD"]["lon"], -81.456)
        self.assertEqual(rows["A8A5DD"]["alt_ft"], 30000)

    def test_later_row_does_not_clear_earlier_fields(self):
        rows: dict = {}
        muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG3])), rows)
        muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG4])), rows)
        # velocity row (MSG4) has no lat/lon columns populated — position
        # from the earlier MSG3 row must survive, not get reset to None.
        self.assertEqual(rows["A8A5DD"]["lat"], 42.123)
        self.assertEqual(rows["A8A5DD"]["speed_kt"], 420)
        self.assertEqual(rows["A8A5DD"]["heading"], 270)

    def test_callsign_row_sets_callsign(self):
        rows: dict = {}
        muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG3])), rows)
        muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG1])), rows)
        self.assertEqual(rows["A8A5DD"]["callsign"], "TEST123")

    def test_non_msg_row_ignored(self):
        rows: dict = {}
        icao = muninn._ingest_sbs1_row(["SEL", "1", "1"], rows)
        self.assertIsNone(icao)
        self.assertEqual(rows, {})

    def test_short_row_ignored(self):
        rows: dict = {}
        icao = muninn._ingest_sbs1_row(["MSG", "3"], rows)
        self.assertIsNone(icao)

    def test_missing_icao_ignored(self):
        rows: dict = {}
        r = list(next(muninn.csv.reader([MSG3])))
        r[4] = ""
        icao = muninn._ingest_sbs1_row(r, rows)
        self.assertIsNone(icao)


class ParseStreamTargetTests(unittest.TestCase):

    def test_host_and_port(self):
        self.assertEqual(muninn._parse_stream_target("192.168.1.50:30003"),
                        ("192.168.1.50", 30003))

    def test_host_only_defaults_port_30003(self):
        self.assertEqual(muninn._parse_stream_target("192.168.1.50"),
                        ("192.168.1.50", 30003))

    def test_non_numeric_port_raises(self):
        with self.assertRaises(ValueError):
            muninn._parse_stream_target("host:notaport")

    def test_empty_host_raises(self):
        with self.assertRaises(ValueError):
            muninn._parse_stream_target(":30003")

    def test_port_zero_raises(self):
        with self.assertRaises(ValueError):
            muninn._parse_stream_target("host:0")

    def test_port_above_65535_raises(self):
        with self.assertRaises(ValueError):
            muninn._parse_stream_target("host:65536")

    def test_port_65535_is_valid(self):
        self.assertEqual(muninn._parse_stream_target("host:65535"),
                        ("host", 65535))


class ReadSocketLinesTests(unittest.TestCase):

    def test_yields_complete_lines_across_chunks(self):
        a, b = socket.socketpair()
        try:
            a.sendall(b"MSG,3,1,1,AAAAAA,")
            a.sendall(b"1,x,y\n")
            a.sendall(b"MSG,1,1,1,BBBBBB,1,x,y\n")
            a.close()  # peer closes -> generator should stop after draining
            lines = list(muninn._read_socket_lines(b, idle_timeout=1))
        finally:
            b.close()
        self.assertEqual(lines, [
            "MSG,3,1,1,AAAAAA,1,x,y",
            "MSG,1,1,1,BBBBBB,1,x,y",
        ])

    def test_yields_none_on_idle_then_resumes(self):
        a, b = socket.socketpair()
        try:
            gen = muninn._read_socket_lines(b, idle_timeout=0.2)
            self.assertIsNone(next(gen))  # nothing sent yet -> idle timeout
            a.sendall(b"hello\n")
            self.assertEqual(next(gen), "hello")
        finally:
            a.close()
            b.close()

    def test_oversized_segment_is_dropped_not_yielded(self):
        """A peer sending a giant blob with no newline (garbage feed, or a
        malicious host given how --stream takes an arbitrary user-supplied
        address) must not grow the buffer without bound, and the oversized
        segment must never be handed to the caller — even if a newline
        eventually shows up glued directly onto the end of it (no separate
        newline of its own), which is exactly the case that a naive
        check-after-extraction implementation gets wrong."""
        a, b = socket.socketpair()
        try:
            junk = b"x" * (muninn._STREAM_MAX_LINE_BYTES + 1024)
            a.sendall(junk + b"\n")
            a.close()
            lines = list(muninn._read_socket_lines(b, idle_timeout=1))
        finally:
            b.close()
        self.assertEqual(lines, [])

    def test_recovers_cleanly_once_oversized_junk_stops(self):
        """After an oversized, newline-less run is dropped, a real line
        that arrives once the junk has stopped (and drained on its own
        idle timeout) must come through clean, with no leftover bytes
        from the dropped segment glued onto it."""
        a, b = socket.socketpair()
        try:
            gen = muninn._read_socket_lines(b, idle_timeout=0.3)
            a.sendall(b"x" * (muninn._STREAM_MAX_LINE_BYTES + 4096))
            saw_idle = False
            for _ in range(50):
                if next(gen) is None:
                    saw_idle = True
                    break
            self.assertTrue(saw_idle, "expected an idle gap once the junk drained")
            a.sendall(b"MSG,1,1,1,CCCCCC,1,x,y\n")
            self.assertEqual(next(gen), "MSG,1,1,1,CCCCCC,1,x,y")
        finally:
            a.close()
            b.close()

    def test_strips_trailing_cr_for_crlf_feeds(self):
        """Some dump1090/readsb builds emit CRLF line endings (confirmed —
        see the real capture in examples/sbs1_real.txt). A trailing \\r
        must not leak into the yielded line and corrupt whichever CSV
        column happens to be last."""
        a, b = socket.socketpair()
        try:
            a.sendall(b"MSG,1,1,1,DDDDDD,1,x,y\r\n")
            a.close()
            lines = list(muninn._read_socket_lines(b, idle_timeout=1))
        finally:
            b.close()
        self.assertEqual(lines, ["MSG,1,1,1,DDDDDD,1,x,y"])


class RealCaptureParityTests(unittest.TestCase):
    """Feeds a real dump1090 SBS-1 capture through the exact socket
    line-reader + row-ingest path --stream uses, in small chunks that
    deliberately don't align to line boundaries, and checks the result
    matches parse_sbs1() (the file-based parser) byte for byte. This is
    the real regression check that the streaming path and the file path
    never drift apart on real-world data — including its CRLF endings,
    which is what first exposed the CR-stripping bug above."""

    @unittest.skipUnless(REAL_SBS1_CAPTURE.exists(), "real capture fixture missing")
    def test_streamed_ingestion_matches_file_parser_on_real_capture(self):
        raw = REAL_SBS1_CAPTURE.read_bytes()
        a, b = socket.socketpair()
        try:
            def sender():
                chunk = 137  # not aligned to any line length, on purpose
                for i in range(0, len(raw), chunk):
                    a.sendall(raw[i:i + chunk])
                a.close()

            t = threading.Thread(target=sender, daemon=True)
            t.start()

            rows: dict = {}
            for line in muninn._read_socket_lines(b, idle_timeout=2):
                if not line:
                    continue
                r = next(muninn.csv.reader([line]), None)
                if r:
                    muninn._ingest_sbs1_row(r, rows)
            t.join(timeout=5)
        finally:
            b.close()

        streamed = {}
        for icao, e in rows.items():
            rec = muninn._norm_record(icao=icao, callsign=e["callsign"],
                                     lat=e["lat"], lon=e["lon"],
                                     alt_ft=e["alt_ft"], speed_kt=e["speed_kt"],
                                     heading=e["heading"], first_seen=e["first_seen"])
            if rec:
                streamed[icao] = rec

        from_file = muninn.parse_sbs1(REAL_SBS1_CAPTURE)

        self.assertTrue(from_file, "fixture parser returned nothing — fixture missing data?")
        self.assertEqual(set(streamed.keys()), set(from_file.keys()))
        for icao in from_file:
            self.assertEqual(streamed[icao], from_file[icao], f"mismatch for {icao}")


class FlushStreamRecordsTests(unittest.TestCase):

    def setUp(self):
        self.rows = {}
        muninn._ingest_sbs1_row(next(muninn.csv.reader([MSG3])), self.rows)
        r2 = list(next(muninn.csv.reader([MSG3])))
        r2[4] = "BBBBBB"
        muninn._ingest_sbs1_row(r2, self.rows)

    def test_only_dirty_icaos_are_flushed(self):
        calls = []
        orig = muninn.upload
        muninn.upload = lambda records, *a, **k: (calls.append(records) or 0)
        try:
            dirty = {"A8A5DD"}
            args = _fake_stream_args(upload=True)
            muninn._flush_stream_records(self.rows, dirty, "fakekey", args)
        finally:
            muninn.upload = orig
        self.assertEqual(len(calls), 1)
        icaos = {r["icao"] for r in calls[0]}
        self.assertEqual(icaos, {"A8A5DD"})
        self.assertEqual(dirty, set())  # cleared after successful upload

    def test_no_dirty_is_a_noop(self):
        orig = muninn.upload
        muninn.upload = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("upload should not be called"))
        try:
            muninn._flush_stream_records(self.rows, set(), "fakekey",
                                        _fake_stream_args(upload=True))
        finally:
            muninn.upload = orig

    def test_failed_upload_keeps_dirty_for_retry(self):
        orig = muninn.upload
        muninn.upload = lambda *a, **k: 1  # nonzero rc == failure
        try:
            dirty = {"A8A5DD"}
            muninn._flush_stream_records(self.rows, dirty, "fakekey",
                                        _fake_stream_args(upload=True))
        finally:
            muninn.upload = orig
        self.assertEqual(dirty, {"A8A5DD"})

    def test_writes_out_dir_when_set(self):
        with tempfile.TemporaryDirectory() as td:
            dirty = {"A8A5DD"}
            args = _fake_stream_args(out_dir=td)
            muninn._flush_stream_records(self.rows, dirty, None, args)
            written = list(Path(td).glob("stream-*.wdgwars.json"))
            self.assertEqual(len(written), 1)


if __name__ == "__main__":
    unittest.main()
