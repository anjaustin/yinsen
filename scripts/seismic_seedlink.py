#!/usr/bin/env python3
"""
Seismic SeedLink Shim — IRIS/GFZ to stdout (v2 protocol)

Streams real-time seismic waveforms from the global SeedLink network
to the Yinsen C processor. 100 Hz, 3-component broadband seismograms.

This is a 100x speed increase over the ISS Lightstreamer feed (~1 Hz).
At 100 Hz with variable-amplitude seismic signals, the CfC must track
P-waves (seconds), S-waves (seconds), and surface waves (tens of seconds)
simultaneously — a genuine multi-timescale temporal problem where tau
actually matters.

Protocol (v2):
    CALIBRATE,N -> data -> ENROLL,M -> data -> FINALIZE -> data
    Each data line: channel_id,timestamp,value

Pipe to C processor:
    python scripts/seismic_seedlink.py | ./examples/iss_telemetry --stdin

Channels:
    0: HHZ (vertical, 100 Hz)
    1: HHN (north-south, 100 Hz)
    2: HHE (east-west, 100 Hz)

Dependencies:
    pip install obspy

Created by: Tripp + Manus
Date: February 2026
"""

import sys
import time
import signal
import argparse
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Station configurations — verified working on GFZ SeedLink
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = {
    "STU": {
        "server": "geofon.gfz-potsdam.de:18000",
        "network": "GE",
        "station": "STU",
        "channels": ["HHZ", "HHN", "HHE"],  # 100 Hz
        "label": "Stuttgart, Germany",
    },
    "WLF": {
        "server": "geofon.gfz-potsdam.de:18000",
        "network": "GE",
        "station": "WLF",
        "channels": ["HHZ", "HHN", "HHE"],
        "label": "Walferdange, Luxembourg",
    },
}

DEFAULT_STATION = "STU"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_seedlink(station_key, calibrate_samples, enroll_samples, max_samples):
    """Stream seismic data from SeedLink to stdout with v2 protocol."""
    from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

    cfg = STATIONS[station_key]
    ch_map = {ch: i for i, ch in enumerate(cfg["channels"])}
    n_channels = len(cfg["channels"])

    sample_count = [0]
    start_time = [None]
    cal_total = calibrate_samples * n_channels
    enroll_total = enroll_samples * n_channels
    calibrated = [False]
    enrolled = [False]
    phase = ["calibrate" if calibrate_samples > 0 else "enroll"]

    # Emit first protocol command
    if calibrate_samples > 0:
        sys.stdout.write(f"CALIBRATE,{calibrate_samples}\n")
        sys.stdout.flush()
        eprint(f"Phase 0: Calibrating ({calibrate_samples} samples/channel)...")
    elif enroll_samples > 0:
        sys.stdout.write(f"ENROLL,{enroll_samples}\n")
        sys.stdout.flush()
        eprint(f"Phase 1: Enrolling ({enroll_samples} samples/channel)...")

    class SeismicClient(EasySeedLinkClient):
        def on_data(self, trace):
            ch_name = trace.stats.channel
            ch_id = ch_map.get(ch_name, -1)
            if ch_id < 0:
                return

            sr = trace.stats.sampling_rate
            data = trace.data
            n = len(data)

            # Use trace start time as reference
            if start_time[0] is None:
                start_time[0] = trace.stats.starttime.timestamp

            # Output each sample with its computed timestamp
            base_ts = trace.stats.starttime.timestamp - start_time[0]
            dt = 1.0 / sr

            try:
                for i in range(n):
                    t = base_ts + i * dt
                    sys.stdout.write(f"{ch_id},{t:.4f},{data[i]}\n")
                    sample_count[0] += 1

                sys.stdout.flush()
            except BrokenPipeError:
                eprint("Pipe closed by consumer.")
                self.close()
                return

            # Phase transitions
            if (
                phase[0] == "calibrate"
                and not calibrated[0]
                and sample_count[0] >= cal_total
            ):
                calibrated[0] = True
                phase[0] = "enroll"
                eprint(f"Calibration complete ({sample_count[0]} samples).")
                if enroll_samples > 0:
                    sys.stdout.write(f"ENROLL,{enroll_samples}\n")
                    sys.stdout.flush()
                    eprint(f"Phase 1: Enrolling ({enroll_samples} samples/channel)...")

            enroll_start = cal_total if calibrate_samples > 0 else 0
            if (
                enroll_samples > 0
                and not enrolled[0]
                and sample_count[0] >= enroll_start + enroll_total
            ):
                sys.stdout.write("FINALIZE\n")
                sys.stdout.flush()
                enrolled[0] = True
                phase[0] = "detect"
                eprint(f"Enrollment finalized. Detecting.")

            # Max samples
            if max_samples > 0 and sample_count[0] >= max_samples:
                eprint(f"Reached {max_samples} samples. Closing.")
                self.close()

            # Progress
            if sample_count[0] % (n_channels * 1000) == 0:
                elapsed = time.time() - (
                    start_time[0] if start_time[0] else time.time()
                )
                rate = sample_count[0] / max(elapsed, 0.001) if start_time[0] else 0
                eprint(
                    f"  [{phase[0]}] {sample_count[0]} samples "
                    f"({sample_count[0] / n_channels:.0f}/ch, ~{rate:.0f} samp/s)"
                )

    try:
        eprint(f"Connecting to {cfg['server']}...")
        eprint(f"  Station: {cfg['network']}.{cfg['station']} ({cfg['label']})")
        eprint(f"  Channels: {', '.join(cfg['channels'])}")

        client = SeismicClient(cfg["server"], autoconnect=False)
        for ch in cfg["channels"]:
            client.select_stream(cfg["network"], cfg["station"], ch)
        client.connect()

        eprint("Connected. Streaming seismic data...")
        client.run()

    except KeyboardInterrupt:
        eprint("\nInterrupted.")
    except BrokenPipeError:
        eprint("\nPipe closed.")
    except Exception as e:
        eprint(f"Error: {type(e).__name__}: {e}")

    eprint(f"Done. {sample_count[0]} total samples.")


def main():
    parser = argparse.ArgumentParser(
        description="Seismic SeedLink Shim — real-time seismograms to stdout"
    )
    parser.add_argument(
        "--station",
        choices=list(STATIONS.keys()),
        default=DEFAULT_STATION,
        help=f"Station to stream from (default: {DEFAULT_STATION})",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        default=500,
        help="Calibration samples per channel (default: 500 = 5s at 100Hz)",
    )
    parser.add_argument(
        "--enroll",
        type=int,
        default=2000,
        help="Enrollment samples per channel (default: 2000 = 20s at 100Hz)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after N total samples (0 = unlimited)",
    )
    parser.add_argument(
        "--list-stations", action="store_true", help="List available stations"
    )

    args = parser.parse_args()

    if args.list_stations:
        for key, cfg in STATIONS.items():
            print(
                f"  {key}: {cfg['network']}.{cfg['station']} "
                f"({cfg['label']}) via {cfg['server']}"
            )
            print(f"       Channels: {', '.join(cfg['channels'])}")
        return

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    run_seedlink(args.station, args.calibrate, args.enroll, args.max_samples)


if __name__ == "__main__":
    main()
