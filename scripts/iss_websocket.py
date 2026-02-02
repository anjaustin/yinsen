#!/usr/bin/env python3
"""
ISS Telemetry WebSocket Shim v2 — Lightstreamer to stdout

Thin bridge between ISS Mimic's Lightstreamer feed and the Yinsen
C processor. Subscribes to selected ISS telemetry parameters and
outputs (channel_id, timestamp, value) lines to stdout.

v2 protocol: CALIBRATE → data → ENROLL → data → FINALIZE → data
  Calibration computes per-channel input statistics for pre-scaling.
  This fixes the CMG hidden-state degeneration found in probes 1-2.

Pipe to the C processor:
    python scripts/iss_websocket.py --demo | ./examples/iss_telemetry --stdin

Or with calibration + enrollment:
    python scripts/iss_websocket.py --calibrate 100 --enroll 300 | ./examples/iss_telemetry --stdin

Dependencies:
    pip install lightstreamer-client-lib

ISS Lightstreamer details:
    Server: push.lightstreamer.com
    Adapter: ISSLIVE
    Each subscription update delivers (TimeStamp, Value) per item.

Telemetry IDs (from ISS Mimic / AMPS):
    CMG vibration, coolant temps, cabin pressure, O2 levels, etc.
    Full list: https://github.com/ISS-Mimic/Mimic

Created by: Tripp + Manus
Date: February 2026
"""

import sys
import time
import signal
import argparse
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Channel Mapping
#
# Maps ISS telemetry item names to channel IDs for the C processor.
# These are real AMPS identifiers from the ISS Mimic project.
#
# CMG (Control Moment Gyroscope) telemetry:
#   CMG vibration and wheel speed data for bearing health monitoring.
#   This IS the vibration PdM use case — just on ISS hardware.
#
# Thermal:
#   External Thermal Control System (ETCS) loop temperatures.
#
# Atmosphere:
#   Cabin pressure and O2 partial pressure from CDRA/OGS.
# ─────────────────────────────────────────────────────────────────────────────

# Default channel configuration — v2 (validated against live feed 2026-02-01)
# Format: (channel_id, lightstreamer_item_name, human_label)
#
# Channel selection criteria: CONTINUOUS-valued parameters only.
# Discrete/boolean values (CMG3/CMG4 status=1, O2=28) are excluded.
# All 8 channels verified to return float-valued telemetry from live ISS.
DEFAULT_CHANNELS = [
    # CMG — wheel speed and spin motor current (bearing health proxy)
    (0, "USLAB000058", "CMG1-WhlSpd"),  # CMG1 Wheel Speed (~745)
    (1, "USLAB000059", "CMG2-WhlSpd"),  # CMG2 Wheel Speed (~24)
    (2, "USLAB000060", "CMG1-SpnCur"),  # CMG1 Spin Motor Current (~17)
    (3, "USLAB000061", "CMG2-SpnCur"),  # CMG2 Spin Motor Current (~9)
    # Thermal — ETCS Loop A/B outlet temperatures (Fahrenheit on live feed)
    (4, "S6000008", "ETCS-A-Tout"),  # ETCS Loop A Outlet Temp (~236°F)
    (5, "P6000008", "ETCS-B-Tout"),  # ETCS Loop B Outlet Temp (~323°F)
    # Atmosphere
    (6, "AIRLOCK000049", "CabinP-mmHg"),  # Cabin Total Pressure (~743 mmHg)
    (7, "NODE3000003", "Atmo-ppCO2"),  # Atmosphere CO2/trace (~2.26)
]


def eprint(*args, **kwargs):
    """Print to stderr (doesn't interfere with stdout pipe)."""
    print(*args, file=sys.stderr, **kwargs)


def run_live(channels, calibrate_samples=0, enroll_samples=0, max_samples=0):
    """
    Connect to ISS Lightstreamer and stream telemetry to stdout.

    v2 Protocol output:
        If calibrate_samples > 0:
            CALIBRATE,<n>
            <data lines during calibration>
        If enroll_samples > 0:
            ENROLL,<n>
            <data lines during enrollment>
            FINALIZE
        <data lines forever>

    Each data line: channel_id,timestamp,value
    """
    try:
        from lightstreamer.client import (
            LightstreamerClient,
            Subscription,
            SubscriptionListener,
        )
    except ImportError:
        eprint("ERROR: lightstreamer-client-lib not installed.")
        eprint("  pip install lightstreamer-client-lib")
        eprint("")
        eprint("Falling back to demo mode (synthetic data).")
        eprint("Use --demo to suppress this message.")
        run_demo(channels, calibrate_samples, enroll_samples, max_samples)
        return

    item_names = [ch[1] for ch in channels]
    ch_id_map = {ch[1]: ch[0] for ch in channels}

    # Connection
    client = LightstreamerClient("https://push.lightstreamer.com", "ISSLIVE")

    # Subscription
    sub = Subscription(mode="MERGE", items=item_names, fields=["TimeStamp", "Value"])

    sample_count = [0]
    calibrated = [False]
    enrolled = [False]
    start_time = [None]
    phase = ["calibrate" if calibrate_samples > 0 else "enroll"]

    # Emit first protocol command
    if calibrate_samples > 0:
        sys.stdout.write(f"CALIBRATE,{calibrate_samples}\n")
        sys.stdout.flush()
        eprint(f"Phase 0: Calibrating ({calibrate_samples} samples/channel)...")
    elif enroll_samples > 0:
        sys.stdout.write(f"ENROLL,{enroll_samples}\n")
        sys.stdout.flush()
        eprint(
            f"Phase 1: Enrolling ({enroll_samples} samples/channel) [no calibration]..."
        )

    cal_total = calibrate_samples * len(channels)

    class Listener(SubscriptionListener):
        def onItemUpdate(self, update):
            item_name = update.getItemName()
            ts_str = update.getValue("TimeStamp")
            val_str = update.getValue("Value")

            if ts_str is None or val_str is None:
                return

            try:
                timestamp = float(ts_str)
                value = float(val_str)
            except (ValueError, TypeError):
                return

            ch_id = ch_id_map.get(item_name, -1)
            if ch_id < 0:
                return

            # Use wall clock for timestamps (ISS timestamps are fractional
            # day numbers — not directly useful for CfC dt computation).
            # Wall clock gives actual sampling interval in seconds.
            wall_now = time.time()
            if start_time[0] is None:
                start_time[0] = wall_now

            rel_ts = wall_now - start_time[0]

            sys.stdout.write(f"{ch_id},{rel_ts:.1f},{value}\n")
            sys.stdout.flush()

            sample_count[0] += 1

            # Calibration → Enrollment transition
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

            # Enrollment finalization
            enroll_start = cal_total if calibrate_samples > 0 else 0
            if (
                enroll_samples > 0
                and not enrolled[0]
                and sample_count[0] >= enroll_start + enroll_samples * len(channels)
            ):
                sys.stdout.write("FINALIZE\n")
                sys.stdout.flush()
                enrolled[0] = True
                phase[0] = "detect"
                eprint(f"Enrollment complete. Now detecting.")

            # Max samples limit
            if max_samples > 0 and sample_count[0] >= max_samples:
                eprint(f"Reached {max_samples} samples. Disconnecting.")
                client.disconnect()

            # Progress
            if sample_count[0] % 100 == 0:
                eprint(f"  [{phase[0]}] {sample_count[0]} samples (t={rel_ts:.0f}s)")

    sub.addListener(Listener())
    client.subscribe(sub)

    eprint(f"Connecting to ISS Lightstreamer...")
    eprint(f"  Server: push.lightstreamer.com")
    eprint(f"  Adapter: ISSLIVE")
    eprint(f"  Items: {len(item_names)}")
    if calibrate_samples > 0:
        eprint(f"  Calibrate: {calibrate_samples} samples/channel")
    if enroll_samples > 0:
        eprint(f"  Enroll: {enroll_samples} samples/channel")
    for ch_id, item, label in channels:
        eprint(f"    ch{ch_id}: {item} ({label})")

    client.connect()

    eprint("Connected. Streaming telemetry to stdout...")
    eprint("Press Ctrl-C to stop.\n")

    # Keep alive until interrupted or max samples reached
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        eprint("\nInterrupted. Disconnecting...")
    finally:
        client.disconnect()
        eprint(f"Done. {sample_count[0]} total samples.")


def run_demo(channels, calibrate_samples=0, enroll_samples=0, max_samples=0):
    """
    Demo mode: generate synthetic telemetry locally.
    Same patterns as the C simulator but from Python.
    Useful for testing the pipe without network access.

    v2: supports CALIBRATE → ENROLL → FINALIZE protocol.
    """
    import math
    import random

    eprint("Running in demo mode (synthetic ISS telemetry, v2 protocol)")

    n_channels = len(channels)

    # Phase 0: Calibration (or direct enrollment if no calibration)
    if calibrate_samples > 0:
        sys.stdout.write(f"CALIBRATE,{calibrate_samples}\n")
        sys.stdout.flush()
        eprint(f"Phase 0: Calibrating ({calibrate_samples} samples/channel)...")
    elif enroll_samples > 0:
        sys.stdout.write(f"ENROLL,{enroll_samples}\n")
        sys.stdout.flush()
        eprint(
            f"Phase 1: Enrolling ({enroll_samples} samples/channel) [no calibration]..."
        )

    ORBITAL_PERIOD = 5520.0
    dt = 10.0
    t = 0.0
    sample_count = 0
    cal_total = calibrate_samples * n_channels
    calibrated = False
    enrolled = False

    if max_samples <= 0:
        max_samples = 100000  # ~278 hours at 10s interval

    rng = random.Random(42)

    def generate(ch_id, t):
        phase = 2.0 * math.pi * t / ORBITAL_PERIOD
        noise = rng.gauss(0, 1)

        if ch_id < 4:  # CMG vibration
            base = 0.001 + 0.0001 * ch_id
            return base + 0.0002 * math.sin(phase) + 0.0002 * noise
        elif ch_id < 6:  # Coolant temp
            base = 15.0 + (2.0 if ch_id == 5 else 0.0)
            return base + 8.0 * math.sin(phase) + 0.5 * noise
        elif ch_id == 6:  # Cabin pressure
            return 101.3 + 0.02 * math.sin(phase) + 0.05 * noise
        elif ch_id == 7:  # O2
            cdra = (t % 900.0) / 900.0
            return 21.3 + 0.15 * (cdra - 0.5) + 0.1 * noise
        return noise

    try:
        while sample_count < max_samples:
            for ch_id, _, _ in channels:
                value = generate(ch_id, t)
                sys.stdout.write(f"{ch_id},{t:.1f},{value:.6f}\n")
                sys.stdout.flush()
                sample_count += 1

                # Calibration → Enrollment transition
                if (
                    calibrate_samples > 0
                    and not calibrated
                    and sample_count >= cal_total
                ):
                    calibrated = True
                    eprint(f"Calibration complete ({sample_count} samples)")
                    if enroll_samples > 0:
                        sys.stdout.write(f"ENROLL,{enroll_samples}\n")
                        sys.stdout.flush()
                        eprint(
                            f"Phase 1: Enrolling ({enroll_samples} samples/channel)..."
                        )

                # Enrollment finalization
                enroll_start = cal_total if calibrate_samples > 0 else 0
                if (
                    enroll_samples > 0
                    and not enrolled
                    and sample_count >= enroll_start + enroll_samples * n_channels
                ):
                    sys.stdout.write("FINALIZE\n")
                    sys.stdout.flush()
                    enrolled = True
                    eprint(f"Enrollment complete ({sample_count} samples). Detecting.")

            t += dt

            if sample_count % (n_channels * 100) == 0:
                phase_label = (
                    "calibrate"
                    if not calibrated and calibrate_samples > 0
                    else ("enroll" if not enrolled and enroll_samples > 0 else "detect")
                )
                eprint(
                    f"  [{phase_label}] {sample_count} samples, t={t:.0f}s "
                    f"({t / ORBITAL_PERIOD:.2f} orbits)"
                )

            # Simulate real-time rate (optional, remove for max speed)
            # time.sleep(0.01)

    except KeyboardInterrupt:
        eprint("\nInterrupted.")
    except BrokenPipeError:
        eprint("\nPipe closed by consumer.")

    eprint(f"Done. {sample_count} total samples.")


def main():
    parser = argparse.ArgumentParser(
        description="ISS Telemetry WebSocket Shim v2 — Lightstreamer to stdout"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data instead of live ISS feed",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        default=100,
        help="Number of calibration samples per channel (default: 100, 0=skip)",
    )
    parser.add_argument(
        "--enroll",
        type=int,
        default=300,
        help="Number of enrollment samples per channel (default: 300)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after N total samples (0 = unlimited)",
    )
    parser.add_argument(
        "--list-channels", action="store_true", help="Print channel mapping and exit"
    )

    args = parser.parse_args()

    if args.list_channels:
        print("Channel mapping (v2):")
        for ch_id, item, label in DEFAULT_CHANNELS:
            print(f"  {ch_id}: {item:20s}  {label}")
        print(
            f"\nProtocol: CALIBRATE,{args.calibrate} -> data "
            f"-> ENROLL,{args.enroll} -> data -> FINALIZE -> data"
        )
        return

    # Handle SIGPIPE gracefully (pipe closed by consumer)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if args.demo:
        run_demo(DEFAULT_CHANNELS, args.calibrate, args.enroll, args.max_samples)
    else:
        run_live(DEFAULT_CHANNELS, args.calibrate, args.enroll, args.max_samples)


if __name__ == "__main__":
    main()
