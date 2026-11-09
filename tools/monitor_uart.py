#!/usr/bin/env python3
"""Record UART without deliberately toggling DTR/RTS; stop with Ctrl+C.

Requires pyserial. Close this process before flashing. Opening some USB adapters
can still reset the board; wait for network/time synchronization after opening.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import serial

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(mode=0o600, exist_ok=True)
    args.output.chmod(0o600)
    uart = serial.Serial()
    uart.port, uart.baudrate, uart.timeout = args.port, args.baud, .25
    uart.dtr = uart.rts = False
    try:
        uart.open()
        with args.output.open('ab', buffering=0) as output:
            while True:
                data = uart.read(8192)
                if data:
                    output.write(data)
    except KeyboardInterrupt:
        pass
    finally:
        uart.close()


if __name__ == '__main__':
    main()
