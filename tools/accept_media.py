#!/usr/bin/env python3
"""Continuous video publishing was retired in v0.3.0.

Use tools/accept_camera.py to verify on-demand photographs and resource release.
The historical receiver acceptance tool remains in Git history.
"""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error('Continuous publishing is retired; use tools/accept_camera.py')
