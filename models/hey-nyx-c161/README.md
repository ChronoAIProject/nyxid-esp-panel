# Hey Nyx c161 preview

These are the original INT8 bytes delivered in GPU research commit `06fe353c`,
not a retrained or re-exported model. The 60,952-byte TFLite file has SHA256
`c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81`.
The profile preserves the development-board trial's threshold, five-step
smoothing, 309-frame context and continuous 81,600-sample startup prime.

This preview branch selects c161 by default. Normal production NyxID endpoints
and each panel's own saved production credentials apply; no account credentials,
Wi-Fi configuration, local API address or microphone recordings are embedded.
The single microphone owner, AEC/AGC, ordered lookback/FIFO and PCM speed are
unchanged. The idle detector consumes raw samples from that same owner; call
audio and its preroll remain on the processed path.

The actual user trial needed three attempts to wake. Its successful wake
delivered the complete following request and a normal response. Subsequent user
feedback still found wake quality unsatisfactory. This is an opt-in preview,
not an assertion of mature wake reliability or long-call/AEC acceptance.
See `research/wakeword_p4/results/mac-c161-development-p4-trial-v1.json`.

To build this checkout with the original Hi ESP detector instead, use a fresh
build directory and explicitly clear both CMake selectors:

```sh
idf.py -B build-hi-esp \
  -D NYX_PANEL_EXPERIMENTAL_WAKE_MODEL= \
  -D NYX_PANEL_EXPERIMENTAL_WAKE_STREAMING_PROFILE= build
```
