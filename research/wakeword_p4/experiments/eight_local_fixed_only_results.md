# Released fixed-sound and C161 listeners on 100 matched inputs

Both digital P4 guests opened on all eight issued voices through PCM48 and AAC in weak-target 079, at sample 41,472 with 471 ms left before the command. The two erased, two Willow and 16 previous-voice conversions stayed closed on both. In 077 the two listeners opened before the command on all 24 room/voice inputs. The microphones and electrical references were identical for both runs. The 200 UARTs cover 106 listening cycles per receiver. [Protocol](eight_local_fixed_only_protocol.md) · [Per-input events and allocations](../results/4090-eight-local-fixed-only.json) · [UARTs](../results/4090-eight-local-fixed-only.zip)

| Input | Standalone first-open sample | C161 plus fixed marker, reference-rearm |
| --- | ---: | ---: |
| We need six cups of water | Closed | 37,632 (model) |
| Unaddressed voices with music, case 072 | Closed | 52,992 (model) |
| Plain synthetic Bella in 079 with the competing speaker muted | Closed | 47,616 (model) |
| Panel playing Bella, electrical reference present | 2/4 opened | 0/4 opened |
| Same panel microphones, reference zeroed | 4/4 opened | 4/4 opened |

With a synchronous reference the standalone opened in geometries 1 and 3 at 48,384 and 48,128. In the 20 two-speaker collisions the first-open counts at offsets −800, −400, 0, +400 and +800 ms were 4/4, 3/4, 2/4, 2/4 and 2/4 for standalone, and 4/4, 2/4, 0/4, 3/4 and 4/4 for reference-rearm. At +400 and +800 ms all seven openings on the latter had a marker clock specific to the mixture. The records include the panel-alone and external-alone clocks for attribution.

On the second creation in each of six boots, standalone used 0 additional internal heap bytes and 9,492 external bytes. The C161 receiver used 94,136 and 13,308 bytes, respectively. Both returned to their second-cycle starting heaps after destruction in all six cases. These are simulated listener allocations. The physical board supplies the minimum whole-device heap, dropped reads and call audio. All inputs were previously consumed and contribute zero additional independent background hours.
