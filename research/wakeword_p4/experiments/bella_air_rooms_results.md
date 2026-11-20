# Stereo Hye Nyx playback in the Aachen hall and stairway

The released Hye Nyx baseline A and candidate B each opened on all 12 presentations at the original player setting, and again on all 12 with the target speaker raised 6 dB. A and B opened on the same 256-sample read in every pair, 467–479 ms before command arrival. These simulations use two additional measured spaces, three previously used synthetic commands, and a pair of opposing source directions within each space. The right ear of a measurement head provides the microphone response. [Registered protocol and dataset](bella_air_rooms_protocol.md) · [96 PCM identities and four response hashes](../results/4090-bella-air-rooms-plan.json)

| Timely opens in six presentations per space | Original A / B / Willow | Target +6 dB, A / B / Willow |
| --- | ---: | ---: |
| Aula Carolina hall | 6 / 6 / 2 | 6 / 6 / 2 |
| Stairway | 6 / 6 / 1 | 6 / 6 / 4 |

The fixed marker accounted for all Nyx openings. At each gain both receivers stayed closed on all 12 address-erased presentations and on each other's addresses. All 96 quantized GPU-rendered microphones matched a separate CPU convolution sample for sample. The normalization preserves the relative levels within each distributed response pair. The simulated levels have no absolute SPL calibration. [Paired clocks](../results/4090-bella-air-rooms.json) · [384 native evidence members](../results/4090-bella-air-rooms.zip)

## Hall master level

With the target raised 6 dB, the hall microphone clipped in 14 of 24 waveforms, including six with clipping before command onset. There were 2,321 saturated samples across those 24 waveforms. Several files share the same command audio. The stairway and all original-level microphones had zero clipped samples. Reducing the common master by 6 dB when playing the same +6 dB stereo files restored the original target level and left the competing speaker 6 dB lower. All 24 hall microphones then had zero clipped samples. The first opening clocks, including all closed controls, matched the nominal-master clocks exactly for both receivers. [Registered common-master comparison](bella_air_master_protocol.md) · [24 paired inputs and clocks](../results/4090-bella-air-master.json) · [96 native evidence members](../results/4090-bella-air-master.zip)

## Received command

Each entry gives Qwen / Whisper turbo counts for a complete target-command suffix across six presentations, including missed openings. Both readers processed the complete decoded send. The table preserves the two readiness schedules and speaker settings.

| Space and player level | A immediate | B immediate | Willow immediate | A delayed | B delayed | Willow delayed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hall, original | 2 / 3 | 2 / 3 | 2 / 1 | 2 / 3 | 2 / 3 | 2 / 1 |
| Hall, target +6 dB | 3 / 2 | 3 / 3 | 2 / 2 | 3 / 3 | 2 / 1 | 2 / 2 |
| Hall, target +6 dB with master −6 dB | 3 / 2 | 3 / 3 | 2 / 2 | 3 / 2 | 3 / 1 | 2 / 2 |
| Stairway, original | 5 / 3 | 5 / 5 | 1 / 1 | 5 / 5 | 5 / 5 | 1 / 1 |
| Stairway, target +6 dB | 5 / 6 | 6 / 6 | 4 / 4 | 6 / 6 | 5 / 5 | 4 / 4 |

In the hall's 0° presentation of 077 at the original player setting, A opened 473 ms before the target command “turn down the hall light.” Qwen returned the opposite speaker's “Hey Nick, what time is it?” on both sends. Removing clipping with the common master did not consistently improve target-command recovery. The stairway provided stronger first-command results for the same playback files. On these fixtures B had the strongest immediate score in the stairway and A the strongest delayed score. The paired board trial continues with A as baseline. These reader outputs measure the received audio. The actual Mac/P4 microphone, reference, service words, heap and physical gain calibration remain outstanding for this handoff. [72 full-send comparisons](../results/4090-bella-air-rooms-commands.json) · [18 hall master comparisons](../results/4090-bella-air-master-commands.json)

The two send-path archives contain the capture, codec and sender UARTs from 57 and 14 openings. All 213 logs had zero watchdog alerts and matched their individual hashes after anonymous Release downloads. [171 UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-air-rooms-commands.zip) · [42 hall master UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-air-master-commands.zip) · [Synchronized stereo files used on both room paths](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-stereo-player-6db-v1.zip)
