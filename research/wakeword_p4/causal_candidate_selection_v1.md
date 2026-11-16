# Candidate selection after three causal-unit controls

Three complete-wave training controls share the same model, initialization and5000 sampler schedule:

| objective | target hits / 720 | negative events / 3600 |
|---|---:|---:|
| frame allocation CE | 530 | 149 |
| frame + equal position CE | 520 | 39 |
| frame + position + local context contrast | 538 | 51 |

The local contrast run improves target recall over position weighting but worsens negative events. Position weighting is the lower-confusion control, but its target recall remains insufficient and the later trained delay40 INT8 model only reaches476/720 targets and26/3600 negatives under bounded policy. None is an eligible replacement.

No checkpoint or threshold is selected from these consumed training counts. The contextual checkpoint is not exported or sent to Mac. Any next comparison must use the predeclared unused development set and the same model/INT8 protocol; the protected final sets remain closed. Frozen v30 remains the only handoff model.
