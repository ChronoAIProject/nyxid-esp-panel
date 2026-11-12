# Full balanced training feature pack v1

Input exactly8064original full-grid rows,manifest SHA256
e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0,
originalreport SHA611cbe534996f26025ff35bafc04078ce170fed14ee07dce657c57f540085c97.
6056aa9 independent audit passes all8064 originalmechanical gates,hashes,
duration arrays and exactfixed-host conversion. All672IDs already training-only.
No pilot rows/held-out identities/old or new development/final data added.

Exactly3variants per source=24192 attempts; same numerical augmentation recipe
as preregistered US pack:complete original serializedPCM;Gaussian noise25dB;
full convolution RIRdelays0/160/480/960,amplitudes1/.25/-.12/.06 plusnoise15dB.
For each variant normalize signal RMS to.08 before noise;explicit augmentation
only, frozen source bytes untouched. Lead32000/tail16000;include entireRIRtail.
Global postmixheadroom gain min(1,.95/peak). No trimming or silence removal.
Seed first16hex SHA256(originalfilename+':variant:piper-balanced-feature-v1').
Store prequantizationfloat64,PCM16WAV,rawfrontenduint16,file/PCM/data hashes,
fullsource/duration provenance through originalmanifest,all gains/seed/order.
Frontend consumes serializedPCM,not the originalfloat. Original source rows
adapt only metadata names:voice=str(speaker_id),length_multiplier=length_scale,
path/hash from files.wav. Originalmanifest never rewritten.

Everyattempt must be finite,16kmonoPCM16,nonempty/nonzero,noPCMendpoints,
fullsignal/outputlength preserved,exact fixed-libraryroundtrip. Save all
attempts and any unsupported contexts;no quality/score-based filtering.
Existingcontext150 ledger e=ceil((32000+signal_samples)/160)-1;
raw[e-150:e] eligible only ife>=150,e<=len(raw),begin=(e-150)*160<=32000,
stop=(e+1)*160>=32000+signal_samples. Save complete rawhistory. All672×4cells
must remain nonempty;report every unsupported byID/phrase/speed/variant.
This is fullcontainer support,not acoustic phoneme truth or receptivefield.

Independent read-only audit must rehash everyoriginal and augmentedfile,repeat
all augmentationfloat arrays,PCMconversion,serializedPCMfrontend,ledger/crops
and labels,pluscounts/grid/sourceisolation. Unsupported rows retained and
independently checked rather than hidden. Same numerical/frontendlibraries
limit implementationindependence. Oldqualityprovenance gaps stay explicit.

No wake-modelscoring,training,threshold/architecture/checkpoint/loss changes
or promotion inthisfeaturestage. A future fitting plan must separately freeze
sampler/source mass and calibration,independent RNG/smoke/export checks and
originaloldstage/acceptance gates. V30/newdev/finals preserved.
