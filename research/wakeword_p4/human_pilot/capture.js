const el = id => document.getElementById(id);
const [participant, condition, progress, prompt, status, record, save] =
  ["participant", "condition", "progress", "prompt", "status", "record", "save"].map(el);

const rows = [];
let active, trialList, spec, firstSettings;
const processing = ["echoCancellation", "noiseSuppression", "autoGainControl"];

function render() {
  const current = trialList[rows.length];
  progress.textContent = current ? `${rows.length + 1} of ${trialList.length}` : "Recording complete";
  prompt.textContent = current?.text || "Save the session to this Mac";
  record.disabled = !current;
  save.disabled = rows.length !== trialList.length;
}

function encode(buffer) {
  const samples = new Float32Array(buffer);
  const pcm = new DataView(new ArrayBuffer(samples.length * 2));
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(32767 / 32768, samples[i]));
    pcm.setInt16(i * 2, Math.round(v * 32768), true);
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192)
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}

function closeCapture(capture) {
  capture.stream?.getTracks().forEach(track => track.stop());
  if (capture.context && capture.context.state !== "closed") void capture.context.close();
}

record.addEventListener("click", async () => {
  participant.setCustomValidity(/^[A-Za-z0-9_-]{1,24}$/.test(participant.value) ? "" : "Use letters, numbers, dashes or underscores.");
  if (active || !participant.reportValidity()) return;
  const current = trialList[rows.length];
  const capture = active = {};
  record.disabled = true;
  status.textContent = "Opening the microphone…";
  try {
    const constraints = Object.fromEntries(processing.map(key => [key, {exact: false}]));
    capture.stream = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: {ideal: 1}, ...constraints}
    });
    const track = capture.stream.getAudioTracks()[0];
    const settings = track.getSettings();
    if (![1, 2].includes(settings.channelCount) || processing.some(key => settings[key] !== false))
      throw new Error("The microphone could not provide unprocessed audio. Try Chrome.");
    capture.context = new AudioContext();
    const sampleRate = capture.context.sampleRate;
    const identity = JSON.stringify([settings.deviceId, settings.channelCount, sampleRate]);
    if (firstSettings && identity !== firstSettings)
      throw new Error("The microphone or sample rate changed. Restore the original input.");
    await capture.context.audioWorklet.addModule("./recorder.js");
    const source = capture.context.createMediaStreamSource(capture.stream);
    const worklet = new AudioWorkletNode(capture.context, "pilot-recorder", {
      channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers",
      processorOptions: {samples: sampleRate * spec.clip_seconds}
    });
    const muted = capture.context.createGain();
    muted.gain.value = 0;
    const buffer = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("No complete recording received.")), 8000);
      worklet.onprocessorerror = () => { clearTimeout(timeout); reject(new Error("Capture failed.")); };
      worklet.port.onmessage = ({data}) => {
        if (data === "speak") status.textContent = "Speak now";
        else { clearTimeout(timeout); resolve(data); }
      };
      source.connect(worklet).connect(muted).connect(capture.context.destination);
      capture.context.resume().catch(error => { clearTimeout(timeout); reject(error); });
    });
    if (buffer.byteLength !== sampleRate * spec.clip_seconds * Float32Array.BYTES_PER_ELEMENT)
      throw new Error("Recording length did not match the trial.");
    const pcm = encode(buffer);
    rows.push({...current, sample_rate: sampleRate, input_channels: settings.channelCount, pcm_s16le_base64: pcm});
    firstSettings = identity;
    participant.disabled = condition.disabled = true;
    status.textContent = "Recorded.";
  } catch (error) {
    status.textContent = error.message || "Capture failed.";
  } finally {
    closeCapture(capture);
    active = null;
    render();
  }
});

save.addEventListener("click", async () => {
  if (rows.length !== trialList.length) return;
  const data = {protocol: spec.protocol, participant: participant.value, condition: condition.value,
                automated_browser: navigator.webdriver, input_processing: "disabled", rows};
  save.disabled = true;
  status.textContent = "Saving on this Mac…";
  try {
    const response = await fetch("./save", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify(data)});
    status.textContent = await response.text();
    if (!response.ok) save.disabled = false;
  } catch {
    status.textContent = "Could not reach the local recorder. Leave this tab open and try saving again.";
    save.disabled = false;
  }
});

addEventListener("pagehide", () => { if (active) closeCapture(active); });
try {
  const response = await fetch("./protocol.json", {cache: "no-store"});
  if (!response.ok) throw new Error("Could not load the prompts.");
  spec = await response.json();
  for (const [id, name] of Object.entries(spec.conditions)) condition.add(new Option(name, id));
  trialList = spec.pairs.flatMap(({suffix, first}, i) =>
    [first, Object.keys(spec.words).find(id => id !== first)].map(arm =>
      ({id: `p${i + 1}-${arm}`, pair: i + 1, arm, text: spec.words[arm] + suffix})));
  status.textContent = "Ready when you are.";
  render();
} catch (error) {
  status.textContent = error.message;
}
