class PilotRecorder extends AudioWorkletProcessor {
  constructor({processorOptions: {samples}}) {
    super();
    this.pcm = new Float32Array(samples);
    this.offset = 0;
    this.cued = false;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel?.length) return true;
    const count = Math.min(channel.length, this.pcm.length - this.offset);
    this.pcm.set(channel.subarray(0, count), this.offset);
    this.offset += count;
    if (!this.cued && this.offset >= sampleRate * .4) {
      this.cued = true;
      this.port.postMessage("speak");
    }
    if (this.offset === this.pcm.length) {
      this.port.postMessage(this.pcm.buffer, [this.pcm.buffer]);
      return false;
    }
    return true;
  }
}

registerProcessor("pilot-recorder", PilotRecorder);
