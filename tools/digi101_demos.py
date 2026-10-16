#!/usr/bin/env python3
"""Audition WAVs for DH-101 (the engine alone, as the Digitakt's SRC stage will feed its filter)."""
import os, subprocess, tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = os.path.join(HERE, "digi101_render")   # tools/digi101_render.c
OUT = os.path.join(HERE, "..", "out", "demos")
os.makedirs(OUT, exist_ok=True)
SR = 48000

def r(knobs, seq, secs, sweep=None):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        env = dict(os.environ)
        env.pop("SWEEP", None)
        if sweep:
            env["SWEEP"] = sweep
        subprocess.run([RENDER, f.name, str(secs), " ".join(map(str, knobs)), seq], check=True, env=env)
        return wavfile.read(f.name)[1].astype(np.float64) / 32768

def fade(x, ms=4):
    n = int(SR * ms / 1000)
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x

def gap(s=0.25):
    return np.zeros(int(SR * s))

def save(name, parts, gain=0.7):
    x = np.concatenate(parts) * gain
    wavfile.write(os.path.join(OUT, name), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print(name, "%.1f s" % (len(x) / SR))

# knob order: B RNGE, C SAW, E PULS, F PW, G PWM, H SUB, D NOIS
SAW = [32, 127, 0, 0, 0, 0, 0]

# 1. range: one note (A2) at 16' 8' 4' 2'
save("01_range_16-8-4-2.wav", sum(([fade(r([z] + SAW[1:], "45:1.0", 1.0)), gap()] for z in (0, 32, 64, 96)), []))

# 2. the mixer's sources one at a time, then a classic mix
src = [
    [32, 127, 0, 0, 0, 0, 0],     # saw
    [32, 0, 127, 0, 0, 0, 0],     # square
    [32, 0, 127, 100, 0, 0, 0],   # narrow pulse
    [32, 0, 0, 0, 0, 42, 0],      # sub -1 oct
    [32, 0, 0, 0, 0, 85, 0],      # sub -2 oct
    [32, 0, 0, 0, 0, 127, 0],     # sub -2 oct 25 %
    [32, 0, 0, 0, 0, 0, 127],     # noise
    [32, 110, 0, 0, 0, 34, 0],    # saw + sub -1
    [32, 90, 90, 40, 0, 30, 20],  # saw + pulse + sub + a little noise
]
save("02_sources_saw-square-pulse-sub1-sub2-sub25-noise-mixes.wav",
     sum(([fade(r(k, "45:1.0", 1.0)), gap()] for k in src), []))

# 3. PW by hand: a slow sweep 50 % -> 5 % -> 50 %
a = r([32, 0, 127, 0, 0, 0, 0], "45:2.0", 2.0, sweep="3 0 127")
b = r([32, 0, 127, 127, 0, 0, 0], "45:2.0", 2.0, sweep="3 127 0")
save("03_pw_manual_sweep.wav", [fade(np.concatenate([a, b]))])

# 4. PWM from the LFO: slow, then fast
save("04_pwm_lfo_slow-then-fast.wav",
     [fade(r([32, 0, 127, 110, 25, 0, 0], "45:3.0", 3.0)), gap(),
      fade(r([32, 0, 127, 90, 55, 0, 0], "45:2.0", 2.0))])

# 5. PWM from the envelope: short, then long decay, repeated notes
seq = ",".join(["45:0.5"] * 4)
save("05_pwm_env_short-then-long.wav",
     [fade(r([32, 0, 127, 127, 80, 0, 0], seq, 2.0)), gap(),
      fade(r([32, 0, 127, 127, 115, 0, 0], seq, 2.0))])

# 6. in context: a 16-step bassline through a STAND-IN filter and amp (a resonant 2-pole low-pass with an
# envelope, roughly the Digitakt mk1's) - only to hear the oscillator musically; the real filter is the unit's
bpm, step = 124, 60 / 124 / 4
notes = [33, 33, 45, 33, 36, 33, 45, 43, 33, 33, 45, 33, 31, 33, 40, 43]
seq = ",".join("%d:%.6f" % (n, step) for n in notes * 2)
dry = r([32, 120, 0, 0, 0, 40, 0], seq, step * 32)
dry2 = r([32, 0, 127, 120, 35, 60, 0], seq, step * 32)

def stand_in(x, cut0=180.0, envamt=2200.0, res=0.75, decay=0.12):
    y = np.zeros_like(x)
    lp = bp = 0.0
    per = int(step * SR)
    for i in range(len(x)):
        t = (i % per) / SR
        e = np.exp(-t / decay)
        fc = cut0 + envamt * e
        f = 2 * np.sin(np.pi * fc / SR)
        q = 2 * (1 - res)
        hp = x[i] - lp - q * bp
        bp += f * hp
        lp += f * bp
        amp = np.exp(-t / 0.18) if t < step * 0.9 else np.exp(-t / 0.18) * (1 - (t - step * 0.9) / (step * 0.1))
        y[i] = np.tanh(lp * 1.4) * amp
    return y

save("06_bassline_saw+sub_then_pwm+sub_STANDIN_FILTER.wav",
     [stand_in(dry), stand_in(dry2, cut0=220, envamt=1600, res=0.6)], gain=0.8)
