from __future__ import division

import numpy as np
import scipy.io.wavfile as wf

# This code stolen off another project of mine.
class AudioSymbol:
    def __init__(self, duration):
        self.duration = duration
    def emit(self, samplecount, samplerate, prev_theta):
        # -> (samples, new_theta)
        raise NotImplementedError()

class Silence(AudioSymbol):
    def emit(self, samplecount, samplerate, prev_theta):
        return (np.zeros(samplecount), 0)

class Tone(AudioSymbol):
    def __init__(self, frequency, duration, amplitude=1.0, phase=None):
        AudioSymbol.__init__(self, duration)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
    def emit(self, samplecount, samplerate, prev_theta):
        rfreq = 2*np.pi * self.frequency / samplerate
        phase = self.phase
        if phase is None:
            phase = prev_theta

        res = np.sin(np.arange(0, samplecount) * rfreq + phase) * self.amplitude
        return (res, samplecount * rfreq + phase)

class LinearSweep(AudioSymbol):
    def __init__(self, freqstart, freqend, duration, amplitude=1.0, phase=None):
        AudioSymbol.__init__(self, duration)
        self.freqstart = freqstart
        self.freqend = freqend
        self.amplitude = amplitude
        self.phase = phase
    def emit(self, samplecount, samplerate, prev_theta):
        phase = self.phase
        if phase is None:
            phase = prev_theta

        freqs = np.linspace(self.freqstart, self.freqend, samplecount, endpoint=True)
        pts = np.zeros(samplecount+1)
        pts[0] = phase
        pts[1:] = freqs * (2*np.pi / samplerate)
        phases = np.cumsum(pts)
        return (np.sin(phases[:-1]), phases[-1])

def gen_wav(syms, samplerate):
    total_dur = sum(sym.duration for sym in syms) * samplerate

    out = np.zeros(total_dur)
    pos = 0
    curduration = 0
    last_theta = 0
    for sym in syms:
        # keep the time synced closely to the sample rate
        symdur = int(samplerate * (curduration - pos / samplerate + sym.duration) + 1e-3)
        out[pos:pos+symdur], last_theta = sym.emit(symdur, samplerate, last_theta)
        pos += symdur
        curduration += sym.duration
    return (out.flatten() * 32768 - 0.5).astype('int16')

def vox_signal():
    return [Tone(f, 0.1) for f in (1900, 1500, 1900, 1500, 2300, 1500, 2300, 1500)]

def vis_signal(id):
    # VIS header
    out = [Tone(1900, 0.3), Tone(1200, 0.01), Tone(1900, 0.3)]

    # add even parity bit
    parity = sum(map(int, '{:07b}'.format(id))) % 2
    vis = id | (parity * 0x80)

    out += [Tone(1200, 0.03)] # start bit
    out += [Tone(1100 if (vis & (1<<i)) else 1300, 0.03) for i in xrange(8)]
    out += [Tone(1200, 0.03)] # stop bit

    return out

def eof_signal():
    return [Tone(f, 0.1) for f in (1900, 1500, 1900, 1500)]

def convert_flag_row(row, dur, samplerate):
    maxpx = int(dur * samplerate)
    assert maxpx >= len(row), "Not enough flag space: max width %d px" % maxpx
    res = []

    padding = dur - len(row)/samplerate
    res.append(Tone(2300, padding / 2.0))

    for px in row:
        # deliberately turn down the contrast a bit to make it
        # easier to pick out the flag from the padding (if slightly misaligned)
        res.append(Tone(1550 + 700 * px / 255.0, 1.0/samplerate))

    res.append(Tone(2300, padding / 2.0))
    return res

def gen_rows(samplerate):
    from PIL import Image

    w = 320
    h = 256

    cover_img = np.asarray(Image.open('cover.png'))
    flag_img = np.asarray(Image.open('flag.png'))
    flag_x, flag_y, flag_w, flag_h = 258, 44, 16, 16

    cover_freqs = (cover_img / 255.0 * 800) + 1500

    # generate 256 rows.
    # each row: G, R, then B, with 1500Hz=0%, 2300Hz=100%
    colordur = 0.13974

    for y in xrange(h):
        row = []
        for chan in (1, 2, 0):
            if chan == 0:
                row.append(Tone(1200, 0.009)) # hsync

            crow = []
            crow.append(Tone(1500, colordur/w/2))
            for x in xrange(w-1):
                crow.append(LinearSweep(cover_freqs[y,x,chan], cover_freqs[y,x+1,chan],
                    colordur/w))
            crow.append(Tone(1500, colordur/w/2))

            if flag_y <= y < flag_y + flag_h:
                crow[flag_x:flag_x+flag_w] = convert_flag_row(flag_img[y-flag_y,:,chan],
                    colordur * flag_w / w, samplerate)

            row += crow
        yield row

if __name__ == '__main__':
    samplerate = 96000

    sstv = [Silence(0.1)]
    sstv += vox_signal()
    sstv += vis_signal(60) # Scottie S1

    sstv += [Tone(1200, 0.009)]
    for row in gen_rows(samplerate):
        sstv += row

    sstv += eof_signal()
    sstv += [Silence(0.1)]

    wf.write('public/terebeep.wav', samplerate, gen_wav(sstv, samplerate=samplerate))
