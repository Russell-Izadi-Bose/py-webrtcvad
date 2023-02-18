"""    
This script performs voice activity detection on a single file in the LibriSpeech dataset.
"""

import sys
import librosa
import webrtcvad
import numpy as np
import soundfile

def binary_sequence_to_events(binary_sequence, frame_length_ms, label='speech'):
    # convert binary sequence to list of integers
    int_sequence = [int(i) for i in binary_sequence]

    # initialize variables
    start_time = 0
    end_time = 0
    events = []

    frame_length_s = frame_length_ms / 1000
    # loop through the sequence and find events
    for i in range(len(int_sequence)):
        if int_sequence[i] == 1 and end_time == 0:
            # event has started
            start_time = i * frame_length_s
            end_time = start_time
        elif int_sequence[i] == 0 and end_time != 0:
            # event has ended
            end_time = i * frame_length_s
            events.append((start_time, end_time, label))
            start_time = 0
            end_time = 0

    # if there is an event that has not ended
    if end_time != 0:
        events.append((start_time, len(int_sequence) * frame_length_s, label))

    return events


def main(args):
    assert len(args) == 3

    path_file, mode, frame_length_ms = args

    info = soundfile.info(path_file)
    print(info)

    assert frame_length_ms in (10, 20, 30)
    sr = librosa.get_samplerate(path_file)
    assert sr in (8000, 16000, 32000, 48000)
    print('sr', sr)

    y, sr = soundfile.read(path_file)
    print(y.shape, y.dtype, y.min(), y.max())
    frame_length = int(sr * frame_length_ms / 1000)
    print(frame_length)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length)
    print(frames.shape)

    vad = webrtcvad.Vad(mode)
    output = []
    for frame in frames.T:
        ints = (frame * 32767).astype(np.int16)
        little_endian = ints.astype('<u2')
        buf = little_endian.tobytes()
        is_speech = vad.is_speech(buf, sr)
        output.append(is_speech)

    output = np.array(output)
    print(output.shape)

    # plot to png
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].set_title(f"mode:{mode}, frame:{frame_length_ms} (ms)")
    axes[0].plot(y)
    time = np.linspace(0, y.shape[-1]-1, output.shape[-1])
    axes[1].plot(time, output)
    axes[1].xaxis.set_ticks(axes[1].get_xticks())
    axes[1].xaxis.set_ticklabels([f"{x / sr:.0f}" for x in axes[1].get_xticks()])
    plt.savefig("vad.png")

    # save to txt
    events = binary_sequence_to_events(output, frame_length_ms)
    with open("vad.txt", "w") as f:
        for event in events:
            f.write(f"{event[0]:.3f}\t{event[1]:.3f}\t{event[2]}\n")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mode = 3
        frame_length_ms = 10
        path_file = "/ext/datasets/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
        args = path_file, mode, frame_length_ms
    else:
        args = sys.argv[1:]
    main(args)
