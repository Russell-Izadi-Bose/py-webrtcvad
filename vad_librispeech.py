
import os
import sys
import glob
import librosa
import webrtcvad
import numpy as np
import soundfile

def vad(path, mode, frame_length_ms):
    assert frame_length_ms in (10, 20, 30)
    sr = librosa.get_samplerate(path)
    assert sr in (8000, 16000, 32000, 48000)

    y, sr = soundfile.read(path)
    frame_length = int(sr * frame_length_ms / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length)

    vad = webrtcvad.Vad(mode)
    output = []
    for frame in frames.T:
        ints = (frame * 32767).astype(np.int16)
        little_endian = ints.astype('<u2')
        buf = little_endian.tobytes()
        is_speech = vad.is_speech(buf, sr)
        output.append(is_speech)

    output = np.array(output)
    return output


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


def vad_librispeech(path_in, path_out, subfoler='test-clean', mode=3, frame_length_ms=10):

    files = glob.glob(os.path.join(path_in, subfoler, '**/**.flac'), recursive=True)

    for i_file, file in enumerate(files):
        print(i_file+1, len(files))

        binary_sequence = vad(file, mode, frame_length_ms)
        events = binary_sequence_to_events(binary_sequence, frame_length_ms, label='speech')

        txt_file = os.path.join(path_out, file[len(path_in)+1:]).replace('.flac', '.txt')

        # Create the parent directories if they do not exist
        os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        with open(txt_file, "w") as f:
            for event in events:
                f.write(f"{event[0]:.3f}\t{event[1]:.3f}\t{event[2]}\n")
        break


if __name__ == '__main__':
    if len(sys.argv) == 1:
        path_in = "/ext/datasets/LibriSpeech"
        path_out = "/ext/projects/brooklyn_bridge/data/vad_librispeech"
        subfoler = 'test-clean'
        mode = 3
        frame_length_ms = 10
        args = path_in, path_out, subfoler, mode, frame_length_ms
    elif len(sys.argv) == 6:
        path_in, path_out, subfoler, mode, frame_length_ms = sys.argv[1:]

    vad_librispeech(path_in, path_out, subfoler, mode, frame_length_ms)
