import torch
from mido import MidiFile
import numpy as np

fmidi = "lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid"

mid = MidiFile(fmidi)

NOTES = 128 + 1  # BLANK
WINDOW_SIZE = 50
WINDOW_SLIDE = 20
BLANK = [0] * NOTES
BLANK[NOTES - 1] = 1
BLANK = torch.tensor(BLANK)
# for i, track in enumerate(mid.tracks):
#     print('Track {}: {}'.format(i, track.name))
#     channels = []
#     for msg in track:
#         data = msg.dict()
#         if 'channel' in data:
#             channels.append(data['channel'])
#             print(msg.dict())
#         else:
#             pass
#             print(msg.dict())
#     print(f"Channels = {set(channels)}")


def slice_midi_tracks(tracks):
    channels = []
    for msg in tracks:
        data = msg.dict()
        note = [0] * NOTES
        if 'note' not in data:
            continue
        note[data['note']] = 1
        note = torch.tensor(note)
        channels.append(note)

    slices = []
    for i in range(0, len(channels), WINDOW_SLIDE):
        list_size = min(WINDOW_SIZE, len(channels) - i)
        slices.append(torch.stack(
            channels[i:i + list_size] + [BLANK] * (WINDOW_SIZE - list_size)
        ))
    if len(slices) == 0:
        return None
    return torch.stack(slices)


def MidiFileToBatchEntry(fmidi):
    tracks = []
    mid = MidiFile(fmidi)
    for i, track in enumerate(mid.tracks):
        channels = slice_midi_tracks(track)
        if channels == None:
            continue
        tracks.append(channels)
    if len(tracks) == 0:
        return torch.empty((0))
    return torch.cat(tracks)


if __name__ == "__main__":
    new_song_notes = MidiFileToBatchEntry("new_song.mid")
    print(new_song_notes.shape)
    print("--")
    fmidi_notes = MidiFileToBatchEntry(fmidi)
    print(fmidi_notes.shape)
