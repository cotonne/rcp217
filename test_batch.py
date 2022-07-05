import torch
import numpy as np
from mido import Message
from .main import BLANK, WINDOW_SIZE, NOTES, slice_midi_tracks, MidiFileToBatchEntry


MIDI_NOTE = 69


def build_a_note(value=MIDI_NOTE):
    return Message('note_on', note=value, velocity=27, time=12)


def test_slicing_a_lot_more_of_notes():
    MISSING = 5
    midi_tracks = [build_a_note() for _ in range(0, 60 - MISSING)]
    midi_slices = slice_midi_tracks(midi_tracks)
    note = [0] * NOTES
    note[MIDI_NOTE] = 1
    note = torch.tensor(note)
    slice = torch.stack([
        torch.stack([note] * WINDOW_SIZE),
        torch.stack([note] * (WINDOW_SIZE - 15) +
                    [BLANK] * 15),
        torch.stack([note] * (WINDOW_SIZE - 35) +
                    [BLANK] * 35)
    ])
    print(np.where(midi_slices.numpy() == 1))
    print(np.where(slice.numpy() == 1))
    assert torch.equal(midi_slices, slice)


def test_slicing_with_special_messages():
    MISSING = 5
    midi_tracks = [Message('program_change', program=12, time=0)] + \
        [build_a_note() for _ in range(0, 10)]
    midi_slices = slice_midi_tracks(midi_tracks)
    note = [0] * NOTES
    note[MIDI_NOTE] = 1
    note = torch.tensor(note)
    slice = torch.stack([
        torch.stack([note] * 10 + [BLANK] * 40),
    ])
    print(np.where(midi_slices.numpy() == 1))
    print(np.where(slice.numpy() == 1))
    assert torch.equal(midi_slices, slice)


def test_slicing_with_only_special_messages():
    MISSING = 5
    midi_tracks = [Message('program_change', program=12, time=0)]
    midi_slices = slice_midi_tracks(midi_tracks)
    assert midi_slices == None


def test_read_midi():
    new_song_notes = MidiFileToBatchEntry("new_song.mid")
    note = [0] * NOTES
    note[MIDI_NOTE] = 1
    note = torch.tensor(note)
    slice = torch.stack([
        torch.stack([note] * 2 + [BLANK] * (WINDOW_SIZE - 2)),
    ])
    assert torch.equal(new_song_notes, slice)
