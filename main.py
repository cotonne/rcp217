import math
import time
import os

import torch
from mido import Message, MidiFile, MidiTrack
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss,
    MSELoss,
    TransformerEncoderLayer,
    TransformerEncoder,
    Linear,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from typing import List
from heapq import heappush, heappop, nsmallest
from random import randint

BATCH_SIZE = 16

midi_directory = "lmd_matched"
fmidi = "lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set to True to quickly test the processing pipeline (load data, learn, generate...)
TEST = False
# If True, train the model and save the model weights else load the weights
TRAIN_MODEL = True
# Path where the weights of the model are saved
PATH = "model.h5"
DATASET_SIZE = 10000
INDEX_ENTRY = 0 if TEST else randint(0, 10000)
EPOCHS = 1 if TEST else 500
# Embedding dimension
MIDI_NOTES = 128
NOTES = MIDI_NOTES + 1 + 1  # START + STOP
VELOCITY_INDEX = NOTES
DURATION_INDEX = NOTES + 1
ENTRY_SIZE = NOTES + 1 + 1  # Velocity + Duration
WINDOW_SIZE = 50
WINDOW_SLIDE = 30
WINDOW_INPUT = WINDOW_SIZE - 1
MAX_VELOCITY = 127

print(f"Using device {device}")


def note(index: int, velocity: int) -> Tensor:
    note_array = [0] * ENTRY_SIZE
    note_array[index] = 1
    note_array[VELOCITY_INDEX] = velocity
    note_array[DURATION_INDEX] = 0  #TODO
    return torch.tensor(note_array)


START = note(NOTES - 2, 0)
STOP = note(NOTES - 1, 0)


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


#
class PositionalEncoding(torch.nn.Module):
    """
    Append positional information to the input vector for the transformer
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def slice_midi_tracks(tracks):
    channels = []
    for msg in tracks:
        data = msg.dict()
        if 'note' not in data or data['type'] != 'note_on':
            continue
        channels.append(note(data['note'], data['velocity'] / MAX_VELOCITY))

    slices = []
    for i in range(0, len(channels), WINDOW_SLIDE):
        list_size = min(WINDOW_SIZE, len(channels) - i)
        slices.append(torch.stack(
            channels[i:i + list_size] + [STOP] * (WINDOW_SIZE - list_size)
        ))
    if len(slices) == 0:
        return None
    return torch.stack(slices)


def midi_file_to_batch_entry(midi_filename):
    tracks = []
    try:
        midi_file = MidiFile(midi_filename)
    except Exception as e:
        print(f"Fail to read file. Current file : {midi_filename}")
        print(e)
        return torch.empty((0))
    # print(f"midi_file = {midi_file.ticks_per_beat }")
    for i, track in enumerate(midi_file.tracks):
        channels = slice_midi_tracks(track)
        if channels is None:
            continue
        tracks.append(channels)
    if len(tracks) == 0:
        return torch.empty((0))
    return torch.cat(tracks).type(torch.FloatTensor)


def generate_square_subsequent_mask(sz: int):
    """
    Generate triangular matrix to hide future notes
    ```
        >>> generate_square_subsequent_mask(5)
        tensor([[0., -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0.]])
    ```
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)


class TransformerModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ENTRY_SIZE, dropout=0.2)
        dim_feedforward = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        encoder_layers = TransformerEncoderLayer(
            d_model=ENTRY_SIZE, nhead=ENTRY_SIZE, dim_feedforward=dim_feedforward, dropout=0.2, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = Linear(in_features=ENTRY_SIZE, out_features=ENTRY_SIZE)
        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor):
        output = self.pos_encoder(src)
        output = self.transformer_encoder(output, src_mask)
        output = self.decoder(output)
        return output


def evaluate_model(model: torch.nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    src_mask = generate_square_subsequent_mask(WINDOW_INPUT)
    eval_data = eval_data
    with torch.no_grad():
        inputs = eval_data[:, :-1, :]
        output = model(inputs.to(device), src_mask)
        expected_output = eval_data[:, 1:, :].to(device)
        expected_output_notes = expected_output[:, :, :NOTES]
        expected_output_velocity = expected_output[:, :, VELOCITY_INDEX]
        output_notes = output[:, :, :NOTES]
        output_velocity = output[:, :, VELOCITY_INDEX]
        loss = criterion_notes(output_notes, expected_output_notes) + criterion_velocity(output_velocity, expected_output_velocity)
    return loss


def train_model(model, train_loader, criterion):
    model.train()
    src_mask = generate_square_subsequent_mask(WINDOW_INPUT)
    batch = 0
    log_interval = 100
    total_loss = 0.
    start_time = time.time()
    for window in train_loader:
        # We input has all notes except the last
        # We use all notes to predict the last one
        inputs = window[:, :-1, :]
        output = model(
            src=inputs.to(device),
            src_mask=src_mask,
        )
        # the notes we are trying to predict
        expected_output = window[:, 1:, :].to(device)
        expected_output_notes = expected_output[:, :, :NOTES]
        expected_output_velocity = expected_output[:, :, VELOCITY_INDEX]
        output_notes = output[:, :, :NOTES]
        output_velocity = output[:, :, VELOCITY_INDEX]
        loss = criterion_notes(output_notes, expected_output_notes) + criterion_velocity(output_velocity, expected_output_velocity)
        optim.zero_grad()
        loss.backward()
        # To avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        total_loss += loss.item()
        batch += 1
        if batch % log_interval == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'train;{epoch};{cur_loss};{ms_per_batch}')
            total_loss = 0
            start_time = time.time()


def save_to_midi(notes: List[int], filename='generated.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=12, time=0))
    for note in notes:
        track.append(Message('note_on', note=note, velocity=100, time=12))
        track.append(Message('note_off', note=note, velocity=100, time=4))
    mid.save(filename)


def load_dataset():
    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(midi_directory):
        midi_files += [os.path.join(dirpath, file) for file in filenames]
    f_midi_notes = torch.empty((0))
    if TEST:
        midi_files = midi_files[:1]
    for midi_file in midi_files:
        f_midi_notes = torch.cat((f_midi_notes, midi_file_to_batch_entry(midi_file)))
    return f_midi_notes


if __name__ == "__main__":
    new_song_notes = midi_file_to_batch_entry("new_song.mid")
    print(new_song_notes.shape)
    print("--")
    fmidi_notes = load_dataset()

    model = TransformerModel()
    model = model.to(device)
    if TRAIN_MODEL:
        fmidi_notes = fmidi_notes[:DATASET_SIZE, :, :]
        print(fmidi_notes.shape)
        train = fmidi_notes.to(device)
        x_train, x_validation = train_test_split(fmidi_notes, test_size=0.1, train_size=0.9)
        x_train = x_train
        x_validation = x_validation
        # Other choices: SGD, ...
        optim = AdamW(model.parameters(), lr=5e-5)

        train_loader = DataLoader(x_train, batch_size=BATCH_SIZE, shuffle=True)
        criterion_notes = CrossEntropyLoss()
        criterion_velocity = MSELoss()

        print(f'type;epoch;loss;others')
        for epoch in range(EPOCHS):
            train_model(model, train_loader, criterion_notes)
            test_loss = evaluate_model(model, x_validation)
            print(f'test;{epoch};{test_loss};')
        print("Apprentissage terminé !!")

        torch.save(model.state_dict(), "model.h5")

    else:
        model.load_state_dict(torch.load(PATH))


    src_mask = generate_square_subsequent_mask(WINDOW_INPUT)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    with torch.no_grad():
        model.eval()
        data = fmidi_notes[INDEX_ENTRY][:-1, :].to(device)
        dataset = data.unsqueeze(0)
        h = []
        heappush(h, (0, dataset))
        for i in range(100):
            total_log_prob, dataset = heappop(h)
            output_all = model(
                src=dataset[:, -WINDOW_INPUT:, :],
                src_mask=src_mask)
            output = output_all[:, :, :NOTES]
            output_flat = -log_softmax(output.view(-1, NOTES))
            label_id = torch.argmin(output_flat[-1]).cpu().item()
            log_prob = output_flat[-1][label_id].cpu().item()
            print(f"label_id = {label_id}, log_prob = {log_prob}")
            new_note = note(label_id, 63).unsqueeze(0).to(device)
            data = torch.cat((data, new_note))
            dataset = data.unsqueeze(0)
            total_log_prob += log_prob
            if label_id >= MIDI_NOTES:
                print("Final result")
            heappush(h, (total_log_prob, dataset))
            h = nsmallest(3, h)

    _, result = np.where(fmidi_notes[INDEX_ENTRY][:-1, :NOTES].cpu())
    result = result.tolist()
    prob, t = h[0]
    t = t[:, :, :NOTES]
    generated = np.where(t.cpu())[2]
    result = result + generated.tolist()

    save_to_midi(result)

    # print([layer.self_attn for layer in model.transformer_encoder.layers])
