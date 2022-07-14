import math
import time
import os

import torch
from mido import Message, MidiFile, MidiTrack
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss, TransformerEncoderLayer,
    TransformerEncoder,
    Linear,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader

midi_directory = "lmd_matched"
fmidi = "lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}")

mid = MidiFile(fmidi)

def note(index: int) -> Tensor:
    note_array = [0] * NOTES
    note_array[index] = 1
    return torch.tensor(note_array)

# Embedding dimension
NOTES = 128 + 1 + 1  # START + STOP
WINDOW_SIZE = 50
WINDOW_SLIDE = 20
START = note(NOTES - 2)
STOP = note(NOTES - 1)


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


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
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
        if 'note' not in data:
            continue
        channels.append(note(data['note']))

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
    midi_file = MidiFile(midi_filename)
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
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class TransformerModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(NOTES, dropout=0)
        dim_feedforward = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        encoder_layers = TransformerEncoderLayer(
            d_model=NOTES, nhead=NOTES, dim_feedforward=dim_feedforward, dropout=0, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = Linear(in_features=NOTES, out_features=NOTES)
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
    src_mask = generate_square_subsequent_mask(WINDOW_SIZE - 1)
    eval_data = eval_data
    with torch.no_grad():
        inputs = eval_data[:, :-1, :]
        expected_output = eval_data[:, 1:, :]
        output = model(inputs, src_mask)
        loss = criterion(output, expected_output)
    return loss


def train_model(model, train_loader, criterion):
    model.train()
    print("Epoch", epoch)
    src_mask = generate_square_subsequent_mask(WINDOW_SIZE - 1)
    batch = 0
    log_interval = 10
    total_loss = 0.
    start_time = time.time()
    for window in train_loader:
        window = window.type(torch.FloatTensor)
        # We input has all notes except the last
        # We use all notes to predict the last one
        inputs = window[:, :-1, :]
        # the words we are trying to predict
        expected_output = window[:, 1:, :]
        output = model(
            src=inputs,
            src_mask=src_mask,
        )
        loss = criterion(output, expected_output)
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
            print(f'| epoch {epoch:3d} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    new_song_notes = midi_file_to_batch_entry("new_song.mid")
    print(new_song_notes.shape)
    print("--")
    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(midi_directory):
        midi_files += [os.path.join(dirpath, file) for file in filenames]

    fmidi_notes = torch.empty((0))
    for midi_file in midi_files:
        fmidi_notes = torch.cat((fmidi_notes, midi_file_to_batch_entry(fmidi)))

    print(fmidi_notes.shape)
    train = fmidi_notes
    x_train, x_validation = train_test_split(fmidi_notes, test_size=0.1, train_size=0.9)

    model = TransformerModel()
    model = model.to(device)
    # Other choices: SGD, ...
    optim = AdamW(model.parameters(), lr=5e-5)

    train_loader = DataLoader(x_train, batch_size=16, shuffle=True)
    criterion = CrossEntropyLoss()

    for epoch in range(100):
        train_model(model, train_loader, criterion)
        val_loss = evaluate_model(model, x_validation)
        val_ppl = math.exp(val_loss)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
    print("Apprentissage terminé !!")

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))

    src_mask = generate_square_subsequent_mask(WINDOW_SIZE - 1)
    with torch.no_grad():
        model.eval()
        window = train[0]
        window = window.unsqueeze(0)
        window = window[:, :-1, :]
        for i in range(100):
            output = model(
                src=window,
                src_mask=src_mask)
            print(f"output.shape = {output.shape}")
            output_flat = output.view(-1, NOTES)
            print(output_flat)
            print(output_flat.shape)
            label_id = torch.argmax(output_flat[-1])
            print(f"label_id = {label_id}")
            track.append(Message('note_on', note=label_id, velocity=27, time=12))
            track.append(Message('note_off', note=label_id, velocity=64, time=4))
            window = output

    mid.save('generated.mid')
