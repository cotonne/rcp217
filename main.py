import math

import numpy as np
import torch
import torch.nn.functional as F
from mido import MidiFile
from torch.nn import Transformer
from torch.optim import AdamW
from torch.utils.data import DataLoader

fmidi = "lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid"

mid = MidiFile(fmidi)

# Embedding dimension
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


def generate_square_subsequent_mask(sz: int):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class TransformerModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(NOTES, dropout=0)
        self.model = Transformer(nhead=NOTES, num_encoder_layers=12,
                                 d_model=NOTES, batch_first=True)

    def forward(self, x, has_mask=True):
        output = self.pos_encoder(x)
        output = self.model(output)
        return F.log_softmax(output, dim=-1)


if __name__ == "__main__":
    new_song_notes = MidiFileToBatchEntry("new_song.mid")
    print(new_song_notes.shape)
    print("--")
    fmidi_notes = MidiFileToBatchEntry(fmidi)
    print(fmidi_notes.shape)
    train = fmidi_notes
    device = 'cpu'  # 'cuda'

    model = TransformerModel()
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(1):
        print("Epoch", epoch)
        for window in train_loader:
            window = window.type(torch.FloatTensor)
            optim.zero_grad()
            outputs = model(
                src=window,
                tgt=window,
            )
            loss = outputs[0]
            loss.sum().backward()

    with torch.no_grad():
        model.eval()
        window = train[0].type(torch.FloatTensor)
        outputs = model(
            src=window,
            tgt=window,
            tgt_mask=generate_square_subsequent_mask(50))
        label_id = torch.argmax(outputs.logits.squeeze())
        print(f"label_id = {label_id}")

    print("Apprentissage terminé !!")
