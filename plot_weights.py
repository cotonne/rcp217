import main
import torch

model = main.TransformerModel()
model = model.to(main.device)
model.load_state_dict(torch.load(main.PATH))

attention_layer = model.transformer_encoder.layers[0].self_attn.out_proj
weights = attention_layer.state_dict()['weight'].tolist()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = weights
fig, ax = plt.subplots()
im = ax.pcolormesh(data, cmap=cm.gray, edgecolors='white', linewidths=0,
                   antialiased=True)
fig.colorbar(im)

plt.show()