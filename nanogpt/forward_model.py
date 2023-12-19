from nanogpt.nanogpt_module import NanoGptPlayer
import torch
torch.set_grad_enabled(False)
player = NanoGptPlayer(model_name="nanogpt")
sample_input = torch.tensor([[15, 6, 4, 27, 9, 0, 25, 10, 0, 7, 4, 19]]).to("cuda")
# print(player.model(sample_input))
print(player.model(sample_input)[0].argmax(dim=-1))