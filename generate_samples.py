from templates import *
from templates_latent import *
import PIL.Image
import matplotlib.pyplot as plt
import pathlib

device = 'cuda:0'
conf = ffhq256_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))
model.to(device);


torch.manual_seed(4)
output_path = "outputs/"
pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
count = 0
for idx in range(1000):
    img = model.sample(1, device=device)
    PIL.Image.fromarray(img[0].permute(1, 2, 0).cpu(), 'RGB').save(output_path + f'{str(count).zfill(5)}' + '.png')
    count = count + 1