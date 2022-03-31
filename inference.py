from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import torch

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './datasets/shoetrainB/',
    train_batch_size=48,
    train_lr=2e-5,
    train_num_steps=7000000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    fp16=True,  # turn on mixed precision training with apex
    results_folder='./results_multi'
)

# load_path = './results_shoe/model-20.pt'
# save_path = './save/save_shoe/baseline1-model20'
load_path = './results_handbags/model-87.pt'
save_path = './save/save_handbags/baseline1-model87'
loop_num = 1000
trainer.inference(load_path=load_path, save_path=save_path, loop_num=loop_num)
