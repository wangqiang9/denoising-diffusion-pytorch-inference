from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import torch


if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4,5,7'

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)
model = torch.nn.DataParallel(model)
model = model.cuda()


diffusion = GaussianDiffusion(
    model,
    image_size=256,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './datasets/handbagstrainB/',
    image_size=256,
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=7000000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    fp16=True,  # turn on mixed precision training with apex
    results_folder='./models/test_attention'
)

trainer.train()
