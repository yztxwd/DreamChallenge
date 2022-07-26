#!python3

import args_parse

FLAGS = args_parse.parse_common_options(
    log_steps=20,
    datadir='./shards/',
    logdir='./conv_model/',
    batch_size=512,
    momentum=0.5,
    lr=0.0001,
    num_cores=1,
    num_epochs=100,
    num_workers=32,
    opts=[('--num_hybrid_conv', {'type': int, 'default': 1}),
          ('--dropout', {'type': float, 'default': 0.5}),
          ('--ckpt', {'type': str})])

import os
import sys
import torch
import lightning_dreamchallenge
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from tqdm import tqdm
from dataloader import DreamChallengeDataModule 

def main(index, flags):
    # load the checkpoint
    print(f"Load model parameters from {flags.ckpt}...")
    checkpoint = torch.load(flags.ckpt)
    model = lightning_dreamchallenge.ConvolutionalModelHybrid(dropout=flags.dropout, num_hybrid_conv=flags.num_hybrid_conv)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load dataset
    datamodule = DreamChallengeDataModule(data_dir=flags.datadir, batch_size=flags.batch_size, num_workers=1)
    datamodule.setup()
    print("setup!")

    # get tpu
    device = xm.xla_device()
    model = model.to(device)

    def pred_loop_fn(loader):
      model.eval()
      pbar = tqdm(enumerate(loader))
      preds = []
      for step, (data, target) in pbar:
        data = torch.squeeze(data, 0)
        pred = model(data)
        preds.append(pred)
      preds = torch.vstack(preds)

      # save the prediction
      np.savetxt(os.path.join(os.path.dirname(flags.ckpt), "model_preds.txt"), preds.detach().cpu().numpy(), fmt="%.6f")
    
      return preds

    # predict
    pred_loader = datamodule.pred_dataloader()
    pred_device_loader = pl.MpDeviceLoader(pred_loader, device)
    pred_loop_fn(pred_device_loader)

if __name__ == "__main__":
    xmp.spawn(main, args=(FLAGS,), nprocs=FLAGS.num_cores)