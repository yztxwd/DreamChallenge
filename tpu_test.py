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
          ('--names', {'type': str}),
          ('--ckpts', {'type': str})])

import os
import sys
import torch
import lightning_dreamchallenge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from tqdm import tqdm
from dataloader import DreamChallengeDataModule 

def main(flags):
    ckpt = flags.ckpts
    name = flags.names
    dropout = flags.dropout
    num_hybrid_conv =flags.num_hybrid_conv
    device = xm.xla_device()
    ordinal = xm.get_ordinal()
    print(f"ckpt {ckpt} on device xla:{ordinal}")
    # initialize model
    model = lightning_dreamchallenge.ConvolutionalModelHybrid(dropout=dropout, num_hybrid_conv=num_hybrid_conv)
    model = model.to(device)
    # load the checkpoint
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Dropout is {dropout}, num hybrid is {num_hybrid_conv}")
    print(f"Best model loaded from epoch {checkpoint['epoch']}")
    print(f"Load model parameters from {ckpt}...")

    # load dataset
    datamodule = DreamChallengeDataModule(data_dir=flags.datadir, batch_size=flags.batch_size, num_workers=1)
    datamodule.setup()

    def pred_loop_fn(loader):
      model.eval()
      pbar = tqdm(enumerate(loader))
      preds = []
      targets = []
      for step, (data, target) in pbar:
        print(1)
        data = torch.squeeze(data, 0)
        print(2)
        pred = model(data)
        print(3)
        preds.append(pred)
        targets.append(target)
      print("Stack results")
      preds = torch.vstack(preds)
      targets = torch.vstack(targets)

      print("Ready to gather results...")
      if xm.xrt_world_size() > 1:
        preds = xm.all_gather(preds)
        targets = xm.all_gather(preds)
      else:
        print("Skip gather cuz there is only one process")

      preds_cpu = preds.detach().cpu().numpy().flatten()
      targets_cpu = targets.detach().cpu().numpy().flatten()
      if xm.is_master_ordinal():
        # save the prediction
        np.savetxt(os.path.join(os.path.dirname(ckpt), name + "model_test_predictions.txt"), preds_cpu, fmt="%.6f")
        np.savetxt(os.path.join(os.path.dirname(ckpt), name + "model_test_targets.txt"), targets_cpu, fmt="%.6f")
    
      return preds_cpu, targets_cpu

    # predict
    pred_loader = datamodule.test_dataloader()
    pred_device_loader = pl.MpDeviceLoader(pred_loader, device)
    preds_cpu, targets_cpu = pred_loop_fn(pred_device_loader)

    # plot correlation 
    fig = plt.figure(figsize=(12, 12))
    sns.scatterplot(x=preds_cpu, y = targets_cpu)
    plt.savefig(os.path.join(os.path.dirname(ckpt), name + "model_test_scatter.pdf"))

def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  main(flags)

if __name__ == "__main__":
    print(f"Spawn {FLAGS.num_cores} cores ...")
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)