## Install GPU-enabled PyTorch

After running `uv install` to get your CPU-only dependencies, install the CUDA 12.6 wheels with:

```bash
uv pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  --trusted-host download.pytorch.org \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126
