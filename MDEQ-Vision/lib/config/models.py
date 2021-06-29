# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

MDEQ = CN()
MDEQ.FULL_STAGE = CN()
MDEQ.FULL_STAGE.NUM_MODULES = 1
MDEQ.FULL_STAGE.NUM_BRANCHES = 4
MDEQ.FULL_STAGE.NUM_BLOCKS = [1, 1, 1, 1]
MDEQ.FULL_STAGE.NUM_CHANNELS = [64, 128, 256, 512]
MDEQ.FULL_STAGE.BIG_KERNELS = [0, 0, 0, 0]
MDEQ.FULL_STAGE.HEAD_CHANNELS = [32, 64, 128, 256]    # Only for classification
MDEQ.FULL_STAGE.FINAL_CHANSIZE = 2048                 # Only for classification
MDEQ.FULL_STAGE.BLOCK = 'BASIC'
MDEQ.FULL_STAGE.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'mdeq': MDEQ
}
