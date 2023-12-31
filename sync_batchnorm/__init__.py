
from .batchnorm import (
    SynchronizedBatchNorm1d,
    SynchronizedBatchNorm2d,
    SynchronizedBatchNorm3d,
)
from .batchnorm import patch_sync_batchnorm, convert_model
from .replicate import DataParallelWithCallback, patch_replication_callback
