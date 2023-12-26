
   
#!/usr/bin/env python

# specified for pcl pre-trained model

import pickle as pkl
import sys
import torch

if __name__ == "__main__":

    obj = torch.load('path/to/CA2CL.tar', map_location="cpu")
    obj = obj["net"]

    newmodel = {}
    for k, v in obj.items():
        old_k = k
        k = k.replace("module.resnet.", "")
        if 'instance_projector' in k or 'groupDis' in k:
            print('skip')
            continue
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "MOCO", "matching_heuristics": True}

    with open(sys.argv[1], "wb") as f:
        pkl.dump(res, f)