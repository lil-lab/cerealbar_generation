# TODO: Clean this code.

import numpy as np
import torch
from torch.autograd import Variable


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def __eq__(self, other):
        if other is None:
            return False

        poseq = self.position == other._position
        roteq = self.orientation == other._orientation
        # For numpy arrays, torch ByteTensors and variables containing ByteTensors
        if hasattr(poseq, "all"):
            poseq = poseq.all()
            roteq = roteq.all()
        return poseq and roteq

    def __getitem__(self, i):
        if type(i) in [torch.ByteTensor, torch.cuda.ByteTensor]:
            pos = self.position[i[:, np.newaxis].expand_as(self.position)].view([-1, 3])
            rot = self.orientation[i[:, np.newaxis].expand_as(self.orientation)].view([-1, 4])
            return Pose(pos, rot)
        else:
            return Pose(self.position[i], self.orientation[i])

    def cuda(self, device=None):
        self.position = self.position.cuda(device)
        self.orientation = self.orientation.cuda(device)
        return self

    def to_torch(self):
        _position = torch.from_numpy(self.position)
        _orientation = torch.from_numpy(self.orientation)
        return Pose(_position, _orientation)

    def to_var(self):
        _position = Variable(self.position)
        _orientation = Variable(self.orientation)
        return Pose(_position, _orientation)

    def repeat_np(self, batch_size):
        _position = np.tile(self.position[np.newaxis, :], [batch_size, 1])
        _orientation = np.tile(self.orientation[np.newaxis, :], [batch_size, 1])
        return Pose(_position, _orientation)

    def numpy(self):
        pos = self.position
        rot = self.orientation
        if isinstance(pos, Variable):
            pos = pos.data
            rot = rot.data
        if hasattr(pos, "cuda"):
            pos = pos.cpu().numpy()
            rot = rot.cpu().numpy()
        return Pose(pos, rot)

    def __len__(self):
        if self.position is None:
            return 0
        return len(self.position)

    def __str__(self):
        return "Pose " + str(self.position) + " : " + str(self.orientation)
