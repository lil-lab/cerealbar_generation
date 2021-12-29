"""Various utilities dealing with hex coordinates mapping onto tensors representing the environment.

See: https://arxiv.org/pdf/1803.02108.pdf

Authors: Alane Suhr and Noriyuki Kojima
"""
import os
import time
import math
from dataclasses import dataclass

import torch

from agent.environment import position
from ..map_transformations import pose
from .. import util

from typing import Tuple

@dataclass
class AxialPosition:
    # Axis indices
    u: int
    # Row indices
    v: int

    def __str__(self):
        return '%s, %s' % (self.u, self.v)

    def __hash__(self):
        return (self.u, self.v).__hash__()


@dataclass
class CubePosition:
    x: int
    y: int
    z: int


def offset_position_to_axial(offset_position: position.Position, add_u: int = 0, add_v: int = 0) -> AxialPosition:
    """Converts from offset coordinates to axial coordinates.

    Inputs:
        offset_position: Position in offset coordinates.
        max_y: The maximum y-value for an environment, so that coordinates are not negative.
    """
    u = offset_position.x - offset_position.y // 2

    # V and Y are equivalent (rows).
    return AxialPosition(u + add_u, offset_position.y + add_v)


def axial_position_to_offset(axial_position: AxialPosition, max_y: int = 0) -> position.Position:
    """Converts from axial to offset coordinates.

    Inputs:
        axial_position: Position in axial coordinates.
        max_y: The maximum y-value for an environment, to account for offset in axial coordinates that comes from
            avoiding negative u-values.
    """
    x = axial_position.u + axial_position.v // 2 - max_y // 2
    return position.Position(x, axial_position.v)


def axial_position_to_cube(axial_position: AxialPosition) -> CubePosition:
    return CubePosition(axial_position.v, -(axial_position.u + axial_position.v), axial_position.u)


def cube_position_to_axial(cube_position: CubePosition) -> AxialPosition:
    return AxialPosition(cube_position.z, cube_position.x)


def rotate_counterclockwise(axial_position: AxialPosition, u_offset: int = 0, v_offset: int = 0) -> AxialPosition:
    axial_position = AxialPosition(axial_position.u - u_offset, axial_position.v - v_offset)
    cube_position = axial_position_to_cube(axial_position)
    rotated_cube = CubePosition(- cube_position.z, -cube_position.x, -cube_position.y)
    rotated_axial = cube_position_to_axial(rotated_cube)
    return AxialPosition(rotated_axial.u + u_offset, rotated_axial.v + v_offset)


def _get_batch_index_tensor(batch_size: int, env_height: int, env_width: int):
    index_array = torch.tensor([i for i in range(batch_size)])
    index_tensor = index_array.repeat(env_height, env_width, 1)
    return index_tensor.permute(2,0,1).long().detach().to(util.DEVICE)

def _get_offset_index_tensor(env_height: int, env_width: int) -> torch.Tensor:
    # Create a H x W x 2 matrix
    q_col_indices = torch.linspace(0, env_width - 1, env_width)
    r_row_indices = torch.linspace(0, env_height - 1, env_height)
    q_cols, r_rows = torch.meshgrid([q_col_indices, r_row_indices])
    return torch.stack((q_cols, r_rows)).permute(1, 2, 0).long().detach().to(util.DEVICE)


def _get_batched_offset_index_tensor(batch_size: int, env_height: int, env_width: int) -> torch.Tensor:
    # Batch size could include the channel dimension.

    index_tensor = _get_offset_index_tensor(env_height, env_width)

    # Stack it
    return index_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).detach().to(util.DEVICE)


def _get_axial_index_tensor(offset_index_tensor: torch.Tensor,
                            add_u: int = 0, add_v: int = 0) -> torch.Tensor:
    # The offset index tensor is assumed to be of size B x H x W x 2, where B is the batch size (or batch size x
    # channel dimension).
    if offset_index_tensor.size(3) != 2:
        raise ValueError('Offset index tensor should have size B x H x W x 2: %s' % offset_index_tensor.size())

    # v is just the same as r.
    v = offset_index_tensor[:, :, :, 1]

    # u is the axis index. It is q - r // 2.
    u = offset_index_tensor[:, :, :, 0] - v // 2

    # Add the offsets.
    u += add_u
    v += add_v

    if (u < 0).any():
        print(u)
        raise ValueError('Axial index tensor has u negative values. Perhaps you need to add u.')
    if (v < 0).any():
        print(v)
        raise ValueError('Axial index tensor has v negative values. Perhaps you need to add v.')

    return torch.stack((u, v)).permute(1, 2, 3, 0).long()

def _get_cube_index_tensor(axial_index_tensor: torch.Tensor,
                            add_u: int = 0, add_v: int = 0) -> torch.Tensor:
    # The offset index tensor is assumed to be of size B x H x W x 2, where B is the batch size (or batch size x
    # channel dimension).
    if axial_index_tensor.size(3) != 2:
        raise ValueError('Axial index tensor should have size B x H x W x 2: %s' % axial_index_tensor.size())

    u = axial_index_tensor[:, :, :, 0]
    v = axial_index_tensor[:, :, :, 1]


    # x is just the same as v.
    x = v

    # y is -(u + v).
    y = -(u + v)

    # z is just the same as u.
    z = u

    return torch.stack((x, y, z)).permute(1, 2, 3, 0).long()

def _get_offset_axial_indices(batch_size: int, height: int, width: int,
                              additional_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offset_index_tensor = _get_batched_offset_index_tensor(batch_size, height, width)
    axial_index_tensor = _get_axial_index_tensor(offset_index_tensor, add_u=additional_size)

    qs = offset_index_tensor[:, :, :, 0].flatten()
    rs = offset_index_tensor[:, :, :, 1].flatten()
    us = axial_index_tensor[:, :, :, 0].flatten()
    vs = axial_index_tensor[:, :, :, 1].flatten()
    return qs, rs, us, vs


def _get_axial_cube_index_tensors(batch_size: int, height: int, width: int,
                                  additional_size: int) -> Tuple[torch.Tensor, torch.Tensor]:

    # axial coordinate is the pixel cooridnate on the tensor - center
    axial_index_tensor = _get_batched_offset_index_tensor(batch_size, height, width)
    axial_index_tensor -= additional_size
    cube_index_tensor = _get_cube_index_tensor(axial_index_tensor)

    return axial_index_tensor, cube_index_tensor

def _pad_axial_to_square(input_tensor: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Pads an input axial tensor to a square such that the position (0, 0) is in the center of the new square
    tensor."""
    batch_size, num_channels, axial_height, axial_width = input_tensor.size()
    placeholder = torch.zeros((batch_size, num_channels, axial_height * 2 + 1, axial_height * 2 + 1)).to(util.DEVICE)

    pose_batch_size, pose_coord_size = offset.size()
    if pose_batch_size != batch_size:
        raise ValueError('Batch size of pose and input tensor are not the same: %s vs. %s' % (pose_batch_size,
                                                                                              batch_size))
    if pose_coord_size != 2:
        raise ValueError('Pose must have 2 coordinates; has %s' % pose_coord_size)

    # Need to get the us, vs for the offset. Offset is in offset coordinates, not axial coordinates.
    # Pose has 0th index of q, and 1th index of r.
    additional_size = axial_height - axial_width
    qs = offset[:, 0]
    rs = offset[:, 1]
    us = qs - rs // 2

    u_min = axial_height - additional_size - us
    u_max = u_min + axial_height
    v_min = axial_height - rs
    v_max = v_min + axial_width
    # TDOO: maybe remove for loops
    # Not sure which is faster 1. loop of contiguous memory assignments (i.e., square reigion to square reigion mapping)
    #                          vs 2. one-time non-contiguous memory assignments  (i.e., individual indicies to indicies mapping)
    for b in range(batch_size):
        placeholder[b, :, u_min[b]:u_max[b], v_min[b]:v_max[b]] = input_tensor[b,:,:,:]

    # Create a mask for the non-padded region of axil tensor and move the mask to the coordinates of the placeholder
    # If you don't create a mask, some of pixels will be outside of the placeholder after rotation
    mask = torch.zeros((batch_size, axial_height * 2 + 1, axial_height * 2 + 1))
    additional_size = (axial_width - 1) // 2
    _, _, us, vs = _get_offset_axial_indices(batch_size, axial_width, axial_width, additional_size)
    # TDOO: maybe remove for loops
    # Not sure which is faster 1. loop of contiguous memory assignments (i.e., square reigion to square reigion mapping)
    #                          vs 2. one-time non-contiguous memory assignments  (i.e., individual indicies to indicies mapping)
    for b in range(batch_size):
        m_us, m_vs = us + u_min[b], vs + v_min[b]
        mask[b, m_us, m_vs] = 1
        mask = mask.bool()

    return placeholder, mask


def _get_cube_rotation_matrix(rots):
    assert len(rots.size()) == 1, 'Rotation is an one-dimensional tensor: %s' % rots.size()

    T = torch.zeros((rots.size(0), 3, 3))

    # Clockwise 1.047 radians (60') rotation. x is -y, y is -z and z is -x
    rot_matrix = torch.zeros((3, 3))

    # Counter-clockwise.
    rot_matrix[0,2] = -1
    rot_matrix[1,0] = -1
    rot_matrix[2,1] = -1

    # TODO: Make this loop a simple six-way look-up table.
    for i, r in enumerate(rots):
        matrix = torch.eye(3)
        num_iters = r // math.radians(60)
        num_iters = num_iters.long()
        num_iters = num_iters % 6
        for _ in range(num_iters):
            matrix = torch.matmul(matrix, rot_matrix)
        T[i,:,:] = matrix

    return T.detach().to(util.DEVICE)


def _rotate_cube_indices(cube_index_tensor: torch.Tensor, rots: torch.Tensor):
    batch_size, height, width, channels = cube_index_tensor.size()
    assert channels == 3, 'Tensor does not have 3 channels: %s' % channels

    # Calculate rotation matrices for each batch
    cube_rotation_matrix = _get_cube_rotation_matrix(rots)
    cube_index_tensor = cube_index_tensor.permute(0,3,1,2)
    cube_index_tensor = cube_index_tensor.view(batch_size, channels, height * width)
    cube_index_tensor = cube_index_tensor.float()
    cube_index_tensor_rot = torch.bmm(cube_rotation_matrix, cube_index_tensor)

    cube_index_tensor_rot = cube_index_tensor_rot.view(batch_size, channels, height, width)
    cube_index_tensor_rot = cube_index_tensor_rot.permute(0,2,3,1)
    cube_index_tensor_rot = cube_index_tensor_rot.long()
    return cube_index_tensor_rot


def offset_tensor_to_axial(input_tensor: torch.Tensor) -> torch.Tensor:
    """Transforms a tensor representing offset hex representation of an environment to axial coordinates.

    Inputs:
        input_tensor: The input tensor. Should be a square tensor with size B x C x H x W, where H = W.
    """
    # The input tensor is in offset coordinates, and should be a square matrix N x N.
    batch_size, num_channels, env_height, env_width = input_tensor.size()
    assert env_width == env_height, 'Tensor is not square: %s x %s' % (env_width, env_height)

    # Placeholder tensor
    additional_size = (env_width - 1) // 2
    axial_size = env_width + additional_size
    axial_tensor = torch.zeros((batch_size * num_channels, axial_size, env_width)).detach().to(util.DEVICE)

    qs, rs, us, vs = _get_offset_axial_indices(batch_size * num_channels, env_height, env_width, additional_size)
    indexed_input = input_tensor.view(batch_size * num_channels, env_height, env_width)[:, qs, rs]
    axial_tensor[:, us, vs] = indexed_input

    return axial_tensor.view(batch_size, num_channels, axial_size, env_width)


def axial_tensor_to_offset(axial_tensor: torch.Tensor) -> torch.Tensor:
    # Input should be consistent rotation (i.e., not square)
    # B x C x H x W
    # Should return a square tensor
    # Should return B x C x W x W
    batch_size, num_channels, tensor_height, tensor_width = axial_tensor.size()
    env_height, env_width = tensor_width, tensor_width
    assert tensor_height > tensor_width, 'Axial tensor does not have a valid shape: %s x %s' % (env_width, env_height)

    additional_size = tensor_height - tensor_width
    axial_size = env_width + additional_size
    offset_tensor = torch.zeros((batch_size * num_channels, env_width, env_width)).detach().to(util.DEVICE)

    qs, rs, us, vs = _get_offset_axial_indices(batch_size * num_channels, env_height, env_width, additional_size)

    indexed_axial = axial_tensor.view(batch_size * num_channels, tensor_height, tensor_width)[:, us, vs]
    offset_tensor[:, qs, rs] = indexed_axial

    return offset_tensor.view(batch_size, num_channels, env_height, env_width)


def translate_and_rotate(axial_tensor: torch.Tensor, target_poses: pose.Pose, is_axial_coord: bool = False) -> torch.Tensor:
    # Input should be an axial tensor (not square)
    # B x C x H x W
    # Should return a square tensor
    # Should return B x C x H' x W' where H' = W' = some function of W
    padded_tensor, mask = _pad_axial_to_square(axial_tensor, offset=target_poses.position)

    if is_axial_coord:
        center = padded_tensor.shape[-1] // 2
        slack = axial_tensor.shape[-1] //2
        offset = center-slack
        end = center+slack+1
        mask[:,:,:] = False
        mask[:,offset:end,offset:end] = True
    else:
        pass

    # get padded placeholder and get axial and cube index of eaxh pixel locations
    placeholder = torch.zeros(padded_tensor.shape).to(util.DEVICE)
    center = padded_tensor.size(2) // 2
    batch_size, _, height, width = padded_tensor.shape
    axial_index_tensor, cube_index_tensor = _get_axial_cube_index_tensors(batch_size, height,
                                                                          width,
                                                                          additional_size=center)

    # Rotate tensors clockwise by angles specfied by target_poses.orientation
    cube_index_tensor_rot = _rotate_cube_indices(cube_index_tensor, rots=target_poses.orientation) # oonly unique by rotation
    bs = _get_batch_index_tensor(batch_size, height, width)
    us, vs = axial_index_tensor[:, :, :, 0] + center, axial_index_tensor[:, :, :, 1] + center
    us_rot, vs_rot = cube_index_tensor_rot[:, :, :, 2] + center, cube_index_tensor_rot[:, :, :, 0] + center

    bs, us, vs, us_rot, vs_rot = bs[mask], us[mask], vs[mask], us_rot[mask], vs_rot[mask]
    indexed_padded = padded_tensor[bs, :, us, vs]
    placeholder[bs, :, us_rot, vs_rot] = indexed_padded

    return placeholder


def untransate_and_unrotate(input_tensor: torch.Tensor, source_poses: pose.Pose) -> torch.Tensor:
    # Input tensor should be square
    # B x C x H' x W' where H' = W' = some function of W
    # Should return an axial tensor (not square)
    pass


class Hex_Rotator():
    """
    Speed up translation and operation if local map size if fixed.
    """
    def __init__(self):
        # hyperparam
        self._center = 5

        # mask
        self._mask = torch.zeros((1, 11, 11)) 
        _, _, us, vs = _get_offset_axial_indices(1, 5, 5, 2)
        m_us, m_vs = us + 3, vs + 3
        self._mask[:, m_us, m_vs] = 1
        self._mask = self._mask.bool()
        self._mask[:,:,:] = False
        self._mask[:,3:8,3:8] = True

        # precompute get_axial_cube_index_tensors
        self._axial_index_tensor, cube_index_tensor =  _get_axial_cube_index_tensors(1, 11, 11, additional_size=5)
        cube_index_tensor = torch.cat([cube_index_tensor for _ in range(6)], 0)
        self._cube_index_tensor_rots = _rotate_cube_indices(cube_index_tensor, rots=torch.tensor([i * math.radians(60) for i in range(6)]).to(util.DEVICE)) # unique by rotatioon

        self._us, self._vs = self._axial_index_tensor[:, :, :, 0] + self._center, self._axial_index_tensor[:, :, :, 1] + self._center

    def translate_and_rotate(self, axial_tensor: torch.Tensor, target_poses: pose.Pose):
        # pad tensor
        batch_size, _ , _, _ = axial_tensor.shape
        placeholder = torch.zeros(axial_tensor.shape).to(util.DEVICE).type(axial_tensor.type()) #! still the largest bottleneck
        height, width = 11, 11

        # stack mask 
        mask = torch.cat([self._mask for _ in range(batch_size)], 0)

        # get batch tensor
        bs = _get_batch_index_tensor(batch_size, height, width)
        bs = bs[mask]

        # stack us and vs
        us = torch.cat([self._us for _ in range(batch_size)], 0)
        vs = torch.cat([self._vs for _ in range(batch_size)], 0)
        us, vs = us[mask], vs[mask]

        # get rotation
        cube_index_tensor_rot = []
        cube_index_tensor_rot = torch.stack([self._cube_index_tensor_rots[o, ...] for o in (target_poses.orientation/math.radians(60)).long()], 0) 
        us_rot, vs_rot = cube_index_tensor_rot[:, :, :, 2] + self._center, cube_index_tensor_rot[:, :, :, 0] + self._center
        us_rot, vs_rot = us_rot[mask], vs_rot[mask]

        # remove extra indicies
        keep = (us_rot >= 3) & (us_rot <= 7) & (vs_rot  >= 3) & (vs_rot <= 7)
        bs = bs[keep]
        us = us[keep] 
        vs = vs[keep] 
        us -= 3
        vs -= 3
        us_rot = us_rot[keep] 
        vs_rot = vs_rot[keep] 
        us_rot -= 3 
        vs_rot -= 3

        # detach values
        bs.detach()
        us.detach()
        vs.detach()
        us_rot.detach()
        vs_rot.detach()

        # return values
        indexed_padded = axial_tensor[bs, :, us, vs]
        placeholder[bs, :, us_rot, vs_rot] = indexed_padded
        
        return placeholder
    """
    def _pad_axial_to_square(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, axial_height, axial_width = input_tensor.size()
        placeholder = torch.zeros((batch_size, num_channels, axial_height * 2 + 1, axial_height * 2 + 1)).to(util.DEVICE)
        placeholder[:, :, 3:8, 3:8] = input_tensor
        return placeholder
    """