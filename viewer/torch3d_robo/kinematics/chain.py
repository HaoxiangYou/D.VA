from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import torch

import kinematics.transforms as tf
from kinematics.frame import Frame, Link, Joint
from kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_44, axis_and_d_to_pris_matrix


def get_n_joints(th):
    """

    Args:
        th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched

    Returns: The number of joints in the input

    """
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[-1]
    elif isinstance(th, list) or isinstance(th, dict):
        return len(th)
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def get_batch_size(th):
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[0]
    elif isinstance(th, dict):
        elem_shape = get_dict_elem_shape(th)
        return elem_shape[0]
    elif isinstance(th, list):
        # Lists cannot be batched. We don't allow lists of lists.
        return 1
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    return th, N


def get_dict_elem_shape(th_dict):
    elem = th_dict[list(th_dict.keys())[0]]
    if isinstance(elem, np.ndarray):
        return elem.shape
    elif isinstance(elem, torch.Tensor):
        return elem.shape
    else:
        return ()


class Chain:
    """
    Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint.
    """

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        self.identity = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)

        low, high = self.get_joint_limits()
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

        # As we traverse the kinematic tree, each frame is assigned an index.
        # We use this index to build a flat representation of the tree.
        # parents_indices and body_joints_indices all use this indexing scheme.
        # The root frame will be index 0 and the first frame of the root frame's children will be index 1,
        # then the child of that frame will be index 2, etc. In other words, it's a depth-first ordering.
        self.parents_indices = []  # list of indices from 0 (root) to the given frame
        self.body_joint_indices = []
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([self.n_joints, 3], dtype=self.dtype, device=self.device)
        self.link_offsets = []
        self.body_joint_offsets = []
        self.body_joint_type_indices = []
        queue = []
        queue.insert(-1, (self._root, -1, 0))  # the root has no parent so we use -1.
        idx = 0
        self.frame_to_idx = {}
        self.idx_to_frame = {}
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            name_strip = root.name.strip("\n")
            self.frame_to_idx[name_strip] = idx
            self.idx_to_frame[idx] = name_strip
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])

            if root.link.offset is None:
                self.link_offsets.append(None)
            else:
                self.link_offsets.append(root.link.offset.get_matrix())

            joint_indices = []
            joint_type_indices = []
            joint_offsets = []
            for joint in root.joints:
                is_fixed = joint.joint_type == 'fixed'
                is_free = joint.joint_type == "free"

                if joint.offset is None:
                    joint_offsets.append(None)
                else:
                    joint_offsets.append(joint.offset.get_matrix())

                if is_fixed or is_free:
                    joint_indices.append(-1)
                else:
                    jnt_idx = self.get_joint_parameter_names().index(joint.name)
                    self.axes[jnt_idx] = joint.axis
                    joint_indices.append(jnt_idx)
                joint_type_indices.append(Joint.TYPES.index(joint.joint_type))
            
            self.body_joint_indices.append(torch.tensor(joint_indices, device=self.device))
            self.body_joint_type_indices.append(torch.tensor(joint_type_indices, device=self.device))
            self.body_joint_offsets.append(joint_offsets)

            for child in root.children:
                queue.append((child, idx, depth + 1))

            idx += 1
        # We need to use a dict because torch.compile doesn't list lists of tensors
        self.parents_indices = [torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices]

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        self.parents_indices = [p.to(dtype=torch.long, device=self.device) for p in self.parents_indices]
        self.body_joint_type_indices = [typ.to(dtype=torch.long, device=self.device) for typ in self.body_joint_type_indices]
        self.body_joint_indices = [idx.to(dtype=torch.long, device=self.device) for idx in self.body_joint_indices]
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.body_joint_offsets = [[j.to(dtype=self.dtype, device=self.device) for j in joints] for joints in self.body_joint_offsets]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)

        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame: Frame) -> Optional[Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name) -> Optional[Frame]:
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame) -> Optional[Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    @staticmethod
    def _get_joints(frame, exclude_fixed_and_free=True):
        joints = []
        for joint in frame.joints:
            if not (exclude_fixed_and_free and (joint.joint_type == "fixed" or joint.joint_type == "free")):
                joints.append(joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child, exclude_fixed_and_free=exclude_fixed_and_free))
        return joints

    def get_joints(self, exclude_fixed_and_free=True):
        joints = self._get_joints(self._root, exclude_fixed_and_free=exclude_fixed_and_free)
        return joints

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed_and_free=True):
        names = []
        for j in self.get_joints(exclude_fixed_and_free=exclude_fixed_and_free):
            if exclude_fixed_and_free and (j.joint_type == "fixed" or j.joint_type == "free"):
                continue
            names.append(j.name)
        return names

    @staticmethod
    def _find_joint_recursive(name, frame):
        for child in frame.children:
            for joint in child.joints:
                if joint.name == name:
                    return joint
            ret = Chain._find_joint_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name) -> Optional[Link]:
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    def find_joint(self, name):
        for joint in self._root.joints:
            if joint.name == name:
                return self._root.joint
        return self._find_joint_recursive(name, self._root)

    @staticmethod
    def _get_joint_parent_frame_names(frame, exclude_fixed_and_free=True):
        joint_names = []
        if not (exclude_fixed_and_free and (frame.joint.joint_type == "fixed" or frame.joint.joint_type == "free")):
            joint_names.append(frame.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parent_frame_names(child))
        return joint_names

    def get_joint_parent_frame_names(self, exclude_fixed_and_free=True):
        names = self._get_joint_parent_frame_names(self._root, exclude_fixed_and_free=exclude_fixed_and_free)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed_and_free=True) -> Sequence[str]:
        names = []
        if not exclude_fixed_and_free:
            names.append(frame.name)
        else:
            is_free_or_fixed = False
            for joint in frame.joints:
                if joint.joint_type == "fixed" or joint.joint_type == "free":
                    is_free_or_fixed = True
            if not is_free_or_fixed:
                names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child))
        return names

    def get_frame_names(self, exclude_fixed_and_free=True):
        names = self._get_frame_names(self._root, exclude_fixed_and_free=exclude_fixed_and_free)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_links(frame):
        links = [frame.link]
        for child in frame.children:
            links.extend(Chain._get_links(child))
        return links

    def get_links(self):
        links = self._get_links(self._root)
        return links

    @staticmethod
    def _get_link_names(frame):
        link_names = [frame.link.name]
        for child in frame.children:
            link_names.extend(Chain._get_link_names(child))
        return link_names

    def get_link_names(self):
        names = self._get_link_names(self._root)
        return sorted(set(names), key=names.index)

    @lru_cache
    def get_frame_indices(self, *frame_names):
        return torch.tensor([self.frame_to_idx[n] for n in frame_names], dtype=torch.long, device=self.device)

    def print_tree(self, do_print=True):
        tree = str(self._root)
        if do_print:
            print(tree)
        return tree

    def forward_kinematics(self, th, root_pos=None, root_quat=None):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            root_pos: the position of root joint
            root_quat: the orientation of root joit
        Returns:
            A dict of Transform3d objects for each frame.

        """
        frame_indices = self.get_all_frame_indices()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)
        b = th.shape[0]
        
        if root_pos is None:
            root_pos = torch.tensor([0, 0, 0]).to(th).unsqueeze(0).repeat(b, 1)
        if root_quat is None:
            root_quat = torch.tensor([1, 0, 0, 0]).to(th).unsqueeze(0).repeat(b, 1)

        root_pos = self.ensure_tensor(root_pos)
        root_pos = torch.atleast_2d(root_pos)
        root_quat = self.ensure_tensor(root_quat)
        root_quat = torch.atleast_2d(root_quat)
        assert root_pos.shape[0] == b
        assert root_quat.shape[0] == b

        axes_expanded = self.axes.unsqueeze(0).repeat(b, 1, 1)

        # compute all joint transforms at once first
        # in order to handle multiple joint types without branching, we create all possible transforms
        # for all joint types and then select the appropriate one for each joint.
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)

        frame_transforms = {}
        b = th.shape[0]
        for frame_idx in frame_indices:
            frame_transform = tf.Transform3d(pos=root_pos, rot=root_quat, device=th.device).get_matrix()

            # iterate down the list and compose the transform
            for chain_idx in self.parents_indices[frame_idx.item()]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    link_offset_i = self.link_offsets[chain_idx]
                    if link_offset_i is not None:
                        frame_transform = frame_transform @ link_offset_i
                    n_joint = len(self.body_joint_indices[chain_idx])
                    for i in range(n_joint):
                        joint_offset_i = self.body_joint_offsets[chain_idx][i]
                        jnt_idx = self.body_joint_indices[chain_idx][i]
                        jnt_type = self.body_joint_type_indices[chain_idx][i]
                        if jnt_type == 0:
                            pass
                        elif jnt_type == 1:
                            jnt_transform_i = rev_jnt_transform[:, jnt_idx]
                            if joint_offset_i is not None:
                                jnt_transform_i = joint_offset_i @ jnt_transform_i @ joint_offset_i.inverse()

                            frame_transform = frame_transform @ jnt_transform_i
                        elif jnt_type == 2:
                            jnt_transform_i = pris_jnt_transform[:, jnt_idx]
                            frame_transform = frame_transform @ jnt_transform_i

            frame_transforms[frame_idx.item()] = frame_transform

        frame_names_and_transform3ds = {self.idx_to_frame[frame_idx]: tf.Transform3d(matrix=transform) for
                                        frame_idx, transform in frame_transforms.items()}

        return frame_names_and_transform3ds

    def ensure_tensor(self, th):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names(). th must contain all joints in the entire chain.
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, dict):
            # convert dict to a flat, complete, tensor of all joints values. Missing joints are filled with zeros.
            th_dict = th
            elem_shape = get_dict_elem_shape(th_dict)
            th = torch.ones([*elem_shape, self.n_joints], device=self.device, dtype=self.dtype) * torch.nan
            joint_names = self.get_joint_parameter_names()
            for joint_name, joint_position in th_dict.items():
                jnt_idx = joint_names.index(joint_name)
                th[..., jnt_idx] = joint_position
            if torch.any(torch.isnan(th)):
                msg = "Missing values for the following joints:\n"
                for joint_name, th_i in zip(self.get_joint_parameter_names(), th):
                    msg += joint_name + "\n"
                raise ValueError(msg)
        return th

    def get_all_frame_indices(self):
        frame_indices = self.get_frame_indices(*self.get_frame_names(exclude_fixed_and_free=False))
        return frame_indices

    def clamp(self, th):
        """

        Args:
            th: Joint configuration

        Returns: Always a tensor in the order of self.get_joint_parameter_names(), possibly batched.

        """
        th = self.ensure_tensor(th)
        return torch.clamp(th, self.low, self.high)

    def get_joint_limits(self):
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self):
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self):
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(self, param_name):
        low = []
        high = []
        for joint in self.get_joints():
            val = getattr(joint, param_name)
            if val is None:
                # NOTE: This changes the previous default behavior of returning
                # +/- np.pi for joint limits to be more natural for both
                # revolute and prismatic joints
                low.append(-np.inf)
                high.append(np.inf)
            else:
                low.append(val[0])
                high.append(val[1])
        return low, high

    @staticmethod
    def _get_joints_and_child_links(frame):
        joint = frame.joint

        me_and_my_children = [frame.link]
        for child in frame.children:
            recursive_child_links = yield from Chain._get_joints_and_child_links(child)
            me_and_my_children.extend(recursive_child_links)

        if joint is not None and joint.joint_type != 'fixed':
            yield joint, me_and_my_children

        return me_and_my_children

    def get_joints_and_child_links(self):
        yield from Chain._get_joints_and_child_links(self._root)
