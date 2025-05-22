import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch

import kinematics.transforms as tf
from kinematics.transforms import axis_and_angle_to_matrix_33


class Geom(object):
    TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

    def __init__(self, name=None, offset=None, type_id=None, size=None, rgba=None):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.type_id = type_id
        self.size = size
        self.rgba = rgba
        self.name = name 

    def to(self, *args, **kwargs):
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def __repr__(self):
        return "Geom(name={0}, offset={1}, type_id='{2}', size={3}, rgba={4})".format(
            self.name, self.offset, self.type_id, self.size, self.rgba)


class Link(object):
    def __init__(self, name=None, offset=None):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.name = name

    def to(self, *args, **kwargs):
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def __repr__(self):
        return "Link(name='{0}', offset={1}".format(self.name, self.offset,)


class Joint(object):
    TYPES = ['fixed', 'revolute', 'prismatic', 'free']

    def __init__(self, name=None, offset=None, joint_type='fixed', axis=(0.0, 0.0, 1.0),
                 dtype=torch.float32, device="cpu", limits=None,
                 velocity_limits=None, effort_limits=None):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.name = name
        if joint_type not in self.TYPES:
            raise RuntimeError("joint specified as {} type not, but we only support {}".format(joint_type, self.TYPES))
        self.joint_type = joint_type
        if axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        else:
            if torch.is_tensor(axis):
                self.axis = axis.clone().detach().to(dtype=dtype, device=device)
            else:
                self.axis = torch.tensor(axis, dtype=dtype, device=device)
        # normalize axis to have norm 1 (needed for correct representation scaling with theta)
        self.axis = self.axis / self.axis.norm()

        self.limits = limits
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits

    def to(self, *args, **kwargs):
        self.axis = self.axis.to(*args, **kwargs)
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def clamp(self, joint_position):
        if self.limits is None:
            return joint_position
        else:
            return torch.clamp(joint_position, self.limits[0], self.limits[1])

    def __repr__(self):
        return "Joint(name='{0}', offset={1}, joint_type='{2}', axis={3})".format(self.name,
                                                                                  self.offset,
                                                                                  self.joint_type,
                                                                                  self.axis)


# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '

class Frame(object):
    def __init__(self, name=None, link=None, joints=[], geoms=[], children=None):
        self.name = 'None' if name is None else name
        self.link = link if link is not None else Link()
        self.joints = joints if len(joints) > 0 else [Joint()]
        self.geoms = geoms
        if children is None:
            self.children = []

    def __str__(self, prefix='', root=True):
        pointers = [tee] * (len(self.children) - 1) + [last]
        if root:
            ret = prefix + self.name + "\n"
        else:
            ret = ""
        for pointer, child in zip(pointers, self.children):
            ret += prefix + pointer + child.name + "\n"
            if child.children:
                extension = branch if pointer == tee else space
                # i.e. space because last, └── , above so no more |
                ret += child.__str__(prefix=prefix + extension, root=False)
        return ret

    def to(self, *args, **kwargs):
        self.joints = [j.to(*args, **kwargs) for j in self.joints]
        self.link = self.link.to(*args, **kwargs)
        self.geoms = [g.to(*args, **kwargs) for g in self.geoms]
        self.children = [c.to(*args, **kwargs) for c in self.children]
        return self

    def add_child(self, child):
        self.children.append(child)

    def is_end(self):
        return (len(self.children) == 0)

    def get_transform(self, thetas):
        dtype = self.joints[0].axis.dtype
        d = self.joints[0].axis.device
        assert thetas.shape[-1] == len(self.joints)
        t = tf.Transform3d(default_batch_size=theta.shape[0], dtype=dtype, device=d)
        for i in range(thetas.shape[-1]):
            theta = thetas[...,i]
            joint = self.joints[i]
            if joint.offset is not None:
                t = t.compose(joint.offset)
            if joint.joint_type == 'revolute':
                rot = axis_and_angle_to_matrix_33(joint.axis, theta)
                t = t.compose(tf.Transform3d(rot=rot, dtype=dtype, device=d))
            elif joint.joint_type == 'prismatic':
                pos = theta.unsqueeze(1) * joint.axis
                t = t.compose(tf.Transform3d(pos=pos, dtype=dtype, device=d))
            elif joint.joint_type == 'fixed':
                pass
            else:
                raise ValueError("Unsupported joint type %s." % joint.joint_type)

        return t