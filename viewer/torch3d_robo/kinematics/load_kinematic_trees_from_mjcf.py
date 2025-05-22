import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)

import mujoco
from mujoco._structs import _MjModelBodyViews as MjModelBodyViews
from kinematics import chain
from kinematics import frame
import kinematics.transforms as tf

# Converts from MuJoCo joint types to pytorch_kinematics joint types
JOINT_TYPE_MAP = {
    mujoco.mjtJoint.mjJNT_FREE: 'free',
    mujoco.mjtJoint.mjJNT_HINGE: 'revolute',
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic"
}


def body_to_geoms(m: mujoco.MjModel, body: MjModelBodyViews):
    # Find all geoms which have body as parent
    geoms = []
    for geom_id in range(m.ngeom):
        geom = m.geom(geom_id)
        if geom.bodyid == body.id:
            geoms.append(frame.Geom(name=geom.name, offset=tf.Transform3d(rot=geom.quat, pos=geom.pos), type_id=geom.type[0],
                                    size=geom.size, rgba=geom.rgba))
    return geoms


def parse_joints(m, body):
    joints = []
    n_joints = body.jntnum[0]
    if n_joints >= 1:
        joint_address_start = body.jntadr[0]
        for i in range(n_joints):
            joint = m.joint(joint_address_start + i)
            joint_offset = tf.Transform3d(pos=joint.pos)
            joint = frame.Joint(joint.name, offset=joint_offset, axis=joint.axis,
                                        joint_type=JOINT_TYPE_MAP[joint.type[0]],
                                        limits=(joint.range[0], joint.range[1]))
            joints.append(joint)
    else:
        joints.append(frame.Joint(body.name + "_fixed_joint"))
    return joints

def _build_chain_recurse(m, parent_frame, parent_body):
    # iterate through all bodies that are children of parent_body
    for body_id in range(m.nbody):
        body = m.body(body_id)
        if body.parentid == parent_body.id and body_id != parent_body.id:
            child_link = frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))
            child_joints = parse_joints(m, body)
            child_geoms = body_to_geoms(m, body)
            child_frame = frame.Frame(name=body.name, link=child_link, joints=child_joints, geoms=child_geoms)
            parent_frame.children = parent_frame.children + [child_frame, ]
            _build_chain_recurse(m, child_frame, body)


def build_chain_from_mjcf(path):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    path : str
        Path to mujoco xml

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    m = mujoco.MjModel.from_xml_path(path)
    # assume there is only one robot in the scene
    root_body = m.body(1)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joints=parse_joints(m, root_body),
                             geoms=body_to_geoms(m, root_body)
                            )
    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)
