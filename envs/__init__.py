import warnings
from envs.dflex_env import DFlexEnv
from envs.ant import AntEnv
from envs.cheetah import CheetahEnv
from envs.hopper import HopperEnv
from envs.cartpole import CartPoleEnv
from envs.humanoid import HumanoidEnv
try:
    from envs.dflex_diff_render_env import DFlexDiffRenderEnv
    from envs.cartpole_diff_render_env import CartPoleDiffRenderEnv
    from envs.ant_diff_render_env import AntDiffRenderEnv
    from envs.hopper_diff_render_env import HopperDiffRenderEnv
    from envs.cheetah_diff_render_env import CheetahDiffRenderEnv
    from envs.humanoid_diff_render_env import HumanoidDiffRenderEnv
except ImportError:
    warnings.warn("Failed to load Diff Render Env. Continuing without it.", category=ImportWarning)