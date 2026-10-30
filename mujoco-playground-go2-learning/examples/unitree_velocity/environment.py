"""Go2/Go1 speed-command tasks built on Playground's Go1 joystick task.

The external Go2 model is read, never edited.  Its MJX XML is adapted in memory
to provide the sensor names used by the existing observation/reward code.
This is a learning example, not a calibrated controller for a physical robot.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants, joystick

COMMANDS = ("stop", "forward", "backward", "left", "right")


def command_vector(name: str, speed: float = 0.5, yaw_rate: float = 0.6):
    """Return body-frame [vx (m/s), vy (m/s), yaw rate (rad/s)]."""
    vectors = {
        "stop": (0.0, 0.0, 0.0),
        "forward": (speed, 0.0, 0.0),
        "backward": (-speed, 0.0, 0.0),
        "left": (0.0, 0.0, yaw_rate),
        "right": (0.0, 0.0, -yaw_rate),
    }
    if name not in vectors:
        raise ValueError(f"Unknown command {name!r}; choose from {COMMANDS}")
    return jp.asarray(vectors[name], dtype=jp.float32)


def default_config():
    """Keep the established Go1 reward/observation design; narrow the commands."""
    config = joystick.default_config()
    config.impl = "jax"
    config.command_config.speed = 0.5
    config.command_config.yaw_rate = 0.6
    config.command_config.min_scale = 0.5
    return config


class _CommandTask:
    """Shared command curriculum and external command control for Go1/Go2."""

    def sample_command(self, rng, x_k):
        del x_k
        choice_key, scale_key = jax.random.split(rng)
        table = jp.stack([
            command_vector(name, self._config.command_config.speed,
                           self._config.command_config.yaw_rate)
            for name in COMMANDS
        ])
        index = jax.random.randint(choice_key, (), 0, len(COMMANDS))
        scale = jax.random.uniform(
            scale_key, (), minval=self._config.command_config.min_scale,
            maxval=1.0,
        )
        return table[index] * scale

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["rng"], command_key = jax.random.split(info["rng"])
        info["command"] = self.sample_command(command_key, info["command"])
        # Always present: reset/step must return the same JAX pytree structure.
        info["manual_command"] = jp.asarray(False)
        return state.replace(info=info, obs=self._get_obs(state.data, info))

    def step(self, state, action):
        info = dict(state.info)
        frozen = info["manual_command"]
        original_command = info["command"]
        info["steps_until_next_cmd"] = jp.where(
            frozen, jp.int32(1_000_000_000), info["steps_until_next_cmd"]
        )
        state = super().step(state.replace(info=info), action)
        info = dict(state.info)
        info["command"] = jp.where(frozen, original_command, info["command"])
        info["steps_until_next_cmd"] = jp.where(
            frozen, jp.int32(1_000_000_000), info["steps_until_next_cmd"]
        )
        # The upstream task updates command after computing obs.  Rebuild obs
        # here so the next policy action sees the command actually in info.
        return state.replace(info=info, obs=self._get_obs(state.data, info))

    def set_command(self, state, command):
        """Set and freeze a command until reset or another set_command call.

        This also refreshes observations.  Updating info alone leaves the actor
        looking at an old command for its next action.
        """
        command = jp.asarray(command, dtype=jp.float32)
        if command.shape != (3,):
            raise ValueError("command must have shape (3,): [vx, vy, yaw_rate]")
        info = dict(state.info)
        info["command"] = command
        info["manual_command"] = jp.asarray(True)
        info["steps_until_next_cmd"] = jp.int32(1_000_000_000)
        return state.replace(info=info, obs=self._get_obs(state.data, info))


class Go1Velocity(_CommandTask, joystick.Joystick):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(
            task="flat_terrain", config=config or default_config(),
            config_overrides=config_overrides,
        )


def _adapt_go2_xml(model_dir: Path):
    """Merge scene/model XML, simplify collisions, and add task sensors."""
    robot_path = model_dir / "go2_mjx.xml"
    scene_path = model_dir / "scene_mjx.xml"
    for path in (robot_path, scene_path, model_dir / "assets"):
        if not path.exists():
            raise FileNotFoundError(f"Required Go2 model asset not found: {path}")
    robot = ET.parse(robot_path).getroot()
    scene = ET.parse(scene_path).getroot()
    compiler = robot.find("compiler")
    compiler.set("meshdir", "assets")
    for include in list(scene.findall("include")):
        if include.get("file") != "go2_mjx.xml":
            raise ValueError(f"Unsupported scene include: {include.get('file')}")
        scene.remove(include)
    for child in list(robot):
        scene.append(child)

    torso = scene.find(".//body[@name='base']")
    if torso is None:
        raise ValueError("Go2 model must have a body named 'base'")
    torso.set("name", "trunk")
    for foot in go1_constants.FEET_SITES:
        site = scene.find(f".//site[@name='{foot}_foot']")
        if site is None:
            raise ValueError(f"Go2 model has no {foot}_foot site")
        site.set("name", foot)

    # Match the inexpensive Go1 feet-only training model: only feet touch the
    # floor; no body/leg/self collisions.  This intentional simplification is
    # unsuitable for obstacle or fall-contact studies without revisiting it.
    for geom in scene.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    floor = scene.find(".//geom[@name='floor']")
    floor.set("contype", "1")
    floor.set("conaffinity", "0")
    floor.set("condim", "3")
    floor.set("friction", "0.6")
    floor.set("priority", "1")
    for foot in go1_constants.FEET_GEOMS:
        geom = scene.find(f".//geom[@name='{foot}']")
        geom.set("conaffinity", "1")
        geom.set("condim", "3")
    foot_default = scene.find(".//default[@class='foot']/geom")
    foot_default.set("condim", "3")

    sensors = ET.SubElement(scene, "sensor")
    ET.SubElement(sensors, "framezaxis", name="upvector", objtype="site", objname="imu")
    ET.SubElement(sensors, "velocimeter", name="local_linvel", site="imu")
    for foot in go1_constants.FEET_SITES:
        ET.SubElement(sensors, "framelinvel", name=f"{foot}_global_linvel",
                      objtype="site", objname=foot)
        ET.SubElement(sensors, "framepos", name=f"{foot}_pos", objtype="site",
                      objname=foot, reftype="site", refname="imu")
        ET.SubElement(sensors, "contact", name=f"{foot}_floor_found",
                      geom1=foot, geom2="floor", reduce="mindist", num="1", data="found")

    assets = {}
    for path in (model_dir / "assets").rglob("*"):
        if path.is_file():
            # MuJoCo's VFS strips directories; each basename must be unique.
            if path.name in assets:
                raise ValueError(f"Duplicate asset basename: {path.name}")
            assets[path.name] = path.read_bytes()
    return ET.tostring(scene, encoding="unicode"), assets


class Go2Velocity(_CommandTask, joystick.Joystick):
    def __init__(self, model_dir=None, config=None, config_overrides=None):
        config = config or default_config()
        # Skip Go1Env's hard-coded asset loader without altering global names.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        if self._config.impl != "jax":
            raise ValueError("This learning example currently supports impl='jax'")
        self.model_dir = Path(model_dir or (mjx_env.MENAGERIE_PATH / "unitree_go2")).expanduser().resolve()
        self._adapted_xml, self._model_assets = _adapt_go2_xml(self.model_dir)
        self._mj_model = mujoco.MjModel.from_xml_string(self._adapted_xml, self._model_assets)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 20
        # Position targets + joint viscous damping, as in Playground Go1.
        # The source Go2 general actuator includes -0.5*qvel; clear that term
        # here so Kd is not counted both in actuator bias and joint damping.
        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
        self._mj_model.actuator_biasprm[:, 2] = 0.0
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx_env.put_model(self._mj_model, impl="jax")
        self._xml_path = str(self.model_dir / "scene_mjx.xml")
        self._imu_site_id = self._mj_model.site("imu").id
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{foot}_floor_found").id
            for foot in go1_constants.FEET_GEOMS
        ]
        joystick.Joystick._post_init(self)


def make_env(robot="go2", model_dir=None, config_overrides=None):
    """Create an independent task without changing Playground's registry.

    model_dir is the directory containing go2_mjx.xml and assets.  Dotted
    config_overrides can adjust e.g. command_config.speed or noise_config.level.
    """
    if robot == "go2":
        return Go2Velocity(model_dir=model_dir, config_overrides=config_overrides)
    if robot == "go1":
        if model_dir is not None:
            raise ValueError("model_dir is only used by the Go2 model adapter")
        return Go1Velocity(config_overrides=config_overrides)
    raise ValueError("robot must be 'go2' or 'go1'")
