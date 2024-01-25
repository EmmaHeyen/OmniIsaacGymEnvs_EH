# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_nut_bolt.yaml.
"""

# """
# For Linux: "alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh"
# For Windows: "doskey PYTHON_PATH=C:\Users\emmah\AppData\Local\ov\pkg\isaac_sim-2023.1.1\python.bat $*"
# """



import hydra
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, PhysxSchema

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.physx.scripts import physicsUtils, utils

from omniisaacgymenvs.robots.articulations.views.factory_franka_view import (
    FactoryFrankaView,
)
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)

ops="windows" # (choose if "windows" or "linux")

class FactoryEnvPegHole_eh(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()


### CHECK HERE
    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/FactoryEnvPegHole_eh.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_peg_hole_eh.yaml"
        self.asset_info_peg_hole=hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting


    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]

        self._get_env_yaml_params()

    def set_up_scene(self, scene) -> None:
        """Import assets. Add to scene."""

        # Increase buffer size to prevent overflow for Place and Screw tasks
        physxSceneAPI = self._env._world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)

        self.import_franka_assets(add_to_stage=True) # defined in factory_base --> probably fine as is
        self.create_peg_hole_material()
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        self._import_env_assets(add_to_stage=True)

        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.pegs = RigidPrimView(
            prim_paths_expr="/World/envs/.*/peg/peg.*",
            name="pegs_view",
            track_contact_forces=True,
        )
        self.holes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/hole/hole.*",
            name="holes_view",
            track_contact_forces=True,
        )

        scene.add(self.pegs)
        scene.add(self.holes)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

        return

    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""

        super().initialize_views(scene)

        self.import_franka_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)

        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        if scene.object_exists("pegs_view"):
            scene.remove_object("pegs_view", registry_only=True)
        if scene.object_exists("holes_view"):
            scene.remove_object("holes_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("fingertips_view"):
            scene.remove_object("fingertips_view", registry_only=True)

        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.pegs = RigidPrimView(
            prim_paths_expr="/World/envs/.*/peg/peg.*", name="pegs_view"
        )
        self.holes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/hole/hole.*", name="holes_view"
        )

        scene.add(self.pegs)
        scene.add(self.holes)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

    def create_peg_hole_material(self):
        """Define peg and hole material."""

        self.PegHolePhysicsMaterialPath = "/World/Physics_Materials/PegHoleMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.PegHolePhysicsMaterialPath,
            density=self.cfg_env.env.peg_hole_density,                  # from config file (FactoryEnvPegHole_eh.yaml)
            staticFriction=self.cfg_env.env.peg_hole_friction,          # from config file (FactoryEnvPegHole_eh.yaml)
            dynamicFriction=self.cfg_env.env.peg_hole_friction,         # from config file (FactoryEnvPegHole_eh.yaml)
            restitution=0.0,
        )

    def _import_env_assets(self, add_to_stage=True):
        """Set peg and hole asset options. Import assets."""

        assets_root_path = get_assets_root_path()
        if ops=="linux":
            usd_path="usd_path_linux"
        if ops=="windows":
            usd_path="usd_path_windows"   

        
        self.peg_heights = []
        self.peg_widths = []
        self.hole_widths = []
        self.hole_heights = []
        self.hole_drill_hole_heights=[]
        # self.bolt_shank_lengths = []
        # self.thread_pitches = []

        # assets_root_path = get_assets_root_path()

        for i in range(0, self._num_envs):                                              # für jede einzelne env
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))       # desired subassemblies aus config datei von env
            subassembly = self.cfg_env.env.desired_subassemblies[j]                     # z.B. desired_subassembly = ['peg_hole_1', 'peg_hole_1']. Aktuell nur diese desired_subassembly möglich. 
            components = list(self.asset_info_peg_hole[subassembly])                    # werden hier zufällig verschiedene Varianten für jede einzelne env ausgewählt? (in diesem Fall alle gleich da beide Elemente in desired_subassemblies identisch sind?) 
            # print("components_list: ", components)

            peg_translation = torch.tensor(                                             # passt das so??
                [
                    0.0,
                    self.cfg_env.env.peg_lateral_offset,
                    self.cfg_base.env.table_height,                                     # from config file FactoryBase.yaml
                ],
                device=self._device,
            )
            peg_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)



            # ANNAHME: Wir nehmen nur eine Variante von subassemblies, daher brauchen wir die heights etc nicht? 
            # Und: heights etc. bereits durch cad-datei vorgegeben?

            peg_height = self.asset_info_peg_hole[subassembly][components[0]]["height"] # aus factory_asset_info_nut_bolt datei
            peg_width = self.asset_info_peg_hole[subassembly][components[0]][       # aus factory_asset_info_nut_bolt datei
                "width"
            ]
            self.peg_heights.append(peg_height)                                         # nut_height=height of nut. different for different subassemblies --> is probably measurement of nut
            self.peg_widths.append(peg_width)

            peg_file = self.asset_info_peg_hole[subassembly][components[0]][usd_path] # aus factory_asset_info_nut_bolt datei; müsste auch über asset path funktionieren

            if add_to_stage:                                                            # immer TRUE?? (s.oben in def _import_env_assets..)
                add_reference_to_stage(peg_file, f"/World/envs/env_{i}" + "/peg")
                # peg_prim=add_reference_to_stage(peg_file, f"/World/envs/env_{i}" + "/peg")
                
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/peg",
                    translation=peg_translation,
                    orientation=peg_orientation,
                )
                # UsdPhysics.CollisionAPI.Apply(peg_prim)

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/peg/{components[0]}/collisions" # does not work so far, because i have no usd file with that node --> update 19.01.: works now but not for hole yet
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/peg/{components[0]}/collisions/mesh_01"
                    ),
                    self.PegHolePhysicsMaterialPath,
                )

                ##### TODO: Check out task config file -->does task config file only have to have the same name as the task?
                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "peg",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/peg"),
                    self._sim_config.parse_actor_config("peg"),
                )

            hole_translation = torch.tensor(
                [0.0, 0.0, self.cfg_base.env.table_height], device=self._device                             # from config file FactoryBase.yaml
            )
            hole_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            hole_width = self.asset_info_peg_hole[subassembly][components[1]]["width"]                      # quadratische Grundfläche 
            hole_height = self.asset_info_peg_hole[subassembly][components[1]]["height"]
            hole_drill_hole_height = self.asset_info_peg_hole[subassembly][components[1]]["drill_hole_height"]    # Länge Bohrloch  


            # ANNAHME: Wir nehmen nur eine Variante von subassemblies, daher brauchen wir die heights-listen etc nicht? 
            # Und: heights etc. bereits durch cad-datei vorgegeben? (Für hole statt bolt müsste man auch andere variablen nehmen)
            # vorgehen (240116): erstal mitnehmen, maße aus cad-file übernommen. (siehe factory_asset_info_peg_hole_eh.yaml)
            self.hole_widths.append(hole_width)
            self.hole_heights.append(hole_height)
            self.hole_drill_hole_heights.append(hole_drill_hole_height)

            # self.bolt_head_heights.append(bolt_head_height)
            # self.bolt_shank_lengths.append(bolt_shank_length) 
            hole_file = self.asset_info_peg_hole[subassembly][components[1]][usd_path]

            if add_to_stage:
                add_reference_to_stage(hole_file, f"/World/envs/env_{i}" + "/hole")
                # hole_prim=add_reference_to_stage(hole_file, f"/World/envs/env_{i}" + "/hole")

                # UsdPhysics.CollisionAPI.Apply(hole_prim)

                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/hole",
                    translation=hole_translation,
                    orientation=hole_orientation,
                )

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/hole/{components[1]}/collisions"
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/hole/{components[1]}/collisions/mesh_01"
                    ),
                    self.PegHolePhysicsMaterialPath,
                )

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "hole",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/hole"),
                    self._sim_config.parse_actor_config("hole"),
                )

            # thread_pitch = self.asset_info_peg_hole[subassembly]["thread_pitch"]            # Gewindegang
            # self.thread_pitches.append(thread_pitch)


        ####TODO: How do i transform this to my problem???
                
        # For computing body COM pos
        self.peg_heights = torch.tensor(
            self.peg_heights, device=self._device
        ).unsqueeze(-1) # adds a new dimension at the last axis (-1) of the tensor
        self.hole_heights = torch.tensor(
            self.hole_heights, device=self._device
        ).unsqueeze(-1)

        self.hole_drill_hole_heights = torch.tensor(
            self.hole_drill_hole_heights, device=self._device
        ).unsqueeze(-1)




        # For setting initial state - 
        self.peg_widths = torch.tensor( # --> used when resetting gripper  
            self.peg_widths, device=self._device
        ).unsqueeze(-1)
        # self.bolt_shank_lengths = torch.tensor( # --> used when resetting object in screw task
        #     self.bolt_shank_lengths, device=self._device
        # ).unsqueeze(-1)

        # # For defining success or failure (only for screw task)
        # self.bolt_widths = torch.tensor(
        #     self.bolt_widths, device=self._device
        # ).unsqueeze(-1)
        # self.thread_pitches = torch.tensor(
        #     self.thread_pitches, device=self._device
        # ).unsqueeze(-1)

    def refresh_env_tensors(self):
        """Refresh tensors."""

        # Peg tensors
        self.peg_pos, self.peg_quat = self.pegs.get_world_poses(clone=False)            # positions in world frame of the prims with shape (M,3), quaternion orientations in world frame of the prims (scalar-first (w,x,y,z), shape (M,4))
        ''' get world_poses(indices, clone)
        indices: size (M,) Where M <= size of the encapsulated prims in the view. Defaults to None (i.e: all prims in the view).
        here: M=num_envs?
        clone (bool, optional) – True to return a clone of the internal buffer. Otherwise False. Defaults to True.
        '''
        self.peg_pos -= self.env_pos                                                    # A-=B is equal to A=A-B; peg'position relative to env position. env position is absolute to world

        ### TODO: WHAT is a com pos --> Center of mass position --> only needed in screw task?
        self.peg_com_pos = fc.translate_along_local_z(                                  # translate_along_local_z(pos, quat, offset, device). Translate global body position along local Z-axis and express in global coordinates
            pos=self.peg_pos,           # "measured" via get_world_poses
            quat=self.peg_quat,         # "measured" via get_world_poses


            ### TODO: what to put here (offset) --> Schwerpunkt wenn fertig assembled??
            offset=self.hole_drill_hole_heights + self.peg_heights * 0.5,
            device=self.device,
        )
        self.peg_com_quat = self.peg_quat  # always equal

        peg_velocities = self.pegs.get_velocities(clone=False)
        self.peg_linvel = peg_velocities[:, 0:3]        # linear velocities
        self.peg_angvel = peg_velocities[:, 3:6]        # angular velocities

        # ?? neded for my task? (so far only copied and adjusted from nut_bolt_place env)
        self.peg_com_linvel = self.peg_linvel + torch.cross(        # torch.cross(input, other, dim=None, *, out=None) → Tensor 
                                                                    # Returns the cross product of vectors in dimension dim of input and other.
            
            self.peg_angvel,                                        # input
            (self.peg_com_pos - self.peg_pos),                      # other
            dim=1
        )
        self.peg_com_angvel = self.peg_angvel  # always equal

        self.peg_force = self.pegs.get_net_contact_forces(clone=False)

        # Hole tensors
        self.hole_pos, self.hole_quat = self.holes.get_world_poses(clone=False)
        self.hole_pos -= self.env_pos

        self.hole_force = self.holes.get_net_contact_forces(clone=False)
