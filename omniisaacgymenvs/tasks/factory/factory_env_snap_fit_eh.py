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

# deformable imports (in franka_deformable-task only DeformablePrimView is used)
import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView

ops="windows" # (choose if "windows" or "linux")

class FactoryEnvSnapFit_eh(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""
        # print("init")

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()


### CHECK HERE
    def _get_env_yaml_params(self): 
        # print("_get_env_yaml_params")
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/FactoryEnvSnapFit_eh.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_snap_fit_eh.yaml"
        self.asset_info_snap_fit=hydra.compose(config_name=asset_info_path)
        self.asset_info_snap_fit = self.asset_info_snap_fit[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting


    def update_config(self, sim_config): 
        # print("update_config")
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]

        self._get_env_yaml_params()

    def set_up_scene(self, scene) -> None: #TODO
        """Import assets. Add to scene."""
        # print("set_up_scene")

        # Increase buffer size to prevent overflow for Place and Screw tasks
        # print(" get physx context")
        physxSceneAPI = self._env._world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)
        # print("import franka assets")
        self.import_franka_assets(add_to_stage=True) # defined in factory_base --> probably fine as is
        # self.create_snap_fit_material() --> TODO: brauchen wir das? reicht es wenn wir das Material in usd festlegen?
        # print("RLTask set up scene")
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        # print("import assets")
        self._import_env_assets(add_to_stage=True)


        # print("FactoryFrankaView")
        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )

        self.males = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male/collisions/mesh_01.*", name="males_view", track_contact_forces=True, # TODO: adjust male.usd correctly; rigid prim view has to get the mesh paths as input
        )

        # print("RigidPrimView")
        self.females = RigidPrimView(
            prim_paths_expr="/World/envs/.*/female/female.*", name="females_view", track_contact_forces=True, # TODO: adjust female.usd correctly
        )

        # TODO: scene.add
        scene.add(self.males)
        scene.add(self.females)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

        return

    def initialize_views(self, scene) -> None: #TODO
        """Initialize views for extension workflow."""
        # print("initialize_views")

        super().initialize_views(scene)

        self.import_franka_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)


        # print("remove frankas view ")
        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        # print("remove  male view ")
        if scene.object_exists("males_view"):
            scene.remove_object("males_view", registry_only=True)

        # print("remove female view ")
        if scene.object_exists("females_view"):
            scene.remove_object("females_view", registry_only=True)
        # print("remove hands view ")#
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        # print("remove lfingers_view ")
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        # print("remove rfingers_view ")
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        # print("remove fingertips_view ")
        if scene.object_exists("fingertips_view"):
            scene.remove_object("fingertips_view", registry_only=True)
        # print("FactoryFrankaView")
            
        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )

        self.males = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male/collisions/mesh_01_base.*", name="males_view"
        )
        # print("RigidPrimView")
        self.females = RigidPrimView(
            prim_paths_expr="/World/envs/.*/female/female.*", name="females_view"
        )

        # print("add males_arms")
        scene.add(self.males_arms)
        # print("add males_base")
        scene.add(self.males_base)
        # print("add females")
        scene.add(self.females)
        # print("add frankas")
        scene.add(self.frankas)
        # print("add _hands")
        scene.add(self.frankas._hands)
        # print("add _lfingers")
        scene.add(self.frankas._lfingers)
        # print("add _rfingers")##
        scene.add(self.frankas._rfingers)
        # print("add _fingertip_centered")
        scene.add(self.frankas._fingertip_centered)

    def create_snap_fit_material(self): #TODO: only for female und male_base?
        """Define snapfit material."""
        # print("create_snapfit_material")


        self.MalePhysicsMaterialPath = "/World/Physics_Materials/MaleMaterial"
        self.FemalePhysicsMaterialPath = "/World/Physics_Materials/FemaleMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.FemalesPhysicsMaterialPath,
            density=self.cfg_env.env.snap_fit_density,                  # from config file (FactoryEnvSnapFit_eh.yaml) # TODO: does density defined her overwrite the density/mass in usd? I guess it does not overwrite the mass?
            staticFriction=self.cfg_env.env.snap_fit_friction,         # from config file ((FactoryEnvSnapFit_eh.yaml) # TODO: reserach values for friction
            dynamicFriction=self.cfg_env.env.snap_fit_friction,       # from config file (FactoryEnvSnapFit_eh.yaml)
            restitution=0.0,
        )
        utils.addRigidBodyMaterial(
            self._stage,
            self.MaleBasePhysicsMaterialPath,
            density=self.cfg_env.env.snap_fit_density,                  # from config file (FactoryEnvSnapFit_eh.yaml) # TODO: does density defined her overwrite the density/mass in usd? I guess it does not overwrite the mass?
            staticFriction=self.cfg_env.env.snap_fit_friction,         # from config file ((FactoryEnvSnapFit_eh.yaml) # TODO: reserach values for friction
            dynamicFriction=self.cfg_env.env.snap_fit_friction,       # from config file (FactoryEnvSnapFit_eh.yaml)
            restitution=0.0,
        )


    def _import_env_assets(self, add_to_stage=True): #TODO
        """Set snap-fit asset options. Import assets."""
        # print("_import_env_assets")
        print("1")
        assets_root_path = get_assets_root_path()
        if ops=="linux":
            usd_path="usd_path_linux"
        if ops=="windows":
            usd_path="usd_path_windows"   

        print("2")
        self.male_thickness=[]
        self.male_heights_base=[]
        self.male_heights_total = []
        self.male_widths = []
        self.female_widths = []
        self.female_heights = [] # TODO: check whether we need more lists to store any more values

        print("3")
        for i in range(0, self._num_envs):                                              # für jede einzelne env
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))       # desired subassemblies aus config datei von env
            subassembly = self.cfg_env.env.desired_subassemblies[j]                     # z.B. desired_subassembly = ['snap_fit_1', 'snap_fit_1']. Aktuell nur diese desired_subassembly möglich. 
            components = list(self.asset_info_snap_fit[subassembly])                    # werden hier zufällig verschiedene Varianten für jede einzelne env ausgewählt? (in diesem Fall alle gleich da beide Elemente in desired_subassemblies identisch sind?) 
            # print("components_list: ", components)
            # print("4:,",i)
            male_translation = torch.tensor(                                             # passt das so??
                [
                    0.0,
                    self.cfg_env.env.male_lateral_offset,
                    self.cfg_base.env.table_height,                                     # from config file FactoryBase.yaml
                ],
                device=self._device,
            )
            male_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)  # TODO: maybe has to be adjusted? 

            # print("5:,",i)

            # ANNAHME: Wir nehmen nur eine Variante von subassemblies, daher brauchen wir die heights etc nicht? 
            # Und: heights etc. bereits durch cad-datei vorgegeben?

            male_height_total = self.asset_info_snap_fit[subassembly][components[0]]["height_total"]    # aus factory_asset_info_nut_bolt datei
            male_width = self.asset_info_snap_fit[subassembly][components[0]]["width"]      # aus factory_asset_info_nut_bolt datei
            male_height_base=self.asset_info_snap_fit[subassembly][components[0]]["height_base"]

            self.male_widths.append(male_width)
            self.male_heights_total.append(male_height_total)
            self.male_heights_base.append(male_height_base)   

            male_file = self.asset_info_snap_fit[subassembly][components[0]][usd_path]      # aus factory_asset_info_nut_bolt datei; müsste auch über asset path funktionieren
            # print("male_file: ",male_file)
            # print("5:,",i)
            if add_to_stage:                                                                # immer TRUE?? (s.oben in def _import_env_assets..) --> JA
                add_reference_to_stage(male_file, f"/World/envs/env_{i}" + "/male")
                # male_prim=add_reference_to_stage(male_file, f"/World/envs/env_{i}" + "/male")
                # print("6:,",i)
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/male",
                    translation=male_translation,
                    orientation=male_orientation,
                )
                
                # print("6:,",i)
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/male/{components[0]}/collisions" 
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/male/{components[0]}/collisions/mesh_01"
                    ),
                    
                    self.MalePhysicsMaterialPath,
                )
                # print("7:,",i)
                ##### TODO: Check out task config file --> does task config file only have to have the same name as the task?

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "male",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/male"),
                    self._sim_config.parse_actor_config("male"),
                )
            # print("8:,",i)
            female_translation = torch.tensor(
                [0.0, 0.0, self.cfg_base.env.table_height], device=self._device                             # from config file FactoryBase.yaml
            )
            female_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            female_width = self.asset_info_snap_fit[subassembly][components[1]]["width"]                      # quadratische Grundfläche 
            female_height = self.asset_info_snap_fit[subassembly][components[1]]["height"]
            

            self.female_widths.append(female_width)
            self.female_heights.append(female_height)


            female_file = self.asset_info_snap_fit[subassembly][components[1]][usd_path]
            # print("9:,",i)
            if add_to_stage:
                add_reference_to_stage(female_file, f"/World/envs/env_{i}" + "/female")
                # female_prim=add_reference_to_stage(female_file, f"/World/envs/env_{i}" + "/female")

                # UsdPhysics.CollisionAPI.Apply(female_prim)
                # print("10: ",i)
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/female",
                    translation=female_translation,
                    orientation=female_orientation,
                )
                # print("11: ",i)
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/female/{components[1]}/collisions"
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/female/{components[1]}/collisions/mesh_01"
                    ),
                    self.FemalePhysicsMaterialPath,
                )
                # print("12: ",i)
                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "female",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/female"),
                    self._sim_config.parse_actor_config("female"),
                )


        ####TODO: How do i transform this to my problem???
        # print("13: ")        
        # For computing body COM pos
        self.male_heights = torch.tensor(
            self.male_heights, device=self._device
        ).unsqueeze(-1) # adds a new dimension at the last axis (-1) of the tensor
        self.female_heights = torch.tensor(
            self.female_heights, device=self._device
        ).unsqueeze(-1)

        # self.hole_drill_hole_heights = torch.tensor( # TODO: what is the snap-fit-equivalent?
        #     self.hole_drill_hole_heights, device=self._device
        # ).unsqueeze(-1)

        self.list_of_zeros = [0 for _ in range(self.num_envs)]
        
        self.base_pos_zeros = torch.tensor(
            self.list_of_zeros, device=self._device
        ).unsqueeze(-1)



        # print("14: ")

        # For setting initial state - 
        self.male_thickness = torch.tensor( # --> used when resetting gripper  
            self.male_widths, device=self._device
        ).unsqueeze(-1)

    def refresh_env_tensors(self): #TODO
        """Refresh tensors."""
        # print("refresh_env_tensors")

        # male tensors
        self.male_pos, self.male_quat = self.males.get_world_poses(clone=False)            # positions in world frame of the prims with shape (M,3), quaternion orientations in world frame of the prims (scalar-first (w,x,y,z), shape (M,4))
        ''' get world_poses(indices, clone)
        indices: size (M,) Where M <= size of the encapsulated prims in the view. Defaults to None (i.e: all prims in the view).
        here: M=num_envs?
        clone (bool, optional) – True to return a clone of the internal buffer. Otherwise False. Defaults to True.
        '''
        self.male_pos -= self.env_pos                                                    # A-=B is equal to A=A-B; male position relative to env position. env position is absolute to world

        ### TODO: WHAT is a com pos --> Center of mass position --> only needed in screw task?
        self.male_com_pos = fc.translate_along_local_z(                                  # translate_along_local_z(pos, quat, offset, device). Translate global body position along local Z-axis and express in global coordinates
            pos=self.male_pos,           # "measured" via get_world_poses
            quat=self.male_quat,         # "measured" via get_world_poses


            ### TODO: what to put here (offset) 
            offset=self.hole_drill_hole_heights + self.male_heights * 0.5, #  TODO: what is the snap-fit equivalent
            device=self.device,
        )
        self.male_com_quat = self.male_quat  # always equal

        male_velocities = self.males.get_velocities(clone=False)
        self.male_linvel = male_velocities[:, 0:3]        # linear velocities
        self.male_angvel = male_velocities[:, 3:6]        # angular velocities

        # ?? neded for my task? (so far only copied and adjusted from nut_bolt_place env)
        self.male_com_linvel = self.male_linvel + torch.cross(        # torch.cross(input, other, dim=None, *, out=None) → Tensor 
                                                                    # Returns the cross product of vectors in dimension dim of input and other.
            
            self.male_angvel,                                        # input
            (self.male_com_pos - self.male_pos),                      # other
            dim=1
        )
        self.male_com_angvel = self.male_angvel  # always equal

        self.male_force = self.males.get_net_contact_forces(clone=False)

        # female tensors
        self.female_pos, self.female_quat = self.females.get_world_poses(clone=False)
        self.female_pos -= self.env_pos

        self.female_force = self.females.get_net_contact_forces(clone=False)
