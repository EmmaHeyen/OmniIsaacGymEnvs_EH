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

"""Factory-class for snap-fit environment.

Defines environment for snap-fit task.
Inherits base class and abstract environment class. Is inherited by the snap-fit task class (factory_task_snap_fit_eh.py). 

check FactoryEnvSnapFit_eh.yaml to set/define configurations. info about assets (male and female part) is defined in factory_asset_info_snap_fit_eh.yaml

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
from omni.isaac.core.objects import DynamicCuboid

ops="windows" # (choose if "windows" or "linux")

class FactoryEnvSnapFit_eh(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""
        # print("init")

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()

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

    def set_up_scene(self, scene) -> None: 
        """Import assets. Add to scene."""

        # Increase buffer size to prevent overflow
        physxSceneAPI = self._env._world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)
        physxSceneAPI.CreateGpuFoundLostPairsCapacityAttr().Set(256 * 1024)

        self.import_franka_assets(add_to_stage=True) # defined in factory_base.py
        self.create_snap_fit_material() 

        RLTask.set_up_scene(self, scene, replicate_physics=False)

        self._import_env_assets(add_to_stage=True)

        # create views
        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.males_base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male/base.*", name="males_view_base", track_contact_forces=True, 
        )
        self.males_armRight = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male/arm_Right.*", name="males_view_armRight", track_contact_forces=True, 
        )
        self.males_armLeft = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male/arm_Left.*", name="males_view_armLeft", track_contact_forces=True, 
        )

        self.females = RigidPrimView(
            prim_paths_expr="/World/envs/.*/female/female.*", name="females_view", track_contact_forces=True, 
        )         

        scene.add(self.males_base)
        scene.add(self.males_armRight)
        scene.add(self.males_armLeft)

        scene.add(self.females)

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
        if scene.object_exists("males_view"):
            scene.remove_object("males_view", registry_only=True) # TODO!

        if scene.object_exists("females_view"):
            scene.remove_object("females_view", registry_only=True)

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
        self.males_base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male.*", name="males_view_base", track_contact_forces=True, 
        )
        self.males_armRight = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male.*", name="males_view_armRight", track_contact_forces=True, 
        )
        self.males_armLeft = RigidPrimView(
            prim_paths_expr="/World/envs/.*/male/male.*", name="males_view_armLeft", track_contact_forces=True, 
        )
        self.females = RigidPrimView(
            prim_paths_expr="/World/envs/.*/female/female.*", name="females_view"
        )

        scene.add(self.males_base)
        scene.add(self.males_armRight)
        scene.add(self.males_armLeft)

        scene.add(self.females)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

    def create_snap_fit_material(self): 
        """Define snapfit material."""

        self.MalePhysicsMaterialPath = "/World/Physics_Materials/MaleMaterial"
        self.FemalePhysicsMaterialPath = "/World/Physics_Materials/FemaleMaterial"
        self.FrankaFingerPhysicsMaterialPath = "/World/Physics_Materials/FrankaFingerMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.FemalePhysicsMaterialPath,
            density=self.cfg_env.env.snap_fit_density,                  # config file (FactoryEnvSnapFit_eh.yaml) 
            staticFriction=self.cfg_env.env.snap_fit_friction,          # config file (FactoryEnvSnapFit_eh.yaml) 
            dynamicFriction=self.cfg_env.env.snap_fit_friction,         # config file (FactoryEnvSnapFit_eh.yaml)
            restitution=0.0,
        )
        utils.addRigidBodyMaterial(
            self._stage,
            self.MalePhysicsMaterialPath,
            density=self.cfg_env.env.snap_fit_density,                  # config file (FactoryEnvSnapFit_eh.yaml)
            staticFriction=self.cfg_env.env.snap_fit_friction,          # config file ((FactoryEnvSnapFit_eh.yaml) 
            dynamicFriction=self.cfg_env.env.snap_fit_friction,         # config file (FactoryEnvSnapFit_eh.yaml)
            restitution=0.0,
        )

        # add material with high density to Franka fingers to "fix" male part in gripper 
        utils.addRigidBodyMaterial(
            self._stage,
            self.FrankaFingerPhysicsMaterialPath,
            density=self.cfg_env.env.snap_fit_density,                  # config file (FactoryEnvSnapFit_eh.yaml) 
            staticFriction=1000,      
            dynamicFriction=1000,      
            restitution=0.0,
        )

    def _import_env_assets(self, add_to_stage=True): 
        """Set snap-fit asset options. Import assets."""
        # print("_import_env_assets")
        print("1")
        assets_root_path = get_assets_root_path()
        if ops=="linux":
            usd_path="usd_path_linux"
        if ops=="windows":
            usd_path="usd_path_windows"   

        self.male_thicknesses=[]
        self.male_heights_base=[]
        self.male_heights_total = []
        self.male_keypoint_middle_points = []
        self.female_widths = []
        self.female_heights = [] 
        self.checkpoint_z_pos = []

        for i in range(0, self._num_envs):                                              # für jede einzelne env
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))       # desired subassemblies aus FactoryEnvSnapFit_eh.yaml
            subassembly = self.cfg_env.env.desired_subassemblies[j]                     # z.B. desired_subassembly = ['snap_fit_1', 'snap_fit_1'].
            components = list(self.asset_info_snap_fit[subassembly])                    
            male_translation = torch.tensor(                                             
                [
                    0.0,
                    self.cfg_env.env.male_lateral_offset,
                    self.cfg_base.env.table_height,                                     # config file FactoryBase.yaml
                ],
                device=self._device,
            )
            male_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device) 

            male_height_total = self.asset_info_snap_fit[subassembly][components[0]]["height_total"]    # aus factory_asset_info_snap_fit_eh.yaml
            male_height_base=self.asset_info_snap_fit[subassembly][components[0]]["height_base"]
            male_thickness=self.asset_info_snap_fit[subassembly][components[0]]["thickness"]
            male_kp_middle_point=self.asset_info_snap_fit[subassembly][components[0]]["keypoint_middle_point_from_KOS"]
            checkpoint_z_pos = self.asset_info_snap_fit[subassembly][components[0]]["z_pos_checkpoint"]

            self.male_heights_total.append(male_height_total)
            self.male_heights_base.append(male_height_base)   
            self.male_thicknesses.append(male_thickness)   
            self.male_keypoint_middle_points.append(male_kp_middle_point)
            self.checkpoint_z_pos.append(checkpoint_z_pos)

            male_file = self.asset_info_snap_fit[subassembly][components[0]][usd_path]      

            if add_to_stage:                                                               
                add_reference_to_stage(male_file, f"/World/envs/env_{i}" + "/male")

                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/male",
                    translation=male_translation,
                    orientation=male_orientation,
                )
                
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/male/{components[0]}/base/collisions_base" 
                ).SetInstanceable(
                    False
                )  # required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/male/{components[0]}/base/collisions_base"
                    ),
                    self.MalePhysicsMaterialPath,
                )


                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/male/{components[0]}/arm_Right/collisions_armRight" 
                ).SetInstanceable(
                    False
                ) 
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/male/{components[0]}/arm_Right/collisions_armRight"
                    ),
                    self.MalePhysicsMaterialPath,
                )
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/male/{components[0]}/arm_Left/collisions_armLeft" 
                ).SetInstanceable(
                    False
                ) 
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/male/{components[0]}/arm_Left/collisions_armLeft"
                    ),
                    self.MalePhysicsMaterialPath,
                )


                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/franka/panda_leftfinger/collisions" 
                ).SetInstanceable(
                    False 
                ) 
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}" + f"/franka/panda_leftfinger/collisions" 
                    ),
                    self.FrankaFingerPhysicsMaterialPath,
                )

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/franka/panda_rightfinger/collisions" 
                ).SetInstanceable(
                    False
                ) 
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}" + f"/franka/panda_rightfinger/collisions" 
                    ),
                    self.FrankaFingerPhysicsMaterialPath,
                )

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "male",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/male"),
                    self._sim_config.parse_actor_config("male"),
                )

            female_translation = torch.tensor(
                [0.0, 0.0, self.cfg_base.env.table_height], device=self._device                             # from config file FactoryBase.yaml
            )
            female_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            female_width = self.asset_info_snap_fit[subassembly][components[1]]["width_top_inner"]             
            female_height = self.asset_info_snap_fit[subassembly][components[1]]["height_total"]
            
            self.female_widths.append(female_width)
            self.female_heights.append(female_height)

            female_file = self.asset_info_snap_fit[subassembly][components[1]][usd_path]
 
            if add_to_stage:
                add_reference_to_stage(female_file, f"/World/envs/env_{i}" + "/female")

                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/female",
                    translation=female_translation,
                    orientation=female_orientation,
                )

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/female/{components[1]}/collisions"
                ).SetInstanceable(
                    False
                )  # required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/female/{components[1]}/collisions"
                    ),
                    self.FemalePhysicsMaterialPath,
                )

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "female",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/female"),
                    self._sim_config.parse_actor_config("female"),
                )
                
                # add fixed joints to females
                prim_joint=utils.createJoint(self._stage, "Fixed",self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/female/female"),self._stage.GetPrimAtPath(f"/World"))


        # For computing body COM pos
        self.male_heights_total = torch.tensor(
            self.male_heights_total, device=self._device
        ).unsqueeze(-1) # adds a new dimension at the last axis (-1) of the tensor
        self.male_heights_base = torch.tensor(
            self.male_heights_base, device=self._device
        ).unsqueeze(-1)
        self.checkpoint_z_pos = torch.tensor(
            self.checkpoint_z_pos, device=self._device
        ).unsqueeze(-1)
        self.female_heights = torch.tensor(
            self.female_heights, device=self._device
        ).unsqueeze(-1)
        self.male_keypoint_middle_points = torch.tensor(
            self.male_keypoint_middle_points, device=self._device
        ).unsqueeze(-1)
        self.list_of_zeros = [0 for _ in range(self.num_envs)]
        self.base_pos_zeros = torch.tensor(
            self.list_of_zeros, device=self._device
        ).unsqueeze(-1)


        # For setting initial state - 
        self.male_thicknesses = torch.tensor( # --> used when resetting gripper  
            self.male_thicknesses, device=self._device
        ).unsqueeze(-1)

    def _import_test_cubes(self, add_to_stage):

        for i in range(0, self._num_envs):      
                                     
            cube_z_pos = self.asset_info_snap_fit["snap_fit_1"]["male"]["z_pos_checkpoint"]

            cube_translation = torch.tensor(                                           
                [
                    0.0,
                    0.0,
                    self.cfg_base.env.table_height + 0.1125,                                     # from config file FactoryBase.yaml
                ],
                device=self._device,
            )
            

    def refresh_env_tensors(self): 
        """Refresh environment tensors."""
        ''' get world_poses(indices, clone)
        indices: size (M,) Where M <= size of the encapsulated prims in the view. Defaults to None (i.e: all prims in the view).
        here: M=num_envs
        clone (bool, optional) – True to return a clone of the internal buffer. Otherwise False. Defaults to True.
        '''
        # male tensors
        self.male_pos_base, self.male_quat_base = self.males_base.get_world_poses(clone=False)  # positions in world frame of the prims with shape (M,3), quaternion orientations in world frame of the prims (scalar-first (w,x,y,z), shape (M,4))
        self.male_pos_base -= self.env_pos                                                      # A-=B is equal to A=A-B; male position relative to env position. env position is absolute to world

        self.male_pos_armLeft, self.male_quat_armLeft = self.males_armLeft.get_world_poses(clone=False) 
        self.male_pos_armLeft -= self.env_pos 
        self.male_pos_armRight, self.male_quat_armRight = self.males_armRight.get_world_poses(clone=False) 
        self.male_pos_armRight-= self.env_pos 

        self.male_com_pos = fc.translate_along_local_z(                                         # translate_along_local_z(pos, quat, offset, device). Translate global body position along local Z-axis and express in global coordinates
            pos=self.male_pos_base,                                                             # "measured" via get_world_poses
            quat=self.male_quat_base,                                                           # "measured" via get_world_poses
            offset=self.male_heights_total * 0.5, 
            device=self.device,
        )
        self.male_com_quat = self.male_quat_base  # always equal

        male_velocities_base = self.males_base.get_velocities(clone=False)
        male_velocities_armRight = self.males_armRight.get_velocities(clone=False)
        male_velocities_armLeft = self.males_armLeft.get_velocities(clone=False)


        self.male_linvel_base = male_velocities_base[:, 0:3]        # linear velocities 
        self.male_angvel_base = male_velocities_base[:, 3:6]        # angular velocities

        self.male_linvel_armRight = male_velocities_base[:, 0:3] 
        self.male_angvel_armRight = male_velocities_base[:, 3:6]
        self.male_linvel_armLeft = male_velocities_base[:, 0:3] 
        self.male_angvel_armLeft = male_velocities_base[:, 3:6]


        self.male_com_linvel = self.male_linvel_base + torch.cross(         # torch.cross(input, other, dim=None, *, out=None) → Tensor 
                                                                            # Returns the cross product of vectors in dimension dim of input and other.
            
            self.male_angvel_base,                                          # input
            (self.male_com_pos - self.male_pos_base),                       # other
            dim=1
        )
        self.male_com_angvel = self.male_angvel_base  # always equal

        self.male_force_armRight = self.males_armRight.get_net_contact_forces(clone=False) 
        self.male_force_armLeft = self.males_armLeft.get_net_contact_forces(clone=False) 
        self.male_force = self.males_base.get_net_contact_forces(clone=False) 

        self.female_pos, self.female_quat = self.females.get_world_poses(clone=False)     

        # to use positions relative to env_pos: 
        self.female_pos -= self.env_pos
        self.female_force = self.females.get_net_contact_forces(clone=False)
