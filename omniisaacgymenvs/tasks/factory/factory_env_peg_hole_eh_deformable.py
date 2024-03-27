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

"""Factory: class for peg-hole env.

Inherits base class and abstract environment class. Inherited by peg-hole task classes. Not directly executed.

Configuration defined in FactoryEnvPegHole_eh_deformable.yaml. Asset info defined in factory_asset_info_peg_hole_eh_deformable.yaml.
"""




import hydra
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, PhysxSchema

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.physx.scripts import physicsUtils, utils, deformableUtils

from omniisaacgymenvs.robots.articulations.views.factory_franka_view import (
    FactoryFrankaView,
)
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)

# deformableImports:
import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView


ops="linux" # (choose if "windows" or "linux")

class FactoryEnvPegHole_eh_deformable(FactoryBase, FactoryABCEnv):
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
            "task/FactoryEnvPegHole_eh_deformable.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_peg_hole_eh_deformable.yaml"
        self.asset_info_peg_hole=hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole[""][""][""]["tasks"][
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
        # print("set_up_scene")

        # Increase buffer size to prevent overflow for Place and Screw tasks
        # print(" get physx context")
        physxSceneAPI = self._env._world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)
        # print("import franka assets")
        self.import_franka_assets(add_to_stage=True) # defined in factory_base --> probably fine as is
        self.create_peg_hole_material()
        # print("RLTask set up scene")
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        # print("import assets")
        self._import_env_assets(add_to_stage=True)

        # self.frankas = FactoryFrankaView(
        #     prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        # )
        # self.pegs = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/peg/peg.*",
        #     name="pegs_view",
        #     track_contact_forces=True,
        # )
        # self.holes = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/hole/hole.*",
        #     name="holes_view",
        #     track_contact_forces=True,
        # )
        # print("FactoryFrankaView")
        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        # print("RigidPrimView")
        # self.pegs = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/peg/peg.*", name="pegs_view", track_contact_forces=True,
        # )
        print("deformablePrimView_deformable_peg1") 
        self.pegs = DeformablePrimView( 
            prim_paths_expr="/World/envs/.*/peg/peg/collisions/mesh_01", name="pegs_view_deform") # muss hier noch mesh_01 hin?
        print("deformablePrimView_deformable_peg2") 




        print("RigidPrimView2")
        self.holes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/hole/hole.*", name="holes_view", track_contact_forces=True,
        )

        # scene.add(self.pegs)
        # scene.add(self.holes)
        # scene.add(self.frankas)
        # scene.add(self.frankas._hands)
        # scene.add(self.frankas._lfingers)
        # scene.add(self.frankas._rfingers)
        # scene.add(self.frankas._fingertip_centered)


        print("add pegs")
        scene.add(self.pegs)
        print("add holes")
        scene.add(self.holes)
        print("add frankas")
        scene.add(self.frankas)
        print("add _hands")#
        scene.add(self.frankas._hands)
        print("add _lfingers")
        scene.add(self.frankas._lfingers)
        print("add _rfingers")
        scene.add(self.frankas._rfingers)
        print("add _fingertip_centered")
        scene.add(self.frankas._fingertip_centered)
        print("done setting up scene")

        return

    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""
        print("initialize_views")

        super().initialize_views(scene)

        self.import_franka_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)


        # print("remove frankas view ")
        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        # print("remove  pegs view ")

        if scene.object_exists("pegs_view_deform"): 
            scene.remove_object("pegs_view_deform", registry_only=True)
        # print("remove holes view ")
        if scene.object_exists("holes_view"):
            scene.remove_object("holes_view", registry_only=True)
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
        # print("RigidPrimView")
        # self.pegs = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/peg/peg.*", name="pegs_view"
        # )

        print("deformablePrimView_deformable_peg3") # test 25.03.2024 deformable peg
        self.pegs = DeformablePrimView(
            prim_paths_expr="/World/envs/.*/peg/peg/collisions/mesh_01", name="pegs_view_deform" # muss hier noch mesh_01 hin?
        )
        print("deformablePrimView_deformable_peg4") 



        print("RigidPrimView2")
        self.holes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/hole/hole.*", name="holes_view"
        )
        # print("add pegs")
        # scene.add(self.pegs) # test 25.03.2024
        scene.add(self.pegs)
        # print("add holes")
        scene.add(self.holes)
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

    def create_peg_hole_material(self):
        """Define peg and hole material."""
        # print("create_peg_hole_material")

        # self.PegHolePhysicsMaterialPath = "/World/Physics_Materials/PegHoleMaterial"
        # self.PegPhysicsMaterialPath = "/World/Physics_Materials/PegMaterial"
        self.PegPhysicsMaterialPath_deformable = "/World/Physics_Materials/PegMaterial_deform"
        self.HolePhysicsMaterialPath = "/World/Physics_Materials/HoleMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.HolePhysicsMaterialPath,
            density=self.cfg_env.env.peg_hole_density,                  # from config file (FactoryEnvPegHole_eh.yaml)
            staticFriction=self.cfg_env.env.peg_hole_friction,         # from config file (FactoryEnvPegHole_eh.yaml)
            dynamicFriction=self.cfg_env.env.peg_hole_friction,       # from config file (FactoryEnvPegHole_eh.yaml)
            restitution=0.0,
        )
        # utils.addRigidBodyMaterial(
        #     self._stage,
        #     self.PegPhysicsMaterialPath,
        #     density=self.cfg_env.env.peg_hole_density,                  # from config file (FactoryEnvPegHole_eh.yaml)
        #     # staticFriction=peg_hole_friction,  
        #     staticFriction=self.cfg_env.env.peg_hole_friction,         # from config file (FactoryEnvPegHole_eh.yaml)
        #     # dynamicFriction=peg_hole_friction,   
        #     dynamicFriction=self.cfg_env.env.peg_hole_friction,       # from config file (FactoryEnvPegHole_eh.yaml)
        #     restitution=0.0,
        # )
        print("add deform material")
        deformableUtils.add_deformable_body_material( # test 25.03.2024
            self._stage,
            self.PegPhysicsMaterialPath_deformable,
            damping_scale=None,
            density=None,
            dynamic_friction=None,
            elasticity_damping=None,
            poissons_ratio=None,
            youngs_modulus=None,
        )

    def _import_env_assets(self, add_to_stage=True):
        """Set peg and hole asset options. Import assets."""
        # print("_import_env_assets")
        print("1")
        assets_root_path = get_assets_root_path()
        if ops=="linux":
            usd_path="usd_path_linux"
        if ops=="windows":
            usd_path="usd_path_windows"   

        print("2")
        self.peg_heights = []
        self.peg_widths = []
        self.hole_widths = []
        self.hole_heights = []
        self.hole_drill_hole_heights=[]
        # self.bolt_shank_lengths = []
        # self.thread_pitches = []

        
        print("3")
        for i in range(0, self._num_envs):                                              # für jede einzelne env
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))       # desired subassemblies aus config datei von env
            subassembly = self.cfg_env.env.desired_subassemblies[j]                     # z.B. desired_subassembly = ['peg_hole_1', 'peg_hole_1']. Aktuell nur diese desired_subassembly möglich. 
            components = list(self.asset_info_peg_hole[subassembly])                    # werden hier zufällig verschiedene Varianten für jede einzelne env ausgewählt? (in diesem Fall alle gleich da beide Elemente in desired_subassemblies identisch sind?) 
            # print("components_list: ", components)
            # print("4:,",i)
            peg_translation = torch.tensor(                                             # passt das so??
                [
                    0.0,
                    self.cfg_env.env.peg_lateral_offset,
                    self.cfg_base.env.table_height,                                     # from config file FactoryBase.yaml
                ],
                device=self._device,
            )
            peg_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            # print("5:,",i)

            # ANNAHME: Wir nehmen nur eine Variante von subassemblies, daher brauchen wir die heights etc nicht? 
            # Und: heights etc. bereits durch cad-datei vorgegeben?

            peg_height = self.asset_info_peg_hole[subassembly][components[0]]["height"] # aus factory_asset_info_nut_bolt datei
            peg_width = self.asset_info_peg_hole[subassembly][components[0]][       # aus factory_asset_info_nut_bolt datei
                "width"
            ]
            self.peg_heights.append(peg_height)                                         # nut_height=height of nut. different for different subassemblies --> is probably measurement of nut
            self.peg_widths.append(peg_width)

            peg_file = self.asset_info_peg_hole[subassembly][components[0]][usd_path] # aus factory_asset_info_nut_bolt datei; müsste auch über asset path funktionieren
            # print("peg_file: ",peg_file)
            # print("5:,",i)
            if add_to_stage:                                                            # immer TRUE?? (s.oben in def _import_env_assets..)
                add_reference_to_stage(peg_file, f"/World/envs/env_{i}" + "/peg")
                # peg_prim=add_reference_to_stage(peg_file, f"/World/envs/env_{i}" + "/peg")
                # print("6:,",i)
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/peg",
                    translation=peg_translation,
                    orientation=peg_orientation,
                )
                # UsdPhysics.CollisionAPI.Apply(peg_prim)
                # print("6:,",i)
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/peg/{components[0]}/collisions/mesh_01" 
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                # print("add material to peg prim")
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/peg/{components[0]}/collisions/mesh_01"
                    ),
                    # self.PegHolePhysicsMaterialPath,
                    self.PegPhysicsMaterialPath_deformable, # test 25.03.2024 deformable prim
                )
                # print("7:,",i)
                ##### TODO: Check out task config file -->does task config file only have to have the same name as the task?
                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "peg",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/peg"),
                    self._sim_config.parse_actor_config("peg"),
                )
            # print("8:,",i)
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
            # print("9:,",i)
            if add_to_stage:
                add_reference_to_stage(hole_file, f"/World/envs/env_{i}" + "/hole")
                # hole_prim=add_reference_to_stage(hole_file, f"/World/envs/env_{i}" + "/hole")

                # UsdPhysics.CollisionAPI.Apply(hole_prim)
                # print("10: ",i)
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/hole",
                    translation=hole_translation,
                    orientation=hole_orientation,
                )
                # print("11: ",i)
                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/hole/{components[1]}/collisions/mesh_01"
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/hole/{components[1]}/collisions/mesh_01"
                    ),
                    # self.PegHolePhysicsMaterialPath,
                    self.HolePhysicsMaterialPath,
                )
                # print("12: ",i)
                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "hole",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/hole"),
                    self._sim_config.parse_actor_config("hole"),
                )

            # thread_pitch = self.asset_info_peg_hole[subassembly]["thread_pitch"]            # Gewindegang
            # self.thread_pitches.append(thread_pitch)


        ####TODO: How do i transform this to my problem???
        # print("13: ")        
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

        self.list_of_zeros = [0 for _ in range(self.num_envs)]
        
        self.base_pos_zeros = torch.tensor(
            self.list_of_zeros, device=self._device
        ).unsqueeze(-1)

        # print("14: ")

        # For setting initial state - 
        self.peg_widths = torch.tensor( # --> used when resetting gripper  
            self.peg_widths, device=self._device
        ).unsqueeze(-1)

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # print("refresh_env_tensors")

        # Peg tensors
        '''
        def get_simulation_mesh_nodal_positions(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Gets the nodal positions of the simulation mesh for the deformable bodies indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which deformable prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: position tensor with shape (M, max_simulation_mesh_vertices_per_body, 3)
        """

        def get_simulation_mesh_nodal_velocities(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Gets the vertex velocities for the deformable bodies indicated by the indices."""
        '''
        self.peg_pos = self.pegs.get_simulation_mesh_nodal_positions(clone=False) # position tensor with shape (M, max_simulation_mesh_vertices_per_body, 3) where M <= size of the encapsulated prims in the view.
        self.peg_velocities = self.pegs.get_simulation_mesh_nodal_velocities(clone=False) # velocity tensor with shape (M, max_simulation_mesh_vertices_per_body, 3)
        # print("shape peg_pos: ",self.peg_pos.shape) # ([128,176,3])
        # print("shape peg_vel: ",self.peg_velocities.shape) # ([128,176,3])
        # print("shape env_pos: ", self.env_pos.shape) # ([128,3])

        self.peg_pos_middle = self.peg_pos[:,88,:] # middle part?
        self.peg_pos_middle -= self.env_pos                                                    # A-=B is equal to A=A-B; peg'position relative to env position. env position is absolute to world

        self.identity_quat = ( # TODO quat berücksichtigen?
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # ### TODO: WHAT is a com pos --> Center of mass position --> only needed in screw task?
        # self.peg_com_pos = fc.translate_along_local_z(                                  # translate_along_local_z(pos, quat, offset, device). Translate global body position along local Z-axis and express in global coordinates
        #     pos=self.peg_pos,           # "measured" via get_world_poses
        #     # quat=self.peg_quat,   
        #     quat=self.identity_quat,      # "measured" via get_world_poses


        #     ### TODO: what to put here (offset) --> Schwerpunkt wenn fertig assembled?? TODO
        #     offset=self.hole_drill_hole_heights + self.peg_heights * 0.5,
        #     device=self.device,
        # )
        # self.peg_com_quat = self.peg_quat  # always equal

        # peg_velocities = self.pegs.get_velocities(clone=False)
        # self.peg_linvel = peg_velocities[:, 0:3]        # linear velocities
        # self.peg_angvel = peg_velocities[:, 3:6]        # angular velocities

        # ?? needed for my task? (so far only copied and adjusted from nut_bolt_place env)
        # self.peg_com_linvel = self.peg_linvel + torch.cross(        # torch.cross(input, other, dim=None, *, out=None) → Tensor 
        #                                                             # Returns the cross product of vectors in dimension dim of input and other.
            
        #     self.peg_angvel,                                        # input
        #     (self.peg_com_pos - self.peg_pos),                      # other
        #     dim=1
        # )
        # self.peg_com_angvel = self.peg_angvel  # always equal

        # self.peg_force = self.pegs.get_net_contact_forces(clone=False)

        # Hole tensors
        self.hole_pos, self.hole_quat = self.holes.get_world_poses(clone=False)
        self.hole_pos -= self.env_pos

        # self.hole_force = self.holes.get_net_contact_forces(clone=False)
