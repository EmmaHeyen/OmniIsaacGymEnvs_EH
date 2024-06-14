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

# Factory: Class for snap-fit task.

# Inherits the snap-fit envorionment class and the abstract task class.

# Execution: 

# Set PYTHON_PATH: 
# For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh
# For Windows: doskey PYTHON_PATH=C:\Users\...\isaac_sim-2023.1.1\python.bat $* 

# START TRAINING:
# cd OmniIsaacGymEnvs/omniisaacgymenvs
# PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskSnapFit_eh

# TENSORBOARD:
# PYTHON_PATH -m tensorboard.main --logdir runs/.../summaries (run from cd OmniIsaacGymEnvs/omniisaacgymenvs) (JEWEILS ANPASSEN)
# 
# RUNNING: 
# PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskSnapFit_eh test=True checkpoint=runs/.../nn/FactoryTaskSnapFit_eh.pth  (JEWEILS ANPASSEN)
# 

# TRAINING FROM CHECKPOINT:
# PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskSnapFit_eh checkpoint=runs/.../nn/FactoryTaskSnapFit_eh.pth (JEWEILS ANPASSEN)
 
# EXTENSION WORKFLOW:
# windows: C:/Users/emmah/AppData/Local/ov/pkg/isaac_sim-2023.1.1/isaac-sim.gym.bat --ext-folder "C:\Users\emmah"
# linux: /home/mnn_eh/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.gym.sh --ext-folder /home/mnn_eh


import asyncio
import hydra
import math
import omegaconf
import torch
from typing import Tuple
import numpy as np

import omni.kit
from omni.isaac.core.simulation_context import SimulationContext
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.transformations import tf_combine

import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_env_snap_fit_eh import FactoryEnvSnapFit_eh 
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from omni.physx.scripts import physicsUtils, utils
import omni.isaac.core.utils.prims as prims_utils


class FactoryTaskSnapFit_eh(FactoryEnvSnapFit_eh, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """Initialize environment superclass. Initialize instance variables."""
        # print("init")
        super().__init__(name, sim_config, env)

        self._get_task_yaml_params()
        self.epoch=0

    def _get_task_yaml_params(self) -> None:
        # print("_get_task_yaml_params")
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_snap_fit_eh.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_snap_fit = hydra.compose(config_name=asset_info_path)
        self.asset_info_snap_fit = self.asset_info_snap_fit[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting
        
        ppo_path = "train/FactoryTaskSnapFit_ehPPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def post_reset(self) -> None:
        """Reset the world. Called only once, before simulation begins."""

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        self.acquire_base_tensors()
        self._acquire_task_tensors()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Reset all envs
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        asyncio.ensure_future(
            self.reset_idx_async(indices, randomize_gripper_pose=False)
        )

    def _acquire_task_tensors(self) -> None:
        """Acquire tensors."""
      
        self.male_base_pos_local = self.male_heights_total * torch.tensor(    
            [0.0, 0.0, 1.0], device=self.device 
        ).repeat((self.num_envs, 1))-self.male_heights_base*torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))*0.5
       
        female_heights = self.female_heights 
        self.female_middle_point = female_heights * torch.tensor(  
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))*0.5

        self.male_kp_middle_points=self.male_keypoint_middle_points * torch.tensor( # highest position of female part
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))

        '''
        num_keypoints: 4  # number of keypoints used in reward
        keypoint_scale: 0.5  # length of line of keypoints
        '''

        self.keypoint_offsets_1,self.keypoint_offsets_2,self.keypoint_offsets_female_1,self.keypoint_offsets_female_2 = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints)

        self.keypoint_offsets_1[:,2] *= self.cfg_task.rl.keypoint_scale 
        self.keypoint_offsets_2[:,2] *= self.cfg_task.rl.keypoint_scale 
        self.keypoint_offsets_female_1[:,2] *= self.cfg_task.rl.keypoint_scale 
        self.keypoint_offsets_female_2[:,2] *= self.cfg_task.rl.keypoint_scale 

        self.keypoints_male_1 = torch.zeros(  # tensor of zeros of size (num_envs, num_keypoints, 3)
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3), 
            dtype=torch.float32,
            device=self.device,
        )

        self.keypoints_male_2 = torch.zeros_like(self.keypoints_male_1, device=self.device) 
        self.keypoints_female_1 = torch.zeros_like(self.keypoints_male_1, device=self.device) 
        self.keypoints_female_2 = torch.zeros_like(self.keypoints_male_1, device=self.device) # tensor of zeros of same size as self.keypoints_male
        self.keypoints_checkpoint_1 = torch.zeros_like(self.keypoints_male_1, device=self.device) 
        self.keypoints_checkpoint_2 = torch.zeros_like(self.keypoints_male_1, device=self.device)
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

    def pre_physics_step(self, actions) -> None:
        """Reset environmentsand apply actions from the policy. Simulation step called after this method."""
        # print("pre_physics_step")
        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids, randomize_gripper_pose=True)
            
        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids, randomize_gripper_pose=True)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=0.0,
            do_scale=True,
        )

    def reset_idx(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""
        # print("reset_idx")
        self._reset_franka(env_ids)
       
        self._reset_object(env_ids)

        # Close gripper onto male
        self.disable_gravity()  # to prevent male from falling
        self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)    # num_gripper_close_sim_steps= number of timesteps to reserve for closing gripper onto male part during each reset (40)
        self.enable_gravity(gravity_mag=self.cfg_task.sim.gravity_mag)

        if randomize_gripper_pose:
            self._randomize_gripper_pose(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps         # num_gripper_move_sim_steps= number of timesteps to reserve for moving gripper before first step of episode (40)
            )

        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        # Close gripper onto male
        self.disable_gravity()  # to prevent male from falling
        await self._close_gripper_async(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self.enable_gravity(gravity_mag=self.cfg_task.sim.gravity_mag)

        if randomize_gripper_pose:
            await self._randomize_gripper_pose_async(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
            )

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids) -> None:
        """Reset DOF states and DOF targets of Franka."""
        gripper_buffer=1.1  # buffer on gripper DOF pos to prevent initial contact 

        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos, # franka_arm_initial_dof_pos: [0.00871, -0.10368, -0.00794, -1.49139, -0.00083,  1.38774,  0.7861]
                    device=self.device,
                ).repeat((len(env_ids), 1)),
                (self.male_thicknesses * 0.5)                                 # finger pos
                * gripper_buffer,  # buffer on gripper DOF pos to prevent initial contact 
                (self.male_thicknesses * 0.5) * gripper_buffer                # finger pos
                
            ),  # buffer on gripper DOF pos to prevent initial contact 
            dim=-1,
        )  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)
        # print("reset_franka ende")

    def _reset_object(self, env_ids) -> None:
        """Reset to the root states of male and female part."""

        # Randomize root state of male within gripper
        self.male_pos_base[env_ids, 0] = 0.0 # x-coordinate of position
        self.male_pos_base[env_ids, 1] = 0.0 # y-coordinate of position

        self.male_pos_armLeft[env_ids, 0] = self.male_pos_base[env_ids, 0] - 0.023
        self.male_pos_armLeft[env_ids, 1] = self.male_pos_base[env_ids, 1]       
        self.male_pos_armRight[env_ids, 0] = self.male_pos_base[env_ids, 0] + 0.023
        self.male_pos_armRight[env_ids, 1] = self.male_pos_base[env_ids, 1]      

        fingertip_midpoint_pos_reset = 0.58781  

        male_base_pos_local = self.male_heights_total.squeeze(-1) - 0.5*self.male_heights_base.squeeze(-1) 

        self.male_pos_base[env_ids, 2] = fingertip_midpoint_pos_reset + 0.01            # reset positions according to relative position of arms and base. has to be adjusted geometry is changed
        self.male_pos_armLeft[env_ids, 2] = self.male_pos_base[env_ids, 2] - 0.01       
        self.male_pos_armRight[env_ids, 2] = self.male_pos_base[env_ids,2] - 0.01       
        
        male_noise_pos_in_gripper = 2 * ( 
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        male_noise_pos_in_gripper = male_noise_pos_in_gripper @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.male_noise_pos_in_gripper, device=self.device
            )
        )

        # To add noise to male pos in gripper:
        # self.male_pos[env_ids, :] += male_noise_pos_in_gripper[env_ids]
                      
        male_rot_euler = torch.tensor(
            [0.0, 0.0, math.pi * 0.5], device=self.device
        ).repeat(len(env_ids), 1)
        male_noise_rot_in_gripper = 2 * (
            torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        male_noise_rot_in_gripper *= self.cfg_task.randomize.male_noise_rot_in_gripper
        male_rot_euler[:, 2] += male_noise_rot_in_gripper  
        male_rot_quat = torch_utils.quat_from_euler_xyz(
            male_rot_euler[:, 0], male_rot_euler[:, 1], male_rot_euler[:, 2]
        )


        self.male_linvel_base[env_ids, :] = 0.0
        self.male_angvel_base[env_ids, :] = 0.0

        self.male_linvel_armRight[env_ids, :] = self.male_linvel_base[env_ids, :] 
        self.male_angvel_armRight[env_ids, :] = self.male_angvel_base[env_ids, :] 
        self.male_linvel_armLeft[env_ids, :] = self.male_linvel_base[env_ids, :] 
        self.male_angvel_armLeft[env_ids, :] = self.male_angvel_base[env_ids, :] 

        indices = env_ids.to(dtype=torch.int32)
              
        self.male_quat_base=self.identity_quat
        self.male_quat_armleft=self.identity_quat
        self.male_quat_armRight=self.identity_quat
        
        self.males_base.set_world_poses(
            self.male_pos_base[env_ids] + self.env_pos[env_ids],
            self.male_quat_base[env_ids],
            indices,
        )
        self.males_base.set_velocities(
            torch.cat((self.male_linvel_base[env_ids], self.male_angvel_base[env_ids]), dim=1),
            indices,
        )

        self.males_armRight.set_world_poses(
            self.male_pos_armRight[env_ids] + self.env_pos[env_ids],
            self.male_quat_armRight[env_ids],
            indices,
        )

        self.males_armRight.set_velocities(
            torch.cat((self.male_linvel_armRight[env_ids], self.male_angvel_armRight[env_ids]), dim=1),
            indices,
        )

        self.males_armLeft.set_world_poses(
            self.male_pos_armLeft[env_ids] + self.env_pos[env_ids],
            self.male_quat_armleft[env_ids],
            indices,
        )

        self.males_armLeft.set_velocities(
            torch.cat((self.male_linvel_armLeft[env_ids], self.male_angvel_armLeft[env_ids]), dim=1),
            indices,
        )

        # Randomize root state of female
        female_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        female_noise_xy = female_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.female_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.female_pos[env_ids, 0] = ( # x
            self.cfg_task.randomize.female_pos_xy_initial[0] # + female_noise_xy[env_ids, 0]
        )
        self.female_pos[env_ids, 1] = ( # y
            self.cfg_task.randomize.female_pos_xy_initial[1] # + female_noise_xy[env_ids, 1]
        )

        self.female_pos[env_ids, 2] = self.cfg_base.env.table_height # z # table_height=0.4 (from cfg->task->FactoryBase.yaml)

        self.female_quat[env_ids, :] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(len(env_ids), 1)

        indices = env_ids.to(dtype=torch.int32)
        self.females.set_world_poses(
            self.female_pos[env_ids] + self.env_pos[env_ids],
            self.female_quat[env_ids],
            indices,
        )
      

    def _reset_buffers(self, env_ids) -> None:
        # print("_reset_buffers")
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.force_buf[env_ids] = 0

        print("checkpoint_buf_before_resetting: ",self.checkpoint_buf)
        self.checkpoint_buf[env_ids] = 0
        self.checkpoint_buf_keypoint_dist[env_ids] = 0
        self.keypoint_dist_buf[env_ids]=0

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale 
    ) -> None:
        """Apply actions from policy as position/rotation/force/torque targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.fingertip_midpoint_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        if self.cfg_ctrl["do_force_ctrl"]: # false for impedance controller
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.force_action_scale, device=self.device
                    )
                )

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.torque_action_scale, device=self.device
                    )
                )

            self.ctrl_target_fingertip_contact_wrench = torch.cat(
                (force_actions, torque_actions), dim=-1
            )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def post_physics_step(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.refresh_base_tensors()     
            self.refresh_env_tensors()       
            self._refresh_task_tensors()    
            self.get_observations()         
            self.calculate_metrics()        
            self.get_extras()               

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self) -> None:
        """Refresh tensors."""
        # Compute pos of keypoints on gripper, male, and female in world frame

        self.positions_checkpoint = self.female_pos.clone()
        self.positions_checkpoint[:,2] += self.checkpoint_z_pos.squeeze(dim=1)
        
        # KEYPOINTS SHIFTED BY 2CM in X-DIRECTION
        self.num_keypoints=self.cfg_task.rl.num_keypoints
        for idx, keypoint_offset in enumerate(self.keypoint_offsets_1): # keypoints are placed in even distance in a line in z-direction       
            self.keypoints_male_1[:, idx] = tf_combine(                 # shape keypoints_male_1: torch.Size([num_envs, 4, 3])
                self.male_quat_base,
                self.male_pos_base,
                self.identity_quat,
                (keypoint_offset - self.male_kp_middle_points), 
            )[1]
            self.keypoints_female_1[:, idx] = tf_combine( 
                self.female_quat,
                self.female_pos,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
            )[1]
            self.keypoints_checkpoint_1[:, idx] = tf_combine( 
                self.female_quat,
                self.positions_checkpoint,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
                
            )[1]

            
        for idx, keypoint_offset in enumerate(self.keypoint_offsets_2): # keypoints are placed in even distance in a line in z-direction 
            self.keypoints_male_2[:, idx] = tf_combine(                 # shape keypoints_male_2: torch.Size([num_envs, 4, 3])
                self.male_quat_base,
                self.male_pos_base,
                self.identity_quat,
                (keypoint_offset - self.male_kp_middle_points), 
            )[1]
            self.keypoints_female_2[:, idx] = tf_combine( 
                self.female_quat,
                self.female_pos,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
            )[1]
            self.keypoints_checkpoint_2[:, idx] = tf_combine( 
                self.female_quat,
                self.positions_checkpoint,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
            )[1]
        
        # print("shape keypoints: ", self.keypoints_female_2.shape) # torch.Size([num_envs, 4, 3])

    def get_observations(self) -> dict:
        """Compute observations."""
        # print("get_observations")
        # Shallow copies of tensors
        metadata = self.frankas._metadata
        # print("metadata: ", metadata)
        joint_indices = 1 + torch.Tensor([
        metadata.joint_indices["panda_finger_joint1"],
        metadata.joint_indices["panda_finger_joint2"],
        ])

        delta_quat=torch_utils.quat_mul(self.female_quat,self.quat_conjugate(self.male_quat_base))


        self.joint_forces=self.frankas.get_measured_joint_forces(joint_indices=joint_indices)

        self.joint_force_panda_finger_joint1=torch.split(self.joint_forces, 1, dim=1)[0].squeeze(1)
        self.joint_force_panda_finger_joint2=torch.split(self.joint_forces, 1, dim=1)[1].squeeze(1)
               

        self.joint_efforts_fingers=self.frankas.get_measured_joint_efforts(joint_indices=torch.Tensor([7, 8]))
        self.joint_efforts_except_fingers=self.frankas.get_measured_joint_efforts(joint_indices=torch.Tensor([0,1,2,3,4,5,6]))


        obs_tensors = [                         # when observations are changed, also change num_observations in FactoryTaskSnapFit_ehPPO.yaml!
            self.fingertip_midpoint_pos,        # dimension: 3
            self.fingertip_midpoint_quat,       #4
            self.fingertip_midpoint_linvel,     #3
            self.fingertip_midpoint_angvel,     #3
            self.male_pos_base,                 #3
            self.male_quat_base,                #4
            self.female_pos,                    #3
            self.female_quat,                   #4
            self.joint_efforts_except_fingers,  #7                       
        
        ]

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        observations = {self.frankas.name: {"obs_buf": self.obs_buf}}

        return observations

    def calculate_metrics(self) -> None:
        """Update reset and reward buffers."""
        # print("calculate_metrics")
        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self) -> None:
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def _update_rew_buf(self) -> None:

        """Compute reward at the current timestep."""

        # KEYPOINT REWARD
        keypoint_reward = -self._get_keypoint_dist()
        keypoint_reward_scaled = -self._get_keypoint_dist_scaled()
      
        # FORCE reward
        exceeded_force_max = self._get_contact_forces()
        exceeded_keypoint_dist_thresh = keypoint_reward >= self.cfg_task.rl.dist_thresh_for_max_force
        force_reward=exceeded_force_max*exceeded_keypoint_dist_thresh
        self.force_buf+=force_reward
            
        # CHECKPOINT reward
        checkpoint_reached = self._check_reached_checkpoint() # torch.Size([16])
        checkpoint_reached_keypoint_dist = self._check_reached_checkpoint_keypoint_dist()
        
        # reset first_time_checkpoint_reached:
        first_time_checkpoint_reached = torch.zeros_like(self.checkpoint_buf, dtype=torch.bool)
        first_time_checkpoint_reached_keypoint_dist = torch.zeros_like(self.checkpoint_buf_keypoint_dist, dtype=torch.bool)
        
        # check if checkpoint has been reached before (in current epoch):
        first_time_checkpoint_reached = (self.checkpoint_buf == 1) & (checkpoint_reached==1)
        env_checkpoint_reached_first_time = torch.nonzero(first_time_checkpoint_reached).squeeze(dim=1)
        # print("Checkpoint reached in environments:", env_checkpoint_reached_first_time.tolist())
       
        # check if checkpoint has been reached before: 
        first_time_checkpoint_reached = (self.checkpoint_buf == 1) & (checkpoint_reached==1)
        first_time_checkpoint_reached_keypoint_dist = (self.checkpoint_buf_keypoint_dist == 1) & (checkpoint_reached_keypoint_dist==1)
        
        # compute checkpoint reward only if it has been reached for the first time
        if self.cfg_task.rl.checkpoint_checking_method=="keypoint_dist": 
            checkpoint_reward = checkpoint_reached_keypoint_dist * first_time_checkpoint_reached_keypoint_dist
       
        elif self.cfg_task.rl.checkpoint_checking_method=="point_dist":
            checkpoint_reward = checkpoint_reached * first_time_checkpoint_reached

        # TOTAL REWARD (define which hypothesis should be used in FactoryTaskSnapFit_eh.yaml (at rl/ansatz))
        if self.cfg_task.rl.ansatz == "A":
             self.rew_buf[:] = (
                keypoint_reward * self.cfg_task.rl.keypoint_reward_scale  
                - force_reward * self.cfg_task.rl.max_force_scale
        )

        if self.cfg_task.rl.ansatz == "B":
            self.rew_buf[:] = (
                keypoint_reward_scaled * self.cfg_task.rl.keypoint_reward_scale    
                + checkpoint_reward * self.cfg_task.rl.checkpoint_reward_scale        
        )

        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            # Check if male is close to female (=success)
            is_male_close_to_female = self._check_male_close_to_female() 
            print("_check_male_close_to_female: ",is_male_close_to_female)
            self.rew_buf[:] += is_male_close_to_female * self.cfg_task.rl.success_bonus # if close to female part: --> successbonus*1 else successbonus*0; sucess_bonus defined in cfg-task-yaml-file (currently =0)
            self.extras["successes"] = torch.mean(is_male_close_to_female.float())
            success_rate = torch.sum(is_male_close_to_female)/self.num_envs
            print("success rate: ",success_rate.item())
            checkpoint_reached_in_epoch = (self.checkpoint_buf >= 1)

            # add success rate and rate of reached checkpoints to extras, so they can be displayed on tensorboard (can be expanded easily by simply adding new key-value-pair to extras-dict)
            self.extras["successes"] = torch.mean(is_male_close_to_female.float())
            self.extras["checkpoint_reached"] = torch.mean(checkpoint_reached_in_epoch.float())

    def _get_keypoint_offsets(self, num_keypoints) -> torch.Tensor:
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        Last column contains values ranging from -0.5 to 0.5 in a linearly spaced manner. The first two columns are filled with zeros. """
        
        keypoint_offsets_1=torch.zeros((num_keypoints, 3), device=self.device) # tensor for only one axis of keypoints
        keypoint_offsets_2=torch.zeros((num_keypoints, 3), device=self.device) # tensor for only one axis of keypoints
        keypoint_offsets_female_1=torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets_female_2=torch.zeros((num_keypoints, 3), device=self.device)

        keypoint_offsets_1[:, -1]=( # axis 1
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device)-0.5
        )
        keypoint_offsets_2[:, -1]=( # axis 2
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device)-0.5
        )
        keypoint_offsets_female_1[:, -1]=( # axis 1
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device)-0.5
        )
        keypoint_offsets_female_2[:, -1]=( # axis 1
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device)-0.5
        )

        keypoint_offsets_1[:, 0]= -0.02 # move keypoint axis 2 units in  negative x-direction 
        # print("keypoint_offsets_1: ",keypoint_offsets_1)
        keypoint_offsets_2[:, 0]= 0.02 # move keypoint axis 2 units in positive x-direction 
        # print("keypoint_offsets_2: ",keypoint_offsets_2)

        keypoint_offsets_female_1[:, 0]= -0.023
        keypoint_offsets_female_2[:, 0]= 0.023

        return keypoint_offsets_1, keypoint_offsets_2, keypoint_offsets_female_1, keypoint_offsets_female_2

    def _get_keypoint_dist(self) -> torch.Tensor:
        """Get keypoint distance between male and female parts.
            combinations (male_keypoints,female_keypoints): 
            (1,1) and (2,2), (1,2) and (2,1)
            a and b, c and d
        """
        dist_1_1 = (self.keypoints_female_1 - self.keypoints_male_1)
        dist_2_2 = (self.keypoints_female_2 - self.keypoints_male_2)
        dist_2_1 = (self.keypoints_female_2 - self.keypoints_male_1)
        dist_1_2 = (self.keypoints_female_1 - self.keypoints_male_2)

     
        # calc euclidean norm for each keypoint and then sum up the values along one keypoint axis  
        keypoint_dist_A1 = torch.sum(torch.norm(dist_1_1,p=2, dim=-1), dim=-1)
        keypoint_dist_A2 = torch.sum(torch.norm(dist_2_2,p=2, dim=-1), dim=-1)
        keypoint_dist_B1 = torch.sum(torch.norm(dist_2_1,p=2, dim=-1), dim=-1)
        keypoint_dist_B2 = torch.sum(torch.norm(dist_1_2,p=2, dim=-1), dim=-1)

        keypoint_dist_A=torch.add(keypoint_dist_A1,keypoint_dist_A2)
        keypoint_dist_B=torch.add(keypoint_dist_B1,keypoint_dist_B2)

        is_first_step = self.progress_buf[0] == 1

        # decide which combination should be used
        # is only updated for first step of epoch, not for any other steps.--> min kp-dist is chosen at beginning of an epoch and not for each step
        if is_first_step:
            self.min_dist_index = None 

            # find out which distance is smaller
            # 0 for A, 1 for B
            self.min_dist_index = torch.argmin(torch.stack([keypoint_dist_A, keypoint_dist_B], dim=1), dim=1)
           

        # Depending on the chosen index, select the keypoint distance for reward calculation
        keypoint_dist_min = torch.where(self.min_dist_index == 0, keypoint_dist_A, keypoint_dist_B)

        return keypoint_dist_min
    
    def _get_keypoint_dist_scaled(self) -> torch.Tensor:
        """calculate the keypoint distance between male and female part."""

        self._get_keypoint_dist() # to get the current min_dist_index

        dist_1_1 = (self.keypoints_female_1 - self.keypoints_male_1)
        dist_2_2 = (self.keypoints_female_2 - self.keypoints_male_2)
        dist_2_1 = (self.keypoints_female_2 - self.keypoints_male_1)
        dist_1_2 = (self.keypoints_female_1 - self.keypoints_male_2)

        # scaling of the different axes according to FactoryTaskSnapFit_eh.yaml
        scaling_tensor = torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, 
                          self.cfg_task.rl.keypoint_reward_scale_y, 
                          self.cfg_task.rl.keypoint_reward_scale_z], device=self.device)
        
        for dist in (dist_1_1, dist_1_2, dist_2_1, dist_2_2): 
            dist *= scaling_tensor      

        # calc euclidean norm for each keypoint and then sum up the values along one keypoint axis  
        keypoint_dist_A1 = torch.sum(torch.norm(dist_1_1,p=2, dim=-1), dim=-1)
        keypoint_dist_A2 = torch.sum(torch.norm(dist_2_2,p=2, dim=-1), dim=-1)
        keypoint_dist_B1 = torch.sum(torch.norm(dist_2_1,p=2, dim=-1), dim=-1)
        keypoint_dist_B2 = torch.sum(torch.norm(dist_1_2,p=2, dim=-1), dim=-1)

         # print("keypoint_dist_A1 shape", keypoint_dist_A1.shape) # torch.size([num_envs])

        keypoint_dist_A=torch.add(keypoint_dist_A1,keypoint_dist_A2)
        keypoint_dist_B=torch.add(keypoint_dist_B1,keypoint_dist_B2)
 
        # Depending on the chosen min_dist_index, select the keypoint distance for reward calculation
        keypoint_dist_min = torch.where(self.min_dist_index == 0, keypoint_dist_A, keypoint_dist_B)

        return keypoint_dist_min


    def _get_keypoint_dist_checkpoint(self) -> torch.Tensor:
        """Get keypoint distance between male part and checkpoint.
        (not tested extensively. Could possibly support better positioning before final assembly movement in vertical axis. Currently point dist is used instead.)
        """

        self._get_keypoint_dist() # to get the current min_dist_index

        dist_1_1 = (self.keypoints_checkpoint_1 - self.keypoints_male_1)
        dist_2_2 = (self.keypoints_checkpoint_2 - self.keypoints_male_2)
        dist_2_1 = (self.keypoints_checkpoint_2 - self.keypoints_male_1)
        dist_1_2 = (self.keypoints_checkpoint_1 - self.keypoints_male_2)

        # scaling of the different axes according to task config file

        scaling_tensor = torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, 
                          self.cfg_task.rl.keypoint_reward_scale_y, 
                          self.cfg_task.rl.keypoint_reward_scale_z], device=self.device)
        
        for dist in (dist_1_1, dist_1_2, dist_2_1, dist_2_2): 
            dist *= scaling_tensor      

        # calc euclidean norm for each keypoint and then sum up the values along one keypoint axis  
        keypoint_dist_A1 = torch.sum(torch.norm(dist_1_1,p=2, dim=-1), dim=-1)

        # print("keypoint_dist_A1 shape", keypoint_dist_A1.shape) # torch.size([num_envs])

        keypoint_dist_A2 = torch.sum(torch.norm(dist_2_2,p=2, dim=-1), dim=-1)
        keypoint_dist_B1 = torch.sum(torch.norm(dist_2_1,p=2, dim=-1), dim=-1)
        keypoint_dist_B2 = torch.sum(torch.norm(dist_1_2,p=2, dim=-1), dim=-1)

        keypoint_dist_A=torch.add(keypoint_dist_A1,keypoint_dist_A2)
        keypoint_dist_B=torch.add(keypoint_dist_B1,keypoint_dist_B2)
 
        # Depending on the chosen min_dist_index, select the keypoint distance for reward calculation
        keypoint_dist_min = torch.where(self.min_dist_index == 0, keypoint_dist_A, keypoint_dist_B)

        return keypoint_dist_min

    def _randomize_gripper_pose(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""
        # Step once to update PhysX with new joint positions and velocities from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device # initial position of midpoint between fingertips above table
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        # add noise
        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device  
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rotation
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        #rotational noise
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=0.0,
                do_scale=False,

            )

            SimulationContext.step(self._env._world, render=False)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step once to update PhysX with new joint velocities
        SimulationContext.step(self._env._world, render=True)

    async def _randomize_gripper_pose_async(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""

        # Step once to update PhysX with new joint positions and velocities from reset_franka()
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=0.0,
                do_scale=False,
            )

            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step simulation once to update PhysX with new joint velocities
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()

    def _close_gripper(self, sim_steps) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps) -> None:
        """Move gripper fingers to specified DOF position using controller."""
        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion 

        # Step simulation
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, 
                do_scale=False
            )
            SimulationContext.step(self._env._world, render=True)

    async def _close_gripper_async(self, sim_steps) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        await self._move_gripper_to_dof_pos_async(
            gripper_dof_pos=0.0, sim_steps=sim_steps
        )

    async def _move_gripper_to_dof_pos_async(
        self, gripper_dof_pos, sim_steps
    ) -> None:
        """Move gripper fingers to specified DOF position using controller."""
        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion

        # Step simulation
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, 
                do_scale=False
            )
            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    def _check_male_close_to_female(self) -> torch.Tensor:
        """Check if male part is close to female part via keypoint distances."""

        keypoint_dist=self._get_keypoint_dist() # not the scaled one so that the threshold can be used universally even for changed scales

        is_male_close_to_female = keypoint_dist <= self.cfg_task.rl.close_error_thresh

        return is_male_close_to_female

    def _check_reached_checkpoint(self) -> torch.Tensor:
        """Check if male part is close to checkpoint."""

        self.positions_checkpoint = self.female_pos.clone()
        self.positions_checkpoint[:,2] += self.checkpoint_z_pos.squeeze(dim=1)

        checkpoint_dist = torch.norm(self.positions_checkpoint - self.male_pos_base, p=2, dim=-1) # euclidian norm

        is_close_to_checkpoint=torch.zeros_like(checkpoint_dist)

        for i in range(self._num_envs):
            is_close_to_checkpoint[i] = torch.where(
                checkpoint_dist[i] < self.cfg_task.rl.close_checkpoint_error_thresh,
                1,
                0,
            )
   
        self.checkpoint_buf += is_close_to_checkpoint

        return is_close_to_checkpoint
    
    def _check_reached_checkpoint_keypoint_dist(self) -> torch.Tensor:
        """Check if male part is close to checkpoint via keypoint distances. Choose if keypoint distance or point distance should be used in task-config-file."""

        keypoint_dist=self._get_keypoint_dist_checkpoint() # not scaled 

        is_male_close_to_checkpoint = keypoint_dist <= self.cfg_task.rl.close_checkpoint_error_thresh_keypoint_dist

        is_close_to_checkpoint=torch.zeros_like(keypoint_dist)

        for i in range(self._num_envs):
            is_close_to_checkpoint = torch.where(
                is_male_close_to_checkpoint,
                1,
                0,
            )
   
        self.checkpoint_buf_keypoint_dist += is_close_to_checkpoint

        return is_close_to_checkpoint
    
    def quat_conjugate(self,q):
        return torch.cat([q[:, :1], -q[:, 1:]], dim=1)
    
    def _get_contact_forces(self):

        contact_force_armRight = self.males_armRight.get_net_contact_forces()
        contact_force_armLeft = self.males_armLeft.get_net_contact_forces()

        contact_force_mean=(contact_force_armLeft[:,2]+contact_force_armRight[:,2])/2

        collision_detected = torch.abs(contact_force_mean) >= self.cfg_task.rl.max_force_thresh

        return collision_detected

       

