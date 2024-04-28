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

# """Factory: Class for snap-fit place task.

# Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with: 

# PYTHON_PATH: 
# For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh
# For Windows: doskey PYTHON_PATH=C:\Users\emmah\AppData\Local\ov\pkg\isaac_sim-2023.1.1\python.bat $* 

# cd OmniIsaacGymEnvs/omniisaacgymenvs
# PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskSnapFit_eh

# TENSORBOARD:
# PYTHON_PATH -m tensorboard.main --logdir runs/FactoryTaskSnapFit_eh_400epochs_0.1mass_0.3friction-both_0actionpenalty_withNoise/summaries (run from cd OmniIsaacGymEnvs/omniisaacgymenvs)
# 
# RUNNING: 
# PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskSnapFit_eh test=True num_envs=4 checkpoint=runs/FactoryTaskSnapFit_eh_400epochs_0.1mass_0.3friction-both_0.3actionpenalty_withNoise/nn/FactoryTaskSnapFit_eh.pth  (ANPASSEN)
# 
# muss num_envs an eine bestimmte stelle?

# TRAINING FROM CHECKPOINT:
# PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth


# EXTENSION WORKFLOW:
# windows: C:/Users/emmah/AppData/Local/ov/pkg/isaac_sim-2023.1.1/isaac-sim.gym.bat --ext-folder "C:\Users\emmah"
# linux: /home/mnn_eh/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.gym.sh --ext-folder /home/mnn_eh

# """

# """

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


class FactoryTaskSnapFit_eh(FactoryEnvSnapFit_eh, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """Initialize environment superclass. Initialize instance variables."""
        # print("init")
        super().__init__(name, sim_config, env)

        self._get_task_yaml_params()

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
        # print("post_reset")

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
        # print("_acquire_task_tensors")
        # Nut-bolt tensors
      
        self.male_base_pos_local = self.male_heights_total * torch.tensor(    # result: list of tensors like torch.tensor([0.0, 0.0, bolt_head_height]) --> nut_pos_base local is equal to [0,0,bolt_head_height] for each env.
            [0.0, 0.0, 1.0], device=self.device 
        ).repeat((self.num_envs, 1))-self.male_heights_base*torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))*0.5

       
        female_heights = self.female_heights 
        self.female_middle_point = female_heights * torch.tensor(  
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))*0.5

       
        self.male_kp_middle_points=self.male_keypoint_middle_points * torch.tensor( # highest position of female thingy
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))

        # try20240428
        self.male_kp_middle_points[:,2] += 0.01

        print("male_kp_middle_points:", self.male_kp_middle_points)

        # Keypoint tensors
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
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3), # adjusted for snap-fit: multiplied num_keypoints by 2 (number of axes)
            dtype=torch.float32,
            device=self.device,
        )

        self.keypoints_male_2 = torch.zeros_like(self.keypoints_male_1, device=self.device) 
        self.keypoints_female_1 = torch.zeros_like(self.keypoints_male_1, device=self.device) 
        self.keypoints_female_2 = torch.zeros_like(self.keypoints_male_1, device=self.device) # tensor of zeros of same size as self.keypoints_male

        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""
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
        # print("pre_physics_step ende")

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
        self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)    # num_gripper_close_sim_steps= number of timesteps to reserve for closing gripper onto nut during each reset (40)
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
        # print("_reset_franka")
        gripper_buffer=1.1  # buffer on gripper DOF pos to prevent initial contact (was 1.1 before)
        # print("gripper_buffer: ", gripper_buffer)

        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos, # franka_arm_initial_dof_pos: [0.00871, -0.10368, -0.00794, -1.49139, -0.00083,  1.38774,  0.7861]
                    device=self.device,
                ).repeat((len(env_ids), 1)),
                # (self.male_widths * 0.5)
                (self.male_thicknesses * 0.5)                                 # finger pos
                * gripper_buffer,  # buffer on gripper DOF pos to prevent initial contact (was 1.1 before)
                # (self.male_widths * 0.5) * 1.1,
                (self.male_thicknesses * 0.5) * gripper_buffer                # finger pos
                
            ),  # buffer on gripper DOF pos to prevent initial contact (was 1.1 before)
            dim=-1,
        )  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)
        # print("reset_franka ende")

    def _reset_object(self, env_ids) -> None:
        """Reset root states of nut and bolt."""
        # print("reset_object")
        # Randomize root state of nut within gripper
        self.male_pos_base[env_ids, 0] = 0.0 # x-coordinate of position?
        self.male_pos_base[env_ids, 1] = 0.0 # y-coordinate of position?

        # self.male_pos_armLeft[env_ids, 0] = self.male_pos_base[env_ids, 0] - 0.023
        # self.male_pos_armLeft[env_ids, 1] = self.male_pos_base[env_ids, 1]       # TODO right unit??
        # self.male_pos_armRight[env_ids, 0] = self.male_pos_base[env_ids, 0] + 0.023
        # self.male_pos_armRight[env_ids, 1] = self.male_pos_base[env_ids, 1]      # TODO: right unit??

        # TODO: check out if value for fingertip_midpoint_pos_reset has to be adjusted (was 0.58781 before) (franka_finger_length=0.053671, from factory_asset_info_franka_table)
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset # for  fingertip_midpoint_pos_reset = 0.28781 peg "jumped" ot of table?? (TODO)
        # print("fingertip_midpoint_pos_reset: ",fingertip_midpoint_pos_reset)

        male_base_pos_local = self.male_heights_total.squeeze(-1) - 0.5*self.male_heights_base.squeeze(-1) # try 3 # TODO: adjust base_pos to snap fit task

        self.male_pos_base[env_ids, 2] = fingertip_midpoint_pos_reset + 0.01 # TODO: adjust male z pos of male to snapfit task ? TODO: ist peg_base_local random gewählt -->irgendein Abstand der ungefähr passen könnte, damit abstand zwischen fingertip_midpoint_pos_reset und bolt
        

        # self.male_pos_armLeft[env_ids, 2] = self.male_pos_base[env_ids, 2] - 0.01       # TODO right unit?
        # self.male_pos_armRight[env_ids, 2] = self.male_pos_base[env_ids,2] - 0.01       # TODO: right unit?
        
        male_noise_pos_in_gripper = 2 * ( 
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        male_noise_pos_in_gripper = male_noise_pos_in_gripper @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.male_noise_pos_in_gripper, device=self.device
            )
        )
        # self.male_pos[env_ids, :] += male_noise_pos_in_gripper[env_ids]               # TODO: noise for base, armLeft and armRight

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
        # self.male_quat[env_ids, :] = male_rot_quat # TODO: noise

        self.male_linvel_base[env_ids, :] = 0.0
        self.male_angvel_base[env_ids, :] = 0.0

        self.male_linvel_armRight[env_ids, :] = self.male_linvel_base[env_ids, :] 
        self.male_angvel_armRight[env_ids, :] = self.male_angvel_base[env_ids, :] 
        self.male_linvel_armLeft[env_ids, :] = self.male_linvel_base[env_ids, :] 
        self.male_angvel_armLeft[env_ids, :] = self.male_angvel_base[env_ids, :] 


        indices = env_ids.to(dtype=torch.int32)
        # print("checkpoint1 ", env_ids)
               
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

        # try20240428: auskommentiert
        # self.males_armRight.set_world_poses(
        #     self.male_pos_armRight[env_ids] + self.env_pos[env_ids],
        #     self.male_quat_armRight[env_ids],
        #     indices,
        # )

        self.males_armRight.set_velocities(
            torch.cat((self.male_linvel_armRight[env_ids], self.male_angvel_armRight[env_ids]), dim=1),
            indices,
        )

        # try20240428: auskommentiert
        # self.males_armLeft.set_world_poses(
        #     self.male_pos_armLeft[env_ids] + self.env_pos[env_ids],
        #     self.male_quat_armleft[env_ids],
        #     indices,
        # )

        self.males_armLeft.set_velocities(
            torch.cat((self.male_linvel_armLeft[env_ids], self.male_angvel_armLeft[env_ids]), dim=1),
            indices,
        )


        # print("checkpoint2 ", env_ids)
        # print("self.male_pos[env_ids] + self.env_pos[env_ids]=",self.male_pos[env_ids] + self.env_pos[env_ids])

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
            self.cfg_task.randomize.female_pos_xy_initial[0] + female_noise_xy[env_ids, 0]
        )
        self.female_pos[env_ids, 1] = ( # y
            self.cfg_task.randomize.female_pos_xy_initial[1] + female_noise_xy[env_ids, 1]
        )

        # self.female_pos[env_ids, 0] = ( # x ###REMOVED NOISE FOR TESTING # TODO: noise
        #     self.cfg_task.randomize.female_pos_xy_initial[0]
        # )
        # self.female_pos[env_ids, 1] = ( # y ###REMOVED NOISE FOR TESTING  # TODO: noise
        #     self.cfg_task.randomize.female_pos_xy_initial[1]
        # )


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


        # self.positions_checkpoint = torch.zeros_like(self.female_pos)
        # self.positions_checkpoint[:,0] = self.female_pos[:,0]
        # self.positions_checkpoint[:,1] = self.female_pos[:,1]
        # self.positions_checkpoint[:,2] = self.female_pos[:,2]+ self.checkpoint_z_pos.squeeze(dim=1)

        # self.cubes.set_world_poses(
        #     self.positions_checkpoint + self.env_pos[env_ids], 
        #     self.female_quat[env_ids],
        #     indices,   
        # )


        # self.cubes.disable_rigid_body_physics()

        # print("reset_object ende")

    def _reset_buffers(self, env_ids) -> None:
        # print("_reset_buffers")
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        print("checkpoint_buf_before_resetting: ",self.checkpoint_buf)
        self.checkpoint_buf[env_ids] = 0

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale # do_scale:bool # immer false ausßer in pre_physics_step?
    ) -> None:
        """Apply actions from policy as position/rotation/force/torque targets."""
        # print("_apply_actions_as_ctrl_targets")

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
        # print("post_physics_step")
        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.refresh_base_tensors()     # ??? # from factory_schema_class_base.py-file?
            self.refresh_env_tensors()      # from env .py-file 
            self._refresh_task_tensors()    # from current .py-file
            self.get_observations()         # from current .py-file
            self.calculate_metrics()        # from current .py-file
            self.get_extras()               # ???

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self) -> None:
        """Refresh tensors."""
        # Compute pos of keypoints on gripper, nut, and bolt in world frame
        # print("_refresh_task_tensors")

        '''def tf_combine(q1, t1, q2, t2):
            return quat_mul(q1, q2), quat_apply(q1, t2) + t1

        -->combines two transformations, first applying the rotation of the first transformation to the translation of the second transformation ..
        .. and then adding it to the translation of the first transformation. ..
        ..The combined rotation is obtained by multiplying the quaternions of the two transformations.
        '''

        ''' KEYPOINTS SHIFTED BY 2CM in X-DIRECTION
        # self.num_keypoints=self.cfg_task.rl.num_keypoints
        # for idx, keypoint_offset in enumerate(self.keypoint_offsets_1): # keypoints are placed in even distance in a line in z-direction 
        #     # self.keypoints_male_1[:, idx] = tf_combine(                 # shape keypoints_male_1: torch.Size([128, 4, 3])
        #     #     self.male_quat_base,
        #     #     self.male_pos_base,                                       # MARKER 18!!
        #     #     self.identity_quat,
        #     #     (keypoint_offset - self.male_keypoint_middle_points), 
        #     # )[1]
        #     self.keypoints_male_1[:, idx] = tf_combine(                 # shape keypoints_male_1: torch.Size([128, 4, 3])
        #         self.male_quat_base,
        #         self.male_pos_base,
        #         self.identity_quat,
        #         (keypoint_offset - self.male_kp_middle_points), 
        #     )[1]
        #     self.keypoints_female_1[:, idx] = tf_combine( # TODO: adjust to malefemale snap fit part
        #         self.female_quat,
        #         self.female_pos,
        #         self.identity_quat,
        #         (keypoint_offset + self.female_middle_point),  
        #     )[1]

            
        # for idx, keypoint_offset in enumerate(self.keypoint_offsets_2): # keypoints are placed in even distance in a line in z-direction 
        #     # self.keypoints_male_2[:, idx] = tf_combine( # TODO: adjust to male snapfit part
        #     #     self.male_quat_base,
        #     #     self.male_pos_base,                               # MARKER 18!!
        #     #     self.identity_quat,
        #     #     (keypoint_offset - self.male_keypoint_middle_points), 
        #     # )[1]
        #     self.keypoints_male_2[:, idx] = tf_combine(                 # shape keypoints_male_1: torch.Size([128, 4, 3])
        #         self.male_quat_base,
        #         self.male_pos_base,
        #         self.identity_quat,
        #         (keypoint_offset - self.male_kp_middle_points), 
        #     )[1]
        #     self.keypoints_female_2[:, idx] = tf_combine( # TODO: adjust to female snap fit part
        #         self.female_quat,
        #         self.female_pos,
        #         self.identity_quat,
        #         (keypoint_offset + self.female_middle_point),  
        #     )[1]
        '''

        #### TRY20240428: put keypoint axes into the arms

        self.num_keypoints=self.cfg_task.rl.num_keypoints
        for index, keypoint_offset in enumerate(self.keypoint_offsets_1): # keypoints are placed in even distance in a line in z-direction 
            self.keypoints_male_1[:, index] = tf_combine(                 # shape keypoints_male_1: torch.Size([128, 4, 3])
                self.male_quat_armLeft,
                self.male_pos_armLeft,
                self.identity_quat,
                (keypoint_offset - self.male_kp_middle_points), 
            )[1]

        for index, keypoint_offset in enumerate(self.keypoint_offsets_female_1):    
            self.keypoints_female_1[:, index] = tf_combine( # TODO: adjust to malefemale snap fit part
                self.female_quat,
                self.female_pos,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
            )[1]
           
            
        for idx, keypoint_offset in enumerate(self.keypoint_offsets_2): # keypoints are placed in even distance in a line in z-direction 
            self.keypoints_male_2[:, index] = tf_combine(                 # shape keypoints_male_1: torch.Size([num_envs, 4, 3])
                self.male_quat_armRight,
                self.male_pos_armRight,
                self.identity_quat,
                (keypoint_offset - self.male_kp_middle_points), 
            )[1]

        for idx, keypoint_offset in enumerate(self.keypoint_offsets_female_2):
            self.keypoints_female_2[:, index] = tf_combine( 
                self.female_quat,
                self.female_pos,
                self.identity_quat,
                (keypoint_offset + self.female_middle_point),  
            )[1]

        # print("self.keypoint_offsets_female_1:", self.keypoint_offsets_female_1)
        # print("self.keypoint_offsets_female_2:", self.keypoint_offsets_female_2)
        # print("keypoints_female_1: ",self.keypoints_female_1)
        # print("keypoints_female_2: ", self.keypoints_female_2)
        # print("female_pos: ", self.female_pos)
        # print("keypoints_male_1 (left arm): ",self.keypoints_male_1)
        # print("male_pos_armLeft: ",self.male_pos_armLeft)
        # print("keypoints_male_2 (right arms): ",self.keypoints_male_2)
        # print("male_pos_armRight: ",self.male_pos_armRight)
        # print("male_pos_base: ",self.male_pos_base)
        
      
        


        # print("shape keypoint_female_2: ", self.keypoints_female_2.shape) # torch.Size([128, 4, 3])

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
               

        self.joint_efforts=self.frankas.get_measured_joint_efforts(joint_indices=torch.Tensor([7, 8]))

        # print("self.joint_efforts:  ", self.joint_efforts)
        # print("self.joint_forces: ", self.joint_forces)

        # print("self.joint_efforts.shape: ", self.joint_efforts.shape)
        # print("self.joint_forces.shape: ", self.joint_forces.shape)

        # print("female_pos_shape: ", self.female_pos.shape)

        obs_tensors = [ # WENN OBSERVATIONS ANGEPASST WERDEN MÜSSEN AUCH IN PPO.YAML DIE NUM_OBSERVATIONS ANGEPASST WERDEN!
            self.fingertip_midpoint_pos,        #3
            self.fingertip_midpoint_quat,       #4
            self.fingertip_midpoint_linvel,     #3
            self.fingertip_midpoint_angvel,     #3
            self.male_pos_base,                 #3
            self.male_quat_base,              #4
            self.female_pos,                    #3
            self.female_quat,                   #4
            # self.female_pos-self.male_pos_base, #3
            # delta_quat,                 #4
            # self._get_keypoint_dist().unsqueeze(1),          #1
            # self.joint_efforts,                 #2
        
        ]

        # if self.cfg_task.rl.add_obs_female_tip_pos: # --> edit in task config file (add_obs_female_tip_pos is boolean value) # add observation of bolt tip position
        #     obs_tensors += [self.female_tip_pos_local]

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
        # print("_update_reset_buf")
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def _update_rew_buf(self) -> None:
        """Compute reward at current timestep."""
        # print("_update_rew_buf")
        keypoint_reward = -self._get_keypoint_dist()
        # print("keypoint_reward: ", keypoint_reward)

        # self._get_keypoint_dist()
        # keypoint_reward = -self.keypoint_dist_min
        # print("keypoint_dist_min: ",self.keypoint_dist_min)

        # keypoint_reward = -torch.tensor(
        #     [0.0, 0.0, 1.0], device=self.device)

        checkpoint_reached = self._check_reached_checkpoint() # torch.Size([16])
        # print("checkpoint_reward: ", checkpoint_reward)
        # print("checkpoint_reward_shape: ", checkpoint_reward.shape)
        
        # reset first_time_checkpoint_reached
        first_time_checkpoint_reached = torch.zeros_like(self.checkpoint_buf, dtype=torch.bool)
        # first_time_checkpoint_reached.zero()

        # check if checkpoint has been reached before: 
        first_time_checkpoint_reached = (self.checkpoint_buf == 1) & (checkpoint_reached==1)
        # print("first_time_checkpoint_reached: ", first_time_checkpoint_reached)
        # print("checkpoint_buf: ", self.checkpoint_buf)

        env_checkpoint_reached_first_time = torch.nonzero(first_time_checkpoint_reached).squeeze(dim=1)
        # print("Checkpoint reached in environments:", env_checkpoint_reached_first_time.tolist())
        
        # compute checkpoint reward only if it has been reached for the first time
        checkpoint_reward = checkpoint_reached * first_time_checkpoint_reached # bisschen unnötig, weil habe ich ja oben schongechekct. Ginge wahrscheinlich auch mit checkpoint_reward=first_time_checkpoint_reached

        # if first_time_checkpoint_reached:
        #     self.extras["checkpt_reached_1st_time"] = torch.mean(first_time_checkpoint_reached.float()) # RuntimeError: Boolean value of Tensor with more than one value is ambiguous

        
        action_penalty = (
            torch.norm(self.actions, p=2, dim=-1)
            * self.cfg_task.rl.action_penalty_scale
        )
        # print("shape keypoint_reward. ", keypoint_reward.shape)

        '''
        default cfg values: 
        keypoint_reward_scale: 1.0  # scale on keypoint-based reward
        action_penalty_scale: 0.0  # scale on action penalty
        '''

        # keypoint_reward = keypoint_reward * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device)

        # self.rew_buf[:] = (
        #     keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
        #     - action_penalty * self.cfg_task.rl.action_penalty_scale # muss das so? hier wird zwei mal mit der penalty_scale multipliziert (ist in nut_bolt_place task auch so)          
        # )


        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale # muss das so? hier wird zwei mal mit der penalty_scale multipliziert (ist in nut_bolt_place task auch so)          
            + checkpoint_reward * self.cfg_task.rl.checkpoint_reward_scale        
        )

        # print("reward_calculations:/n")
        # print(keypoint_reward, "*" , self.cfg_task.rl.keypoint_reward_scale , "/n")
        # print("-" , action_penalty , "*" , self.cfg_task.rl.action_penalty_scale ,"/n")
        # print("+" , checkpoint_reward, "*" ,self.cfg_task.rl.checkpoint_reward_scale , "/n")
        # print("=" , self.rew_buf[:])



        # keypoint_reward * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x,self.cfg_task.rl.keypoint_reward_scale_y,self.cfg_task.rl.keypoint_reward_scale_z])

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            # Check if nut is close enough to bolt
            is_male_close_to_female = self._check_male_close_to_female() # adjust threshold in task config file
            # print("_check_male_close_to_female: ",is_male_close_to_female)
            self.rew_buf[:] += is_male_close_to_female * self.cfg_task.rl.success_bonus # if close to bolt, --> successbonus*1 else successbonus*0; sucess_bonus defined in cfg-task-yaml-file (currently =0)
            self.extras["successes"] = torch.mean(is_male_close_to_female.float())


    def _get_keypoint_offsets(self, num_keypoints) -> torch.Tensor:
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        Last column contains values ranging from -0.5 to 0.5 in a linearly spaced manner. The first two columns are filled with zeros. """
        # print("_get_keypoint_offsets")
        # keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device) 
        # keypoint_offsets[:, -1] = ( # 
        #     torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5 #(try 06/03/2024: commented out the -0.5; went back on 07/03/2027 but with middle keypoint on topsurface of hole)
        # ) 

        # keypoint_offsets for snap-fit task: two axes right and left along y axis 
        # keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device) # tensor for only one axis of keypoints
        
        # print("tensor keypoint_offsets *from*: ",num_keypoints*(i-1))
        # print("tensor keypoint_offsets *to*: ",num_keypoints*i+1)
        
        # did concatenation work  properly? --> nope but i dont use the concatenation anymore
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



        #TRY20240428: # put axes in arms (offset in x direction=0!) and adjust female offsets with x  offset of 
        
        # keypoint_offsets_1[:, 0]= -0.02 # move keypoint axis 2 units in  negative x-direction # TODO: right units?
        # print("keypoint_offsets_1: ",keypoint_offsets_1)
        # keypoint_offsets_2[:, 0]= 0.02 # move keypoint axis 2 units in positive x-direction # TODO: right units? --> check by adding cube at 0.02 to left/right to see if unit is right
        # print("keypoint_offsets_2: ",keypoint_offsets_2)

        keypoint_offsets_female_1[:, 0]= -0.023
        keypoint_offsets_female_2[:, 0]= 0.023




        # print("keypoint_offset (concatenated): ",keypoint_offsets)

        # print("keypoint_offsets_1: ",keypoint_offsets_1)
        # print("keypoint_offsets_2: ",keypoint_offsets_2)
        

        return keypoint_offsets_1, keypoint_offsets_2, keypoint_offsets_female_1, keypoint_offsets_female_2

    def _get_keypoint_dist(self) -> torch.Tensor:
        """Get keypoint distance between nut and bolt."""
        # print("_get_keypoint_dist")

        # combinations (male_keypoints,female_kepoints): 
        # (1,1) and (2,2), (1,2) and (2,1)
        # a and b, c and d
        
        # keypoint_dist_A1 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_1, p=2, dim=-1), dim=-1 # frobenius/euclid norm
        # )
        # keypoint_dist_A2 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_2, p=2, dim=-1), dim=-1
        # )
        # keypoint_dist_B1 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_1, p=2, dim=-1), dim=-1
        # )
        # keypoint_dist_B2 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_2, p=2, dim=-1), dim=-1
        # )

        # keypoint_dist_A1 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_1, p=2, dim=-1) 
        #     * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),         
            
        #     dim=-1 # frobenius/euclid norm
        # )
        # keypoint_dist_A2 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_2, p=2, dim=-1),
        #     * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),         
        #     dim=-1
        # )


        # keypoint_dist_B1 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_1, p=2, dim=-1)
        #     * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),         
        #     dim=-1
        # )
        # keypoint_dist_B2 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_2, p=2, dim=-1), 
        #     * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),
        #     dim=-1
        # )

        # keypoint_dist_A1 = torch.sum(
        #     torch.norm(
        #         (self.keypoints_female_1 - self.keypoints_male_1)
        #         * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),
        #         p=2, dim=-1
        #     ), dim=-1
        # )

        # keypoint_dist_A2 = torch.sum(
        #     torch.norm(
        #         (self.keypoints_female_2 - self.keypoints_male_2)
        #         * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),
        #         p=2, dim=-1
        #     ), dim=-1
        # )

        # keypoint_dist_B1 = torch.sum(
        #     torch.norm(
        #         (self.keypoints_female_2 - self.keypoints_male_1)
        #         * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),
        #         p=2, dim=-1
        #     ), dim=-1
        # )

        # keypoint_dist_B2 = torch.sum(
        #     torch.norm(
        #         (self.keypoints_female_1 - self.keypoints_male_2)
        #         * torch.tensor([self.cfg_task.rl.keypoint_reward_scale_x, self.cfg_task.rl.keypoint_reward_scale_y, self.cfg_task.rl.keypoint_reward_scale_z], device=self.device),
        #         p=2, dim=-1
        #     ), dim=-1
        # )

        # print("keypoints_female_1.shape: ", self.keypoints_female_1.shape)

       






        dist_1_1 = (self.keypoints_female_1 - self.keypoints_male_1)
        dist_2_2 = (self.keypoints_female_2 - self.keypoints_male_2)
        dist_2_1 = (self.keypoints_female_2 - self.keypoints_male_1)
        dist_1_2 = (self.keypoints_female_1 - self.keypoints_male_2)



        # scaling of the different axes
        for dist in (dist_1_1, dist_2_2, dist_1_2, dist_2_1):
            dist[:,0] *= self.cfg_task.rl.keypoint_reward_scale_x
            dist[:,1] *= self.cfg_task.rl.keypoint_reward_scale_y 
            dist[:,2] *= self.cfg_task.rl.keypoint_reward_scale_z
        

           
           
           
            # dist[:,1]
            # dist[:,2]


        # self.keypoints_female_1[:, 0] / self.cfg_task.rl.keypoint_reward_scale_x
        # normalized_d_y = self.keypoints_female_1[:, 1] / self.cfg_task.rl.keypoint_reward_scale_y
        # normalized_d_z = self.keypoints_female_1[:, 2] / self.cfg_task.rl.keypoint_reward_scale_z


        keypoint_dist_A1 = torch.sum(
            torch.norm(
                dist_1_1,
                p=2, dim=-1
            ), dim=-1
        )

        keypoint_dist_A2 = torch.sum(
            torch.norm(
                dist_2_2,                
                p=2, dim=-1
            ), dim=-1
        )

        keypoint_dist_B1 = torch.sum(
            torch.norm(
                dist_2_1,
                p=2, dim=-1
            ), dim=-1
        )

        keypoint_dist_B2 = torch.sum(
            torch.norm(
                dist_1_2,
                p=2, dim=-1
            ), dim=-1
        )


        keypoint_dist_A=torch.add(keypoint_dist_A1,keypoint_dist_A2)
        keypoint_dist_B=torch.add(keypoint_dist_B1,keypoint_dist_B2)


        # # Stack the tensors along a new axis (axis=1)
        # key_point_dist_stacked = torch.stack([keypoint_dist_A, keypoint_dist_B], dim=1)

        # # Find the minimum value along the new axis (axis=1)
        # keypoint_dist_min, _ = torch.min(key_point_dist_stacked, dim=1)


        is_first_step = self.progress_buf[0] == 1

        # if is_first_step:
        #     # reset self.keypoint_dist_min
        #     self.keypoint_dist_min=torch.zeros_like(keypoint_dist_min)
        #     self.keypoint_dist_min += keypoint_dist_min

        if is_first_step:
            self.chosen_keypoint_index = None 

            # Use torch.argmin to find which distance is smaller
            # 0 for A, 1 for B
            self.min_dist_index = torch.argmin(torch.stack([keypoint_dist_A, keypoint_dist_B], dim=1), dim=1)
            # print("self.min_dist_index:", self.min_dist_index)

        # Depending on the chosen index, select the keypoint distance for reward calculation
        keypoint_dist_min = torch.where(self.min_dist_index == 0, keypoint_dist_A, keypoint_dist_B)

        # Reset logic for the flag at the start of a new episode
     

        # is only updated for first step of epoch, not for any other steps.--> min kp-dist is chosen at beginning of an epoch and not for each step
            
        # print("key_point_dist_stacked: ",key_point_dist_stacked)
        # print("keypoint_dist_min_shape: ",keypoint_dist_min.shape)    # torch.Size([num_envs])   

        return keypoint_dist_min


    def _randomize_gripper_pose(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""
        # print("_randomize_gripper_pose")
        # Step once to update PhysX with new joint positions and velocities from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device # fingertip_midpoint_pos_initial: [0.0, 0.0, 0.2]  # initial position of midpoint between fingertips above table
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
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device # fingertip_midpoint_pos_initial: [0.0, 0.0, 0.2]  # initial position of midpoint between fingertips above table
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
                # do_scale=True, # was False before # TODO
            )

            SimulationContext.step(self._env._world, render=False)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step once to update PhysX with new joint velocities
        SimulationContext.step(self._env._world, render=True)
        # print("_randomize_gripper_pose ende")

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
                # do_scale=True, # was False before # TODO
            )

            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step once to update PhysX with new joint velocities
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()

    def _close_gripper(self, sim_steps) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        # print("_close_gripper")
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps) -> None:
        """Move gripper fingers to specified DOF position using controller."""
        # print("_move_gripper_to_dof_pos")
        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion 

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, 
                do_scale=False
                # do_scale=True # was False before # TODO
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
        # print("_move_gripper_to_dof_pos_async")
        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, 
                do_scale=False
                # do_scale=True # was False before # TODO
            )
            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    def _check_male_close_to_female(self) -> torch.Tensor:
        """Check if nut is close to bolt."""
        # TODO
        # print("_check_male_close_to_female")

        # keypoint_dist_A1 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_1, p=2, dim=-1), dim=-1 # frobenius norm
        # )
        # keypoint_dist_A2 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_2, p=2, dim=-1), dim=-1
        # )
        # keypoint_dist_B1 = torch.sum(
        #     torch.norm(self.keypoints_female_2 - self.keypoints_male_1, p=2, dim=-1), dim=-1
        # )
        # keypoint_dist_B2 = torch.sum(
        #     torch.norm(self.keypoints_female_1 - self.keypoints_male_2, p=2, dim=-1), dim=-1
        # )

        # keypoint_dist_A=torch.add(keypoint_dist_A1,keypoint_dist_A2)
        # keypoint_dist_B=torch.add(keypoint_dist_B1,keypoint_dist_B2)


        # # Stack the tensors along a new axis (axis=1)
        # key_point_dist_stacked = torch.stack([keypoint_dist_A, keypoint_dist_B], dim=1)

        # # Find the maximum value along the new axis (axis=1)
        # keypoint_dist_min, _ = torch.min(key_point_dist_stacked, dim=1)

        keypoint_dist_min=self._get_keypoint_dist()


        is_male_close_to_female = torch.where( # torch.where(condition, input, other) Returns tensor of elements selected from either input or other depending on condition (input if condition=true) 
            torch.sum(keypoint_dist_min, dim=-1) < self.cfg_task.rl.close_error_thresh, # threshold below which nut is considered close enough to bolt
            torch.ones_like(self.progress_buf),                                     # TODO: adjust threshold?
            torch.zeros_like(self.progress_buf),
        )
        

        # is_male_close_to_female = torch.where(
        #     torch.sum(self.keypoint_dist_min, dim=-1) < self.cfg_task.rl.close_error_thresh,
        #     torch.ones_like(self.progress_buf),
        #     torch.zeros_like(self.progress_buf),
        # )

        return is_male_close_to_female

    def _check_reached_checkpoint(self) -> torch.Tensor:
        """Check if nut is close to bolt."""
        # TODO
        # print("_check_male_close_to_female")
        # z_value_checkpoints = 0.1125 # /in mm from table surface
        # self.female_pos[env_ids, 2]
        # print("self.checkpoint_z_pos:", self.checkpoint_z_pos)
        # print("self.checkpoint_z_pos_shape:", self.checkpoint_z_pos.shape)
        # print("female_pos: ", self.female_pos)
        # print("female_pos_shape: ", self.female_pos.shape)

        self.positions_checkpoint = self.female_pos.clone()
        self.positions_checkpoint[:,2] += self.checkpoint_z_pos.squeeze(dim=1)
        # print("self.positions_checkpoint: ",self.positions_checkpoint)
        # print("self.female_pos: ", self.female_pos)

        checkpoint_dist = torch.norm(self.positions_checkpoint - self.male_pos_base, p=2, dim=-1) # frobenius norm

        # print("self.male_pos_base: ", self.male_pos_base)
        # print("checkpoint_dist: ", checkpoint_dist)

        # print("shape checkpoint_dist: ", checkpoint_dist.shape)
        # print("self.male_pos_base: ", self.male_pos_base)



        # dist_to_checkpoint_x = self.female_pos[env_ids, 0]-self.male_pos[env_ids,0] 
        # dist_to_checkpoint_y = self.female_pos[env_ids, 1]-self.male_pos[env_ids,1] 
        # dist_to_checkpoint_z = self.female_pos[env_ids, 2]-self.male_pos[env_ids,2] 

        # checkpoint_dist = dist_to_checkpoint_x + dist_to_checkpoint_x


        # z_pos_checkpoint: 47.5+65 mm from table top
        # checkpoint_pos = female_pos + z_pos_checkpoint


        # is_close_to_checkpoint = torch.where(# torch.where(condition, input, other) Returns tensor of elements selected from either input or other depending on condition (input if condition=true) 
        #     torch.sum(checkpoint_dist,dim=-1) < self.cfg_task.rl.close_checkpoint_error_thresh, # first try: 0.005
        #     torch.ones_like(checkpoint_dist),                                     # TODO: adjust threshold?
        #     torch.zeros_like(checkpoint_dist),
        # )
        is_close_to_checkpoint=torch.zeros_like(checkpoint_dist)
        # print("checkpoint_dist: ", checkpoint_dist)
        # print("checkpoint_dist_shape: ", checkpoint_dist.shape)



        for i in range(self._num_envs):
            is_close_to_checkpoint[i] = torch.where(
                checkpoint_dist[i] < self.cfg_task.rl.close_checkpoint_error_thresh,
                1,
                0,
            )
   
        # is_close_to_checkpoint = torch.where(# torch.where(condition, input, other) Returns tensor of elements selected from either input or other depending on condition (input if condition=true) 
        #     torch.sum(checkpoint_dist,dim=-1) < self.cfg_task.rl.close_checkpoint_error_thresh, # first try: 0.005
        #     1,
        #     0,
        # )

        self.checkpoint_buf += is_close_to_checkpoint




        # print("is_close_to_checkpoint: ", is_close_to_checkpoint)

        return is_close_to_checkpoint
    
    def quat_conjugate(self,q):
    #     if q.shape[0] != 4:
    #         raise ValueError("Input tensor must have 4 elements (w, x, y, z)")
        # w,x,y,z = q
        # return torch.tensor([w,-x,-y,-z])
        return torch.cat([q[:, :1], -q[:, 1:]], dim=1)

