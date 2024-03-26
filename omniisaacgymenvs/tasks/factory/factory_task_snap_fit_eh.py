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

"""Factory: Class for nut-bolt place task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
PYTHON_PATH omniisaacgymenvs/scripts/rlgames_train.py task=FactoryTaskNutBoltPlace
cd OmniIsaacGymEnvs_EH/omniisaacgymenvs
OR: cd OmniIsaacGymEnvs/omniisaacgymenvs
PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskPegHolePlace_eh
tensorboard: PYTHON_PATH -m tensorboard.main --logdir runs/FactoryTaskPegHolePlace_eh_400epochs_0.1mass_0.3friction-both_0actionpenalty_withNoise/summaries (rund from cd OmniIsaacGymEnvs/omniisaacgymenvs)
running: PYTHON_PATH scripts/rlgames_train.py task=FactoryTaskPegHolePlace_eh test=True checkpoint=runs/FactoryTaskPegHolePlace_eh_400epochs_0.1mass_0.3friction-both_0.3actionpenalty_withNoise/nn/FactoryTaskPegHolePlace_eh.pth  (ANPASSEN)
# """
# For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
# For Windows: doskey PYTHON_PATH=C:\Users\emmah\AppData\Local\ov\pkg\isaac_sim-2023.1.1\python.bat $* 
# """

import asyncio
import hydra
import math
import omegaconf
import torch
from typing import Tuple

import omni.kit
from omni.isaac.core.simulation_context import SimulationContext
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.transformations import tf_combine

import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_env_peg_hole_eh import FactoryEnvPegHole_eh # TODO: richtig so??
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)


class FactoryTaskPegHolePlace_eh(FactoryEnvPegHole_eh, FactoryABCTask):
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

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_peg_hole_eh.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_peg_hole = hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting
        # TODO: adjust yaml for ppo --> done
        ppo_path = "train/FactoryTaskPegHolePlace_ehPPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
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
        
        # self.peg_base_pos_local = self.hole_drill_hole_heights * torch.tensor(    # result: list of tensors like torch.tensor([0.0, 0.0, bolt_head_height]) --> nut_pos_base local is equal to [0,0,bolt_head_height] for each env.
        #     [0.0, 0.0, 1.0], device=self.device
        # ).repeat((self.num_envs, 1))                                        # repeat could be omitted since it should be len(bolt_head_height) = num_envs

        # self.peg_base_pos_local = self.peg_heights * torch.tensor(    # result: list of tensors like torch.tensor([0.0, 0.0, bolt_head_height]) --> nut_pos_base local is equal to [0,0,bolt_head_height] for each env.
        #     [0.0, 0.0, 1.0], device=self.device                         # TODO: check if 1mm should be added to peg_heights? to get similar thingy as in Factory paper?
        # ).repeat((self.num_envs, 1))  
        
        # self.peg_base_pos_local = self.base_pos_zeros * torch.tensor(   # funktioniert irgendwie nicht?? ("carb ist not defined" oder so) # result: list of tensors like torch.tensor([0.0, 0.0, bolt_head_height]) --> nut_pos_base local is equal to [0,0,bolt_head_height] for each env.
        #     [0.0, 0.0, 1.0], device=self.device
        # ).repeat((self.num_envs, 1))  

        self.peg_base_pos_local = self.peg_heights * torch.tensor(    # result: list of tensors like torch.tensor([0.0, 0.0, bolt_head_height]) --> nut_pos_base local is equal to [0,0,bolt_head_height] for each env.
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))*0.5

        # self.peg_base_pos_local = self.peg_heights * torch.tensor(    # try 4 (mesh i constructed in isaac sim, KOS in the middle of z)
        #     [0.0, 0.0, 1.0], device=self.device
        # ).repeat((self.num_envs, 1))*0.2


        
        hole_heights = self.hole_heights 
        self.hole_tip_pos_local = hole_heights * torch.tensor( # highest position of hole thingy
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))

        # Keypoint tensors
        '''
        num_keypoints: 4  # number of keypoints used in reward
        keypoint_scale: 0.5  # length of line of keypoints
        '''
        self.keypoint_offsets = (
            self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints)
            * self.cfg_task.rl.keypoint_scale 
        )
        self.keypoints_peg = torch.zeros(  # tensor of zeros of size (num_envs, num_keypoints, 3)
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_hole = torch.zeros_like(self.keypoints_peg, device=self.device) # tensor of zeros of same size as self.keypoints_peg

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

        # Close gripper onto peg
        self.disable_gravity()  # to prevent peg from falling
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

        # Close gripper onto peg
        self.disable_gravity()  # to prevent peg from falling
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
                # (self.peg_widths * 0.5)
                (self.peg_widths * 0.5)                                 # gripper pos?
                * gripper_buffer,  # buffer on gripper DOF pos to prevent initial contact (was 1.1 before)
                # (self.peg_widths * 0.5) * 1.1,
                (self.peg_widths * 0.5) * gripper_buffer                # gripper pos?
                
            ),  # buffer on gripper DOF pos to prevent initial contact (was 1.1 before)
            dim=-1,
        )  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    def _reset_object(self, env_ids) -> None:
        """Reset root states of nut and bolt."""
        # print("reset_object")
        # Randomize root state of nut within gripper
        self.peg_pos[env_ids, 0] = 0.0 # x-coordinate of position?
        self.peg_pos[env_ids, 1] = 0.0 # y-coordinate of position?

        # self.peg_pos[env_ids, 0] = 1.0 # makes no difference??
        # self.peg_pos[env_ids, 1] = 1.0 # makes no difference??


        # TODO: check out if value for fingertip_midpoint_pos_reset has to be adjusted (was 0.58781 before) (franka_finger_length=0.053671, from factory_asset_info_franka_table)
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset # for  fingertip_midpoint_pos_reset = 0.28781 peg "jumped" ot of table?? (TODO)
        # print("fingertip_midpoint_pos_reset: ",fingertip_midpoint_pos_reset)

        # peg_base_pos_local = self.hole_drill_hole_heights.squeeze(-1) # first try in setting peg_pos_base_local
        
        # peg_base_pos_local = self.peg_heights.squeeze(-1) # second try in setting peg_pos_base_local # TODO: check if adding 1mm to peg_heights makes sense? 
        
        
        # peg_base_pos_local = self.base_pos_zeros.squeeze(-1)


        peg_base_pos_local = self.peg_heights.squeeze(-1) * 0.5 # try 3
        # peg_base_pos_local = self.peg_heights.squeeze(-1) *0.2 # try 4; mesh I constructed in isaac sim -->KOS in the middle


        self.peg_pos[env_ids, 2] = fingertip_midpoint_pos_reset - peg_base_pos_local*0.5 # z-coordinate of position? TODO: ist peg_base_local random gewählt -->irgendein Abstand der ungefähr passen könnte, damit abstand zwischen fingertip_midpoint_pos_reset und bolt
        peg_noise_pos_in_gripper = 2 * ( 
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        peg_noise_pos_in_gripper = peg_noise_pos_in_gripper @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.peg_noise_pos_in_gripper, device=self.device
            )
        )
        self.peg_pos[env_ids, :] += peg_noise_pos_in_gripper[env_ids] 

        peg_rot_euler = torch.tensor(
            [0.0, 0.0, math.pi * 0.5], device=self.device
        ).repeat(len(env_ids), 1)
        peg_noise_rot_in_gripper = 2 * (
            torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        peg_noise_rot_in_gripper *= self.cfg_task.randomize.peg_noise_rot_in_gripper
        peg_rot_euler[:, 2] += peg_noise_rot_in_gripper  
        peg_rot_quat = torch_utils.quat_from_euler_xyz(
            peg_rot_euler[:, 0], peg_rot_euler[:, 1], peg_rot_euler[:, 2]
        )
        self.peg_quat[env_ids, :] = peg_rot_quat

        self.peg_linvel[env_ids, :] = 0.0
        self.peg_angvel[env_ids, :] = 0.0

        indices = env_ids.to(dtype=torch.int32)
        # print("checkpoint1 ", env_ids)
        
        
        self.pegs.set_world_poses(
            self.peg_pos[env_ids] + self.env_pos[env_ids],
            self.peg_quat[env_ids],
            indices,
        )
        self.pegs.set_velocities(
            torch.cat((self.peg_linvel[env_ids], self.peg_angvel[env_ids]), dim=1),
            indices,
        )
        # print("checkpoint2 ", env_ids)
        # print("self.peg_pos[env_ids] + self.env_pos[env_ids]=",self.peg_pos[env_ids] + self.env_pos[env_ids])
        # Randomize root state of hole
        hole_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        hole_noise_xy = hole_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.hole_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.hole_pos[env_ids, 0] = ( # x
            self.cfg_task.randomize.hole_pos_xy_initial[0] + hole_noise_xy[env_ids, 0]
        )
        self.hole_pos[env_ids, 1] = ( # y
            self.cfg_task.randomize.hole_pos_xy_initial[1] + hole_noise_xy[env_ids, 1]
        )

        # self.hole_pos[env_ids, 0] = ( # x ###REMOVED NOISE FOR TESTING
        #     self.cfg_task.randomize.hole_pos_xy_initial[0]
        # )
        # self.hole_pos[env_ids, 1] = ( # y ###REMOVED NOISE FOR TESTING
        #     self.cfg_task.randomize.hole_pos_xy_initial[1]
        # )


        self.hole_pos[env_ids, 2] = self.cfg_base.env.table_height # z # table_height=0.4 (from cfg->task->FactoryBase.yaml)

        self.hole_quat[env_ids, :] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(len(env_ids), 1)

        indices = env_ids.to(dtype=torch.int32)
        self.holes.set_world_poses(
            self.hole_pos[env_ids] + self.env_pos[env_ids],
            self.hole_quat[env_ids],
            indices,
        )

    def _reset_buffers(self, env_ids) -> None:
        # print("_reset_buffers")
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale # do_scale:bool
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

        if self.cfg_ctrl["do_force_ctrl"]:
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

        for idx, keypoint_offset in enumerate(self.keypoint_offsets): # keypoints are placed in even distance in a line in z-direction 
            self.keypoints_peg[:, idx] = tf_combine(
                self.peg_quat,
                self.peg_pos,
                self.identity_quat,
                (keypoint_offset - self.peg_base_pos_local), # before:keypoint_offset + self.peg_base_pos_local (changed on 05/03/2024; 14:16) 
            )[1]
            self.keypoints_hole[:, idx] = tf_combine(
                self.hole_quat,
                self.hole_pos,
                self.identity_quat,
                (keypoint_offset + self.hole_tip_pos_local-0.03),  # before:  keypoint_offset + self.hole_tip_pos_local (changed on 05/03/2024; 14:16) # try 07/03/2024: change from bottom of drill hole (keypoint_offset + self.hole_drill_hole_heights) to center of hole, on cube surface (hole_tip_pos_local)
            )[1]

    def get_observations(self) -> dict:
        """Compute observations."""
        # print("get_observations")
        # Shallow copies of tensors
        obs_tensors = [
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_linvel,
            self.fingertip_midpoint_angvel,
            self.peg_pos,
            self.peg_quat,
            self.hole_pos,
            self.hole_quat,
        ]

        if self.cfg_task.rl.add_obs_hole_tip_pos: # TODO: edit in task config file (add_obs_hole_tip_pos is boolean value) # add observation of bolt tip position
            obs_tensors += [self.hole_tip_pos_local]

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
        action_penalty = (
            torch.norm(self.actions, p=2, dim=-1)
            * self.cfg_task.rl.action_penalty_scale
        )

        '''
        default cfg values: 
        keypoint_reward_scale: 1.0  # scale on keypoint-based reward
        action_penalty_scale: 0.0  # scale on action penalty
        '''

        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale
        )

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            # Check if nut is close enough to bolt
            is_peg_close_to_hole = self._check_peg_close_to_hole() # adjust threshold in task config file
            self.rew_buf[:] += is_peg_close_to_hole * self.cfg_task.rl.success_bonus # if close to bolt, --> successbonus*1 else successbonus*0; sucess_bonus defined in cfg-task-yaml-file (currently =0)
            self.extras["successes"] = torch.mean(is_peg_close_to_hole.float())

    def _get_keypoint_offsets(self, num_keypoints) -> torch.Tensor:
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        Last column contains values ranging from -0.5 to 0.5 in a linearly spaced manner. The first two columns are filled with zeros. """
        # print("_get_keypoint_offsets")
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = ( # 
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5 #(try 06/04/2024: commented out the -0.5; went back on 07/03/2027 but with middle keypoint on topsurface of hole)
        ) 
        

        return keypoint_offsets

    def _get_keypoint_dist(self) -> torch.Tensor:
        """Get keypoint distance between nut and bolt."""
        # print("_get_keypoint_dist")
        keypoint_dist = torch.sum(
            torch.norm(self.keypoints_hole - self.keypoints_peg, p=2, dim=-1), dim=-1
        )

        return keypoint_dist

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
            )

            SimulationContext.step(self._env._world, render=True)

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

        # Step once to update PhysX with new joint velocities
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()

    def _close_gripper(self, sim_steps) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        # print("_close_gripper")
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps) -> None:
        """Move gripper fingers to specified DOF position using controller."""
        print("_move_gripper_to_dof_pos")#
        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, do_scale=False
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
                delta_hand_pose, gripper_dof_pos, do_scale=False
            )
            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    def _check_peg_close_to_hole(self) -> torch.Tensor:
        """Check if nut is close to bolt."""
        # print("_check_peg_close_to_hole")
        keypoint_dist = torch.norm(
            self.keypoints_hole - self.keypoints_peg, p=2, dim=-1
        )

        is_peg_close_to_hole = torch.where( # torch.where(condition, input, other) Returns tensor of elements selected from either input or other depending on condition (input if condition=true) 
            torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh, # threshold below which nut is considered close enough to bolt
            torch.ones_like(self.progress_buf),                                     # TODO: adjust threshold?
            torch.zeros_like(self.progress_buf),
        )

        return is_peg_close_to_hole
