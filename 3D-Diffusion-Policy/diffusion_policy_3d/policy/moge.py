from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from adapters import AutoAdapterModel
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply

class LoRALinear(nn.Module):
    def __init__(self, original_linear, lora_rank, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear
        self.lora_A = nn.Linear(original_linear.in_features, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, original_linear.out_features, bias=False)
        self.scaling = alpha / lora_rank

    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

def replace_with_lora(model, lora_rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, lora_rank, alpha))
        else:
            replace_with_lora(module, lora_rank, alpha)

class DiffusionUnetHybridImagePolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            frozen=True,
            # parameters passed to step
            **kwargs):
        super().__init__()
        from moge.model import MoGeModel
        self.img_encoder = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to('cuda')
        lora_rank = 2
        alpha = 4
        replace_with_lora(self.img_encoder, lora_rank, alpha)
        for param in self.img_encoder.parameters():
            param.requires_grad = False

        for name, module in self.img_encoder.named_modules():
            if isinstance(module, LoRALinear):
                for param in module.lora_A.parameters():
                    param.requires_grad = True
                for param in module.lora_B.parameters():
                    param.requires_grad = True
        # AutoAdapterModel.from_pretrained("Ruicheng/moge-vitl")
        # from IPython import embed; embed(); exit()
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        
        # create diffusion model
        # TODO
        robot_state_dim = obs_key_shapes['agent_pos'][0]
        obs_feature_dim = 64 + robot_state_dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        self.img_feature_proj = nn.Linear(1024, 64)
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.frozen = frozen
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        img = obs_dict['img']
        agent_pos = self.normalizer.normalize({'agent_pos':obs_dict['agent_pos']})['agent_pos']


        B, To = img.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        if 'feature' in obs_dict:
            feature = obs_dict['feature']
        else:
            img_shape = img.shape
            img = img.reshape(-1, *img_shape[2:])
            with torch.no_grad():
                feature = self.img_encoder.get_embedding(img)
            feature = feature.reshape(img_shape[0], img_shape[1], -1)
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            feature = feature[:,:self.n_obs_steps].reshape(-1,*feature.shape[2:])
            feature = self.img_feature_proj(feature)
            # reshape back to B, Do
            feature = feature.reshape(B, -1, 64)
            global_cond = torch.cat([agent_pos[:,:self.n_obs_steps], feature], dim=-1).reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            raise NotImplementedError("Not implemented yet")

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        agent_pos = self.normalizer.normalize({'agent_pos':batch['obs']['agent_pos']})['agent_pos']
        if self.frozen:
            feature = batch['obs']['feature']
        else:
            img = batch['obs']['img'][:,:self.n_obs_steps]
            img_shape = img.shape
            img = img.reshape(-1, *img_shape[2:])
            img = img.permute(0, 3, 1, 2)/255.0
            features = []
            feature = self.img_encoder.get_embedding(img)
            feature = feature.reshape(img_shape[0], img_shape[1], -1)
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            feature = feature[:,:self.n_obs_steps].reshape(-1,*feature.shape[2:])
            feature = self.img_feature_proj(feature)
            # reshape back to B, Do
            feature = feature.reshape(batch_size, -1, 64)
            global_cond = torch.cat([agent_pos[:,:self.n_obs_steps], feature], dim=-1).reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            # this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            # nobs_features = self.obs_encoder(this_nobs)
            # # reshape back to B, T, Do
            # nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            # cond_data = torch.cat([nactions, nobs_features], dim=-1)
            # trajectory = cond_data.detach()
            raise NotImplementedError("Not implemented yet")

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
    
    