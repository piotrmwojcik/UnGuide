import torch

class AutoGuidedModel:
    def __init__(self, model_full, model_unlearned, w=0.5, cfg_scale=7.5):
        self.model_full = model_full
        self.model_unlearned = model_unlearned
        self.w = w
        self.cfg_scale = cfg_scale
        self.device = model_full.device

    def apply_model(self, x, t, cond=None, uncond=None, mode="default"):
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c = torch.cat([uncond, cond], dim=0)

        eps_f_uncond, eps_f_cond = self.model_full.apply_model(x_in, t_in, c).chunk(2)
        eps_full = eps_f_uncond + self.cfg_scale * (eps_f_cond - eps_f_uncond)

        eps_u_uncond, eps_u_cond = self.model_unlearned.apply_model(x_in, t_in, c).chunk(2)
        eps_unlearned = eps_u_uncond + self.cfg_scale * (eps_u_cond - eps_u_uncond)

        return eps_full + eps_unlearned
        #return self.w * eps_full + (1 - self.w) * eps_unlearned

    def get_learned_conditioning(self, prompts):
        return self.model_full.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.model_full.decode_first_stage(z)

    def eval(self):
        self.model_full.eval()
        self.model_unlearned.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model_full, name)