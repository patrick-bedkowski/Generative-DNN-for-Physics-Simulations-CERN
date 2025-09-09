import torch

from .proton import generator as proton_generator
from .proton import discriminator as proton_discriminator
from .proton import aux_reg as proton_aux_reg
from .neutron import generator as neutron_generator
from .neutron import discriminator as neutron_discriminator
from .neutron import aux_reg as neutron_aux_reg
from .routers import router

MODEL_REGISTRY = {
    "proton.generator": proton_generator.Generator,
    "proton.generator_unified": proton_generator.GeneratorUnified,
    "proton.discriminator": proton_discriminator.Discriminator,
    "proton.discriminator_unified": proton_discriminator.DiscriminatorUnified,
    "proton.aux_reg": proton_aux_reg.AuxReg,
    "neutron.generator": neutron_generator.GeneratorNeutron,
    "neutron.discriminator": neutron_discriminator.DiscriminatorNeutron,
    "neutron.aux_reg": neutron_aux_reg.AuxRegNeutron,
    "router_v1": router.RouterNetwork,
    "router_attention": router.AttentionRouterNetwork,
}


def build_model(name: str, model_specs: dict, device: torch.device):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**model_specs).to(device)
