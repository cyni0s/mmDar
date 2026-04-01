# model — physics-first Gaussian radar model
"""Model assembly for mmDar v2.

Pipeline: ClassicalFFTFrontend → Deep2DEncoder → PhysicsFirstEncoder → GaussianSetDecoder
Input:  (B, T, 8, 512) complex64 windowed radar frames
Output: Gaussian set (mu, log_sigma, logits) per frame

Top-level model: PhysicsGaussianModel (v2.model.physics_frontend)
"""

from model.physics_frontend import (
    PhysicsGaussianModel,
    PhysicsFirstEncoder,
    ClassicalFFTFrontend,
    Deep2DEncoder,
)
from model.gaussian_head import GaussianSetDecoder, DecoderLayer
from model.beamspace import LearnedBeamspace, DilatedResBlock1d
from model.lista import FFTBeamformer, build_steering_matrix

__all__ = [
    "PhysicsGaussianModel",
    "PhysicsFirstEncoder",
    "ClassicalFFTFrontend",
    "Deep2DEncoder",
    "GaussianSetDecoder",
    "DecoderLayer",
    "LearnedBeamspace",
    "DilatedResBlock1d",
    "FFTBeamformer",
    "build_steering_matrix",
]
