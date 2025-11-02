"""

In this file, we implement different ways to sample noise during the training procedure of DDPMs and LDMs.

"""

from abc import ABC, abstractmethod
import torch
from torch.distributions import Normal

class Sampling(ABC):

    def __init__(self, scheduler, t_min=0):

        self.t_max = scheduler.steps
        self.t_min = t_min

    @abstractmethod
    def __call__(self):
        pass

class UniformSampling(Sampling):

    """
    Uniform temperature sampler
    """
    def __init__(self, scheduler, t_min=0):
        super().__init__(scheduler, t_min)

    def __call__(self, x_0):

        # Add 1 to the t_max, because of how  randint handles the high value. By doing that, we include t_max as a possibility too.
        self.t_max += 1

        size = (x_0.size(0),1) # Take in the size of t
        device = x_0.device    # Take in the device of t
        dtype = x_0.dtype      # Take in the dtype of t

        t = torch.randint(low=self.t_min, high=self.t_max, size=size, device=device)

        return t.to(dtype=dtype)

class NormalSampling(Sampling):

    """
    Temperature sampler with normal distribution
    """

    def __init__(self, scheduler, t_min:int = 0, containment_percentage:float = 0.999):
        super().__init__(scheduler, t_min)

        self.mean = round((self.t_max - self.t_min) / 2) # Sets the mean of the normal distribution as the mid-point temperature

        # Calculate the required std_dev given the containment_percentage within limits
        p_one_tail = (1.0 - containment_percentage)/2
        p_quantile = 1.0-p_one_tail
        m_std_norm = Normal(0.0,1.0)
        z_score = m_std_norm.icdf(torch.tensor(p_quantile, dtype=torch.float32))

        self.std_dev = self.mean/ z_score.item()

    def __call__(self, x_0):

        size = (x_0.size(0),1) # Take in the size of t
        device = x_0.device    # Take in the device of t
        dtype = x_0.dtype      # Take in the dtype of t

        t_float = torch.normal(
            mean= self.mean, std= self.std_dev, size=size, device=device
        )

        t_float = torch.clamp(t_float, min=self.t_min, max=self.t_max)

        t = torch.round(t_float)

        return t.to(dtype=dtype)
