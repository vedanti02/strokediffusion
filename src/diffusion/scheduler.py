"""
Diffusion noise schedulers for training and sampling.
"""
import torch
import math


class Diffusion:
    """Simple linear beta schedule diffusion."""
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.steps = num_steps
        self.device = device
        
        beta = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alpha = 1. - beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to x at timestep t.
        
        Args:
            x: Clean data of shape (B, ...)
            t: Timesteps of shape (B,)
        
        Returns:
            Tuple of (noisy_x, noise)
        """
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise


class CosineScheduler:
    """Cosine noise schedule (better for images/sequences)."""
    
    def __init__(self, num_steps: int = 1000, s: float = 0.008, device: str = "cuda"):
        self.num_steps = num_steps
        self.device = device
        
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas_cumprod = torch.cumprod(1. - self.betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, original: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t using cosine schedule."""
        noise = torch.randn_like(original)
        noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * original +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        return noisy, noise


class XStartScheduler:
    """
    Scheduler for x_start prediction (predicting clean data directly).
    Includes sampling logic for generation.
    """
    
    def __init__(self, num_steps: int = 1000, s: float = 0.008, device: str = "cuda"):
        self.num_steps = num_steps
        self.device = device
        
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Posterior coefficients for x_start prediction
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.clip(self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), 1e-20, 1.0)
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        n_samples: int = 1,
        seq_len: int = 20,
        feat_dim: int = 65,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using x_start prediction.
        
        Args:
            model: Trained model that predicts x_start
            n_samples: Number of samples to generate
            seq_len: Sequence length
            feat_dim: Feature dimension
            verbose: Print progress
        
        Returns:
            Generated samples of shape (n_samples, seq_len, feat_dim)
        """
        if verbose:
            print(f"Generating {n_samples} painting(s) using x_start prediction...")
        
        model.eval()
        x = torch.randn(n_samples, seq_len, feat_dim).to(self.device)
        
        for i, t in enumerate(reversed(range(0, self.num_steps))):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            
            # Model predicts x_start directly
            pred_x_start = model(x, t_batch)
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
            
            # Calculate posterior mean
            posterior_mean = (
                self.posterior_mean_coef1[t] * pred_x_start +
                self.posterior_mean_coef2[t] * x
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                posterior_variance = torch.exp(0.5 * self.posterior_log_variance_clipped[t])
                x = posterior_mean + posterior_variance * noise
            else:
                x = posterior_mean
            
            if verbose and i % 100 == 0:
                if x.abs().max() > 2.0:
                    x = torch.clamp(x, -2.0, 2.0)
                print(f"   Step {t}/{self.num_steps} | Max: {x.max():.2f}")
        
        return x
