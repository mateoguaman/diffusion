import random
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

def make_scatter_animation(frames, filename="animation.mp4", fps=20, s=5, alpha=0.7):
    """
    frames: list of tensors or np arrays of shape (N,2) or (N,d)
    filename: output mp4 file
    fps: frames per second
    s: scatter point size
    alpha: scatter transparency
    """
    # Convert to CPU numpy for safety
    processed = []
    for f in frames:
        if hasattr(f, "detach"):
            f = f.detach().cpu().numpy()
        else:
            f = np.asarray(f)
        processed.append(f)

    # Create figure
    fig, ax = plt.subplots(figsize=(5,5))
    writer = FFMpegWriter(fps=fps)
    
    # Axis limits fixed across frames (prevents popping)
    all_points = np.concatenate(processed, axis=0)
    x_min, x_max = all_points[:,0].min(), all_points[:,0].max()
    y_min, y_max = all_points[:,1].min(), all_points[:,1].max()

    with writer.saving(fig, filename, dpi=150):
        for i, pts in enumerate(processed):
            ax.clear()
            ax.scatter(pts[:,0], pts[:,1], s=s, alpha=alpha)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Frame {i}")
            ax.set_xticks([])
            ax.set_yticks([])
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved animation to {filename}")

class TimeEmbedding(nn.Module):
    def __init__(self, num_freqs, std, max_t):
        super().__init__()
        self.num_freqs = num_freqs
        self.std = std
        self.max_t = max_t
        B = torch.normal(mean=torch.zeros(num_freqs, 1), std=self.std)
        self.register_buffer("B", B)

    def forward(self, t):
        '''
        t is an integer tensor (batch, 1) or (batch,)
        '''
        t_norm = (t / self.max_t).float().reshape(-1, 1)
        angles = 2 * torch.pi * t_norm @ self.B.T
        fourier_features = torch.concatenate([torch.sin(angles), torch.cos(angles)], dim=-1)
        return fourier_features


class MLP(nn.Module):
    def __init__(self, hidden_sizes, data_dim, num_freqs, time_emb_std, max_t):
        super().__init__()
        self.time_embedder = TimeEmbedding(num_freqs, time_emb_std, max_t)
        layers = [nn.Linear(data_dim+(2*num_freqs), hidden_sizes[0]), nn.ELU()]
        for i in range(0, len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_sizes[-1], data_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x, t):
        t_emb = self.time_embedder(t)
        concat_input = torch.concatenate([x.to(torch.float), t_emb.to(torch.float)], dim=-1)
        out = self.model(concat_input)
        return out

class NoiseSchedule:
    def __init__(self, num_steps, beta_1, beta_T):
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T

        self.betas = torch.linspace(self.beta_1, self.beta_T, self.num_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)

    def beta(self, t):
        '''
        t: (n,) tensor of type long of timesteps to query
        '''
        return self.betas[t.to(torch.long)]

    def alpha(self, t):
        '''
        t: (n,) tensor of type long of timesteps to query
        '''
        return self.alphas[t.to(torch.long)]
    
    def alpha_bar(self, t):
        '''
        t: (n,) tensor of type long of timesteps to query
        '''
        return self.alpha_bars[t.to(torch.long)]
        

class DDPM:
    def __init__(self, model, schedule, optimizer, data_dim):
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.data_dim = data_dim

    def sample(self, num_samples=1):
        # Run denoising process for t steps
        with torch.no_grad():
            all_samples = []
            x_t = torch.randn(num_samples, self.data_dim)
            all_samples.append(x_t)
            for t in range(self.schedule.num_steps-1, -1, -1):
                if t > 0:
                    noise = torch.randn(num_samples, self.data_dim) 
                else:
                    noise = 0

                t_tensor = torch.ones(num_samples) * t

                alpha_t = self.schedule.alpha(t_tensor).reshape(-1, 1)
                alpha_bar_t = self.schedule.alpha_bar(t_tensor).reshape(-1, 1)
                sigma_t = torch.sqrt(self.schedule.beta(t_tensor).reshape(-1, 1))

                pred_noise = self.model(x_t, t_tensor)
                x_t = 1/torch.sqrt(alpha_t) * (x_t - (1-alpha_t)/(torch.sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * noise
                all_samples.append(x_t)

        return x_t, all_samples

def load_dataset(num_datapoints=10000, plot_data=False):
    data, _ = make_moons(n_samples = num_datapoints, noise=0.05)
    if plot_data:
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("Ground truth data")
        plt.tight_layout()
        plt.show()
    data = torch.tensor(data, dtype=torch.float32)
    return data

def training_step(ddpm, batch):
    t = torch.randint(0, ddpm.schedule.num_steps, (batch.shape[0],))  # This should be of shape (n,)
    noise = torch.randn(batch.shape)  # This should be of shape (n, d)

    alpha_bar_t = ddpm.schedule.alpha_bar(t).reshape(-1, 1)  # This should be of shape (n,)
    noised_data = torch.sqrt(alpha_bar_t) * batch + torch.sqrt(1 - alpha_bar_t) * noise  # This should be of shape (n, d)

    loss_fn = torch.nn.MSELoss()

    ddpm.optimizer.zero_grad()
    pred_noise = ddpm.model(noised_data, t)
    loss = loss_fn(pred_noise, noise)
    loss.backward()
    ddpm.optimizer.step()

    return loss

def evaluate(ddpm, num_samples=1):
    samples, _ = ddpm.sample(num_samples)

    # TODO: Compute metrics if they exist
    metrics = None

    return samples, metrics


def main():
    # Define training parameters
    num_datapoints = 10000
    denoising_steps = 10
    beta_1 = 0.0001
    beta_T = 0.02
    learning_rate = 3e-3
    batch_size = 512
    epochs = 500 
    evaluate_every = 10
    num_freqs = 8
    time_emb_std = 1

    # Load dataset
    data = load_dataset(num_datapoints=10000, plot_data=True)
    num_datapoints = data.shape[0]
    data_dim = data.shape[-1]

    # Instantiate model, noise schedule, DDPM, and optimizer
    model = MLP([64, 64], data_dim, num_freqs, time_emb_std, denoising_steps)
    schedule = NoiseSchedule(denoising_steps, beta_1, beta_T)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    ddpm = DDPM(model, schedule, optimizer, data_dim)
    
    # Create training loop
    batches = [data[batch_size*i: min(batch_size*(i+1), len(data)), :] for i in range(num_datapoints//batch_size)]

    num_training_steps = 0
    losses = []
    samples = []
    for i in range(epochs):
        random.shuffle(batches)
        for b, batch in enumerate(batches):
            loss = training_step(ddpm, batch)
            print(f"Epoch {i}, batch {b}/{len(batches)}. Loss: {loss}")
            losses.append(loss.detach().cpu().item())
            num_training_steps += 1

            if num_training_steps % evaluate_every == 0:
                sample, _ = evaluate(ddpm)
                samples.append(sample)

    # Final evaluation
    _, all_denoising_steps = ddpm.sample(num_samples=1000)

    # Plot 10 uniformly sampled denoising steps
    num_steps_to_plot = 10
    total_steps = len(all_denoising_steps)
    # Evenly spaced indices from 0 to last step (inclusive)
    step_indices = torch.linspace(0, total_steps - 1, num_steps_to_plot).round().to(torch.long).tolist()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()
    for ax, idx in zip(axes, step_indices):
        x = all_denoising_steps[idx]
        # Use first two dimensions for a simple 2D scatter
        ax.scatter(x[:, 0].detach(), x[:, 1].detach(), s=5, alpha=0.7)
        ax.set_title(f"t = {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    # Plot losses
    plt.plot(np.array(losses))
    plt.title("Losses")
    plt.tight_layout()
    plt.show()

    make_scatter_animation(all_denoising_steps, filename="denoising.mp4", fps=30)



if __name__ == "__main__":
    main()