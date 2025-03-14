"""
visual_cevae.py

This module implements a multi-panel interactive dashboard for the CEVAE.
A fixed DAG representing the CEVAE architecture (nodes: X, T, Y, Z with edges: Z→X, Z→T, Z→Y, T→Y)
is shown alongside time-series plots tracking key parameter statistics (e.g., average absolute weights)
and performance metrics (ELBO loss) over training epochs. In addition, an animated t-SNE visualization of
the latent variable distributions is displayed to illustrate convergence and representation shifts.
The dashboard is updated efficiently with non-blocking rendering, and the final dashboard is saved as an image.
"""

import matplotlib.pyplot as plt
import networkx as nx
import torch
import pyro
from pyro import poutine
from pyro.contrib.cevae import CEVAE, Model, Guide
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from pyro.optim import ClippedAdam
from pyro.infer import SVI
from pyro.contrib.cevae import TraceCausalEffect_ELBO

# --- Dashboard Initialization Utilities ---

def _init_dashboard():
    """
    Create a dashboard figure with 4 subplots arranged in a 2x2 grid:
        - Top-left: Static DAG (architecture)
        - Top-right: Parameter metrics time-series
        - Bottom-left: Performance (ELBO loss) time-series
        - Bottom-right: t-SNE of latent representations
    Returns a dictionary of axes and the figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    dashboard = {
        "dag": axs[0, 0],
        "param": axs[0, 1],
        "perf": axs[1, 0],
        "tsne": axs[1, 1],
        "fig": fig
    }
    plt.ion()  # Enable interactive mode.
    return dashboard

def _init_static_dag():
    """
    Initialize the static DAG for the CEVAE architecture.
    Returns (G, pos) for nodes: X, T, Y, Z with edges: Z→X, Z→T, Z→Y, T→Y.
    """
    import networkx as nx
    G = nx.DiGraph()
    nodes = ["X", "T", "Y", "Z"]
    G.add_nodes_from(nodes)
    edges = [("Z", "X"), ("Z", "T"), ("Z", "Y"), ("T", "Y")]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    return G, pos

def _draw_dag(ax, G, pos, norm_metrics):
    """
    Draw the static DAG on the provided axis, updating node colors based on normalized metrics.
    Colors interpolate between purple (#800080, low) and deep blue (#00008B, high).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#800080", "#00008B"])
    node_colors = [cmap(norm_metrics.get(node, 0.5)) for node in G.nodes()]
    ax.clear()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            arrows=True, arrowstyle='->', arrowsize=20)
    ax.set_title("CEVAE Architecture")
    
def _update_time_series(ax, epochs, series, title, ylabel):
    """
    Update a time-series plot on the given axis.
    epochs: list of epoch numbers.
    series: dict mapping series label to list of values.
    """
    ax.clear()
    for label, values in series.items():
        ax.plot(epochs, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()

def _update_tsne(ax, tsne_emb, title="Latent t-SNE"):
    """
    Update the t-SNE scatter plot on the given axis.
    tsne_emb: array of shape (n_samples, 2)
    """
    ax.clear()
    ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c='green', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.draw()
    plt.pause(0.001)

# --- Custom Visual Model and Guide ---

class VisualModel(Model):
    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist())
            x_val = pyro.sample("x", self.x_dist(z), obs=x)
            t_val = pyro.sample("t", self.t_dist(z), obs=t)
            y_val = pyro.sample("y", self.y_dist(t, z), obs=y)
        return y_val

class VisualGuide(Guide):
    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            t = pyro.sample("t", self.t_dist(x), obs=t, infer={"is_auxiliary": True})
            y = pyro.sample("y", self.y_dist(t, x), obs=y, infer={"is_auxiliary": True})
            pyro.sample("z", self.z_dist(y, t, x))
        return None

# --- Visual CEVAE with Interactive Dashboard and Training History ---

class VisualCEVAE(CEVAE):
    """
    A version of CEVAE that displays an interactive, multi-panel dashboard.
    The dashboard includes:
      - A fixed DAG of the model architecture.
      - Time-series plots of parameter metrics (average absolute weights) for nodes.
      - A time-series plot of the ELBO loss.
      - A dynamic t-SNE visualization of latent variable representations.
    The dashboard is updated every vis_update_interval epochs and saved as "final_dashboard.png".

    Now also includes debug logging to show a sample of X each epoch.
    """
    def __init__(self, feature_dim, outcome_dist="bernoulli",
                 latent_dim=20, hidden_dim=200, num_layers=3, num_samples=100):
        config = dict(feature_dim=feature_dim,
                      latent_dim=latent_dim,
                      hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      num_samples=num_samples)
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        super().__init__(feature_dim, outcome_dist, latent_dim, hidden_dim, num_layers, num_samples)
        self.model = VisualModel(config)
        self.guide = VisualGuide(config)

        # Initialize dashboard and static DAG.
        self.dashboard = _init_dashboard()
        self.G, self.pos = _init_static_dag()

        # Initialize history containers.
        self.history = {
            "epoch": [],
            "param": {"X": [], "T": [], "Y": [], "Z": []},
            "perf": [],  # ELBO loss per epoch.
            "tsne": []   # Store latent representations for t-SNE.
        }
        # Fix a validation batch for t-SNE visualization.
        self._fixed_data = None

    def compute_node_metrics(self):
        """
        Compute average absolute weight values for nodes X, T, and Y.
        For Z, we just set a fixed constant for display.
        """
        metrics = {}
        norm_x = 0.0
        count = 0
        if hasattr(self.model, 'x_nn'):
            for param in self.model.x_nn.fc.parameters():
                norm_x += param.abs().mean().item()
                count += 1
        metrics["X"] = norm_x / count if count > 0 else 0.5

        norm_t = 0.0
        count = 0
        if hasattr(self.model, 't_nn'):
            for param in self.model.t_nn.fc.parameters():
                norm_t += param.abs().mean().item()
                count += 1
        metrics["T"] = norm_t / count if count > 0 else 0.5

        norm_y0 = 0.0
        count0 = 0
        if hasattr(self.model, 'y0_nn'):
            for param in self.model.y0_nn.fc.parameters():
                norm_y0 += param.abs().mean().item()
                count0 += 1
        norm_y1 = 0.0
        count1 = 0
        if hasattr(self.model, 'y1_nn'):
            for param in self.model.y1_nn.fc.parameters():
                norm_y1 += param.abs().mean().item()
                count1 += 1
        avg_y0 = (norm_y0 / count0 if count0 else 0.5)
        avg_y1 = (norm_y1 / count1 if count1 else 0.5)
        metrics["Y"] = (avg_y0 + avg_y1) / 2

        metrics["Z"] = 1.0
        return metrics

    def _extract_latent(self, x, t, y):
        """
        Extract latent variable "z" from the guide using poutine.trace.
        Returns a tensor of shape (n_samples, latent_dim).
        """
        with poutine.trace() as tracer:
            self.guide(x, t, y)
        for name, site in tracer.trace.nodes.items():
            if name == "z":
                return site["value"]
        return None

    def _save_final_dashboard(self, filename):
        self.dashboard["fig"].savefig(filename)

    def fit(self, x, t, y, num_epochs=100, batch_size=100,
            learning_rate=1e-3, learning_rate_decay=0.1, weight_decay=1e-4,
            log_every=100, vis_update_interval=5):
        """
        Train the model using SVI with TraceCausalEffect_ELBO.
        The dashboard is updated every vis_update_interval epochs.
        Also prints out a sample X row for debugging at each epoch.
        """
        # Simple whitening function
        self.whiten = lambda data: (data - data.mean(0)) / (data.std(0) + 1e-6)

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "lrd": learning_rate_decay ** (1 / num_steps)
        })
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())

        losses = []
        # Fix a small batch for t-SNE viz
        self._fixed_data = x[:min(200, x.size(0))]

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # We'll print the first batch's X (after whitening) for debugging:
            for i, (batch_x, batch_t, batch_y) in enumerate(dataloader):
                batch_x = self.whiten(batch_x)
                
                # Debug print for the first batch each epoch
                if i == 0:
                    # Print only the first row to avoid spamming
                    print(f"[DEBUG] EPOCH {epoch}")
                
                loss = svi.step(batch_x, batch_t, batch_y, size=len(dataset)) / len(dataset)
                epoch_loss += loss
            
            losses.append(epoch_loss)
            print(f"Epoch {epoch} loss: {epoch_loss:.4f}")

            # Update the interactive dashboard every vis_update_interval epochs
            if epoch % vis_update_interval == 0:
                current_metrics = self.compute_node_metrics()
                self._update_dashboard(epoch, current_metrics, epoch_loss)

        # Final update + saving of the dashboard
        final_metrics = self.compute_node_metrics()
        self._update_dashboard(num_epochs, final_metrics, losses[-1])
        self._save_final_dashboard("final_dashboard.png")
        return losses

    def _update_dashboard(self, epoch, current_metrics, current_loss):
        """
        Update the multi-panel dashboard:
          - Node color on DAG
          - Time-series param metrics
          - Performance (ELBO loss)
          - t-SNE of latents on a fixed batch
        """
        self.history["epoch"].append(epoch)
        for node in ["X", "T", "Y", "Z"]:
            self.history["param"][node].append(current_metrics[node])
        self.history["perf"].append(current_loss)

        # Normalize param metrics for color
        norm_metrics = {}
        for node in ["X", "T", "Y", "Z"]:
            vals = self.history["param"][node]
            val_min, val_max = min(vals), max(vals)
            denom = (val_max - val_min) + 1e-8
            norm_metrics[node] = (current_metrics[node] - val_min) / denom

        _draw_dag(self.dashboard["dag"], self.G, self.pos, norm_metrics)
        _update_time_series(
            self.dashboard["param"], 
            self.history["epoch"],
            self.history["param"],
            "Parameter Metrics",
            "Metric Value"
        )
        _update_time_series(
            self.dashboard["perf"],
            self.history["epoch"],
            {"ELBO": self.history["perf"]},
            "ELBO Loss",
            "Loss"
        )

        # Now do t-SNE on the "fixed" data
        with torch.no_grad():
            fixed_x = self._fixed_data
            # dummy T and Y for visualization
            fixed_t = torch.zeros(fixed_x.size(0), device=fixed_x.device)
            fixed_y = torch.zeros(fixed_x.size(0), device=fixed_x.device)
            latent = self._extract_latent(fixed_x, fixed_t, fixed_y)
            if latent is not None:
                from sklearn.manifold import TSNE
                latent_np = latent.detach().cpu().numpy()
                tsne = TSNE(n_components=2, random_state=42)
                tsne_emb = tsne.fit_transform(latent_np)
                _update_tsne(self.dashboard["tsne"], tsne_emb)

        self.dashboard["fig"].canvas.draw()
        plt.pause(0.001)
