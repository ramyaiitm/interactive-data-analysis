# analysis.py
# Interactive analysis notebook (percent-format for Jupyter & VS Code)
# Author: Data Scientist (example)
# Contact: 22f3002140@ds.study.iitm.ac.in   <-- email included as a comment (required)
#
# Description:
# - Demonstrates relationships between variables using a synthetic dataset.
# - Interactive slider to change correlation strength and see dynamic markdown + plot.
# - Two core cells have variable dependencies (data generation -> interactive visualization).
# - Comments document data flow between cells.

# %% [1] Imports and Data Generator (Cell 1)
# This cell defines the data generator and creates an initial dataset.
# Data flow: downstream cells will consume `base_x` and the `generate_dataset` function
# so that widgets can request new datasets without re-running this cell manually.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
from typing import Tuple

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def generate_dataset(n: int = 200, correlation_strength: float = 0.7, noise_scale: float = 1.0) -> pd.DataFrame:
    """
    Generate a synthetic dataset where y is correlated with x.
    Args:
        n: number of samples
        correlation_strength: between -1.0 and 1.0 controlling linear dependence
        noise_scale: multiplier for the noise term
    Returns:
        DataFrame with columns ['x', 'y']
    """
    # base_x is a global-like variable produced here and consumed downstream
    base_x = rng.normal(loc=0.0, scale=5.0, size=n)
    # Ensure correlation_strength is within [-1, 1]
    corr = float(max(-1.0, min(1.0, correlation_strength)))
    # Build y as a linear function of x with noise
    y = corr * (base_x * 1.5) + rng.normal(loc=0.0, scale=noise_scale, size=n)
    df = pd.DataFrame({"x": base_x, "y": y})
    return df

# Create an initial dataset to be used as the 'base' by the interactive cell
base_n = 200
base_corr = 0.6
base_noise = 1.0
df_base = generate_dataset(n=base_n, correlation_strength=base_corr, noise_scale=base_noise)

# Brief preview (not required but helpful when opening in a notebook)
print(f"Initial dataset created: n={base_n}, corr={base_corr}, noise={base_noise}")
display(df_base.head())

# %% [2] Interactive visualization with slider(s) (Cell 2)
# This cell depends on the `generate_dataset` function and `df_base` defined above.
# Data flow: user moves the slider -> this cell requests a new dataset via generate_dataset()
# and then updates a Matplotlib scatter plot and a dynamic Markdown summary.

# Create widgets
n_slider = widgets.IntSlider(value=base_n, min=50, max=1000, step=10, description='Samples:')
corr_slider = widgets.FloatSlider(value=base_corr, min=-1.0, max=1.0, step=0.01, description='Corr:')
noise_slider = widgets.FloatSlider(value=base_noise, min=0.1, max=5.0, step=0.1, description='Noise:')

# Output area for plot + markdown
out = widgets.Output(layout={'border': '1px solid #ddd', 'padding': '10px'})

def update_plot(n: int, corr: float, noise: float):
    """
    Update function that regenerates data and redraws plot and summary markdown.
    This function is called whenever any widget changes.
    """
    # 1) Generate dataset (depends on Cell 1's generate_dataset)
    df = generate_dataset(n=n, correlation_strength=corr, noise_scale=noise)
    
    # 2) Compute simple statistics for dynamic markdown
    pearson_r = df['x'].corr(df['y'])
    slope = np.polyfit(df['x'], df['y'], 1)[0]
    intercept = np.polyfit(df['x'], df['y'], 1)[1]
    
    # 3) Clear previous outputs and render new plot + markdown
    with out:
        clear_output(wait=True)
        # Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df['x'], df['y'], alpha=0.6, s=30)
        ax.set_title(f'Scatter plot (n={n}, corr={corr:.2f}, noise={noise:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Overlay regression line
        x_vals = np.linspace(df['x'].min(), df['x'].max(), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, linestyle='--', linewidth=2)
        plt.show()
        
        # Dynamic markdown summary (renders math and values)
        md = f"""
### Dynamic summary

- Pearson correlation (sample): **{pearson_r:.3f}**
- Fitted linear model: $\\hat{{y}} = {slope:.3f}x + {intercept:.3f}$
- Expected theoretical behavior:
  - As **corr** approaches 1, points cluster around a line (high $R$).
  - As **noise** increases, scatter increases and $R$ decreases.
"""
        display(Markdown(md))

# Link widgets to update function using `interactive_output` so layout is separate
controls = widgets.HBox([n_slider, corr_slider, noise_slider])
interactive_out = widgets.interactive_output(update_plot, {
    'n': n_slider,
    'corr': corr_slider,
    'noise': noise_slider
})

# Show the widgets and the output area
display(Markdown("## Interactive controls"))
display(controls)
display(out)

# Initialize the plot once on load
update_plot(n_slider.value, corr_slider.value, noise_slider.value)

# %% [3] Additional dependent cell: derived metrics and comments
# This cell demonstrates a downstream computation that depends on the currently generated dataset.
# Data flow: If you want to perform additional analysis on the "current" dataset shown in Cell 2,
# you should call generate_dataset(...) with the same widget parameters to ensure consistency.
#
# Example: compute RMSE using current slider values (reproducible)
current_df = generate_dataset(n=n_slider.value, correlation_strength=corr_slider.value, noise_scale=noise_slider.value)
rmse = np.sqrt(((current_df['y'] - np.polyval(np.polyfit(current_df['x'], current_df['y'], 1), current_df['x'])) ** 2).mean())
print(f"Derived metric (RMSE) for current sliders: {rmse:.3f}")

# End of notebook-style script
