# analysis.py
# Interactive Marimo Notebook
# Author Email: 22f3002140@ds.study.iitm.ac.in
# Requirements satisfied:
# - Interactive slider widget
# - Variable dependencies across multiple cells
# - Dynamic markdown output
# - Comments explaining data flow

import marimo

app = marimo.App()

# ---------------------------------------------------
# Cell 1: Generate dataset (feeds later cells)
# ---------------------------------------------------
@app.cell
def _(pd=__import__("pandas"), np=__import__("numpy")):
    # Create synthetic dataset
    df = pd.DataFrame({
        "x": np.arange(1, 101),
        "y": np.random.normal(loc=50, scale=10, size=100),
    })
    return df

# ---------------------------------------------------
# Cell 2: Interactive slider widget
# Controls how many rows are displayed
# ---------------------------------------------------
@app.cell
def _(mo=marimo):
    slider = mo.ui.slider(
        start=5,
        stop=50,
        value=10,
        label="Select number of rows"
    )
    slider   # display
    return slider

# ---------------------------------------------------
# Cell 3: Dependent data view
# Uses df from cell 1 and slider from cell 2
# ---------------------------------------------------
@app.cell
def _(df, slider):
    sliced_df = df.head(slider.value)
    sliced_df
    return sliced_df

# ---------------------------------------------------
# Cell 4: Dynamic Markdown output
# Responds to slider value
# ---------------------------------------------------
@app.cell
def _(slider, mo=marimo):
    dynamic_message = f"""
### 🔍 Interactive Analysis  
Displaying **{slider.value}** rows  
(adjust using the slider above)
"""
    mo.md(dynamic_message)
    return dynamic_message

# ---------------------------------------------------
# Cell 5: Summary statistics (depends on df)
# ---------------------------------------------------
@app.cell
def _(df):
    stats = df.describe()
    stats
    return stats

# ---------------------------------------------------
# Launch app
# ---------------------------------------------------
if __name__ == "__main__":
    app.run()
