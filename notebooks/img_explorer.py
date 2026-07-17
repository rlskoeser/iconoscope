import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(
        filetypes=[".parquet"], initial_path=".", selection_mode="file", multiple=False
    )
    file_browser
    return (file_browser,)


@app.cell
def _(df, mo):
    slider = mo.ui.slider(
        start=10,
        stop=df.height,
        label="Number of images",
        value=min(30, df.height),
        show_value=True,
        include_input=True,
    )
    slider
    return (slider,)


@app.cell
def _(file_browser, pl):
    df = pl.DataFrame()
    if file_browser.value:
        df = pl.read_parquet(file_browser.path(index=0))
    return (df,)


@app.cell
def _(df, mo, slider):
    imgs = []
    if not df.is_empty():
        imgs = [
            mo.image(row["image_path"], height=150)
            for row in df.sample(slider.value).iter_rows(named=True)
        ]
    # imgs
    mo.hstack(imgs, wrap=True)
    return


if __name__ == "__main__":
    app.run()
