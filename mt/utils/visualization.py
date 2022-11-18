import matplotlib.pyplot as plt
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def set_matplotlib():
    """
    Sets matplotlib to default settings to look the same every time
    """
    plt.style.use('seaborn')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "arial",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9
    }
    plt.rcParams.update(tex_fonts)


def set_fig_size(width_pt, fraction=1, subplots=(1, 1)):
    """
    Calculates figure size to get consistent text size across the document
    """
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)
