###
### Original script from: Vedrana Andersen Dahl
###

# Load packages.
import numpy as np
import plotly.graph_objects as go
import numpy as np
import skimage.measure
import nibabel as nib
import volvizplotly as vvp

img = nib.load('/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/imagesTr/hepaticvessel_002.nii.gz')
lab_true = nib.load('/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/labelsTr/hepaticvessel_002.nii.gz')
lab_pred = nib.load('reports/save_prediction_mask/hepaticvessel_002/hepaticvessel_002_seg.nii.gz')


vol = np.array(img.get_fdata()).transpose((2, 1, 0))
seg_true = np.array(lab_true.get_fdata()).transpose((2, 1, 0))
seg_pred = np.array(lab_pred.get_fdata()).transpose((2, 1, 0))

# Cropping to xy-bounding box of labels
# bb = seg.any(axis=0)
# x = np.where(bb.any(axis=0))[0]
# y = np.where(bb.any(axis=1))[0]
# vol = vol[:, y[0]: y[-1], x[0]: x[-1]]
# seg = seg[:, y[0]: y[-1], x[0]: x[-1]]
def interactive_volume_slicer(vol, cmin=None, cmax=None, colorscale='Gray',
                  title = '', width=600, height=600):
    '''
    This function is greatly inspired by (basicaly copied from)  
    https://plotly.com/python/visualizing-mri-volume-slices/.
    TODO: input to choose slicing direction
    '''
    if cmin is None:
        cmin = vol.min()
    if cmax is None:
        cmax = vol.max()
    Z, Y, X = vol.shape

    o = np.ones((Y, X))
    surfaces = [go.Surface(
        z = z * o,
        surfacecolor = vol[z],
        colorscale = colorscale,
        cmin=cmin, cmax=cmax)
        for z in range(Z)]

    # you need to name the frame for the animation to behave properly
    frames = [go.Frame(data=surface, name=str(z)) 
        for z, surface in enumerate(surfaces)]

    # Define frames for animation
    fig = go.Figure(frames = frames)

    # Add surface to be displayed before animation starts
    fig.add_trace(surfaces[0])

    frame_args = {
                "frame": {"duration": 0},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": 0, "easing": "linear"},
            }

    # Set slicer placement and settings.
    slider_dict = {
                    "pad": {"t": 100},
                    "len": 0.8,
                    "x": 0.2,
                    "y": 0,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "steps": [
                        {
                            "args": [[f.name], frame_args],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in fig.frames
                    ],
                }

    # Set button placement and settings.
    buttons_dict = {
                    "pad": {"t": 100},
                    "yanchor": "middle",
                    "xanchor": "right",
                    "x": 0.1,
                    "y": 0,
                    "type": "buttons",
                    "direction": "left", # make buttons left-rigth
                    "buttons": [
                        {
                            "args": [None, frame_args],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {   # note [] around None, this makes it a Pause button!!!
                            "args": [[None], frame_args],  
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                }

    d = max(X, Y, Z)
    scene_dict = {
        "xaxis": dict(range=[-1, X], autorange=False),
        "yaxis": dict(range=[-1, Y], autorange=False),    
        "zaxis": dict(range=[-1, Z], autorange=False),
        "aspectratio": dict(x=X/d, y=Y/d, z=Z/d),
    }

    # Layout
    fig.update_layout(
            title=title,
            width=width, height=height,
            scene=scene_dict,
            updatemenus = [buttons_dict],
            sliders = [slider_dict]
    )

    fig.write_html('test.html')
    
interactive_volume_slicer(vol)