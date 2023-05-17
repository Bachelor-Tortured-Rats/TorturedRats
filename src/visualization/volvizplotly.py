###
### Original script from: Vedrana Andersen Dahl
###

import numpy as np
import plotly.graph_objects as go


# MAIN FUNCTIONS

def volume_slicer(vol, slices, 
                cmin=None, cmax=None, colorscale='Gray', show_scale=False,
                fig=None, show=True, title = '', width=600, height=600):
    ''' Visualizes chosen slices from volume.
        
        vol: a 3D numpy array. 
        slices: list of 3 lists containing slice indices in three directions.
           Alternatively, each of the three elements may be one of the strings
           'mid', 'ends', 'first', 'last' or None (None equals empty).
        min, cmax, colorscale and showscale: values passed to plotly 
            surface colormap.
        fig: plotly figure, if None new figure will be created.
        show: whether to show figure, if not figure will be returned.
        title, width, height: values passed to ploty layout.

    '''
  
    if cmin is None:
        cmin = vol.min()
    if cmax is None:
        cmax = vol.max()
      
    dim = vol.shape

    for i in range(len(slices)):
        sl = slices[i]
        if type(sl) is not list:
            if sl=='mid':
                slices[i] = [dim[i]//2]
            elif sl=='ends':
                slices[i] = [0, dim[i]-1]
            elif sl=='first':
                slices[i] = [0]
            elif sl=='last':
                slices[i] = [dim[i]-1]
            elif type(sl) is int:
                slices[i] = [sl]
            else: # inclusiv None
                slices[i] = []

    surfs = []
    for i in range(3): # three directions
        gi = grids(dim,i)
        for j in range(len(slices[i])):
            g = gi.copy()
            s = slices[i][j]
            g[i] *= s
            surf = dict(x=g[2], y=g[1], z=g[0], surfacecolor=volslice(vol, i, s))
            surfs.append(surf)

    common = dict(colorscale=colorscale, cmin=cmin, cmax=cmax, 
                  showscale=show_scale)
    surfaces = [go.Surface({**s, **common}) for s in surfs]

    # Set limits and aspect ratio.
    d = max(dim)
    scene = dict(xaxis = dict(range=[-1, dim[2]], autorange=False),
            yaxis = dict(range=[-1, dim[1]], autorange=False),
            zaxis = dict(range=[-1, dim[0]], autorange=False), 
            aspectratio = dict(x=dim[2]/d, y=dim[1]/d, z=dim[0]/d))
    layout = dict(title=title, width=width, height=height, scene=scene)

    if fig is None:
        fig = go.Figure()
    
    fig.add_traces(surfaces)
    fig.update_layout(layout)  
    
    if show:
        fig.show()
        return
    else:
        return fig


def show_mesh(vertices, faces, **options):
    ''' Show triangle surface mesh in 3d.
        
        vertices and faces: mesh entities as n x 3 numpy arrays
        fig: plotly figure, if None new figure will be created.
        show: whether to show figure, if not figure will be returned.
        add_wireframe and add_surf: flags for showing wireframe and surface. 
            If neither wireframe or surf are chosen, pointcloud is shown. 
        title, width, height: values passed to ploty layout.
        Exeptionally, if only wireframe is requestet, also tet meshes may
        be processed.
        Other arguments are:
            surface_color, surface_opacity,
            wireframe_color, wireframe_width,
            points_color, points_opacity, points_size. 

        TODO: add options for layout not being updated, e.g. when the figure
        already has title, it should not be changed to default empty.    
    '''

    fig = options.get('fig')  # defaults to None
    show = options.get('show', True)
    add_wireframe = options.get('add_wireframe', True)
    add_surface = options.get('add_surface', True)
    surface_color = options.get('surface_color', 'rgb(0,0,255)')
    surface_opacity = options.get('surface_opacity', 1)
    wireframe_color = options.get('wireframe_color', 'rgb(40,40,40)')
    wireframe_opacity = options.get('wireframe_opacity', 1)
    wireframe_width = options.get('wireframe_width', 1)
    points_color = options.get('points_color', 'rgb(0,0,255)')
    points_opacity = options.get('points_opacity', 1)
    points_size = options.get('points_size', 1)
    figure_title = options.get('figure_title', '')
    figure_width = options.get('figure_width', 600)
    figure_height = options.get('figure_height', 600)
    show_legend = options.get('show_legend', False)

    if fig is None:
        fig = go.Figure()
  
    if add_surface and (faces is not None): 
        fig.add_trace(mesh_surface_plot(vertices, faces, 
                surface_color, surface_opacity))
  
    if add_wireframe and (faces is not None):        
        fig.add_trace(mesh_wireframe_plot(vertices, faces, 
                wireframe_color, wireframe_opacity, wireframe_width))
  
    if ((not add_surface) and (not add_wireframe)) or (faces is None):
        fig.add_trace(pointcloud_plot(vertices,
                points_color, points_opacity, points_size))
    
    fig.update_layout(title_text = figure_title, height = figure_height, 
                      width = figure_width, showlegend = show_legend)
    
    if show:
        fig.show()
        return
    else:
        return fig
    

# Helping functions

def grids(dim, i):
    ''' Returns matrices with coordinates for slicing along axis i. '''
    two = dim[:i] + dim[i+1:]
    out = np.mgrid[0:two[0], 0:two[1]]
    out = np.insert(out, i, np.ones(two), axis=0)
    return out


def volslice(vol, i, s):
    ''' Returns volume slice s along axis i.'''
    s_xyz = (slice(None),) * i + (slice(s,s+1),)
    return vol[s_xyz].squeeze(axis=i)

def pointcloud_plot(points, color, opacity, size):

    gm = go.Scatter3d(z=points[:,0], y=points[:,1], x=points[:,2], 
            mode='markers', name='', opacity=opacity,
            marker=dict(color=color, size=size)) 
    return gm 


def mesh_wireframe_plot(vertices, faces, color, opacity, width):

    if faces.shape[1]==3:
        links = [(0, 1), (1, 2), (2, 0)]
    elif faces.shape[1]==4:  # only for wireframe, support for tet meshes
        links = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    lines = np.vstack([faces[:, [l[0], l[1]]] for l in links])
    lines = np.sort(lines, axis=1)
    lines = np.unique(lines, axis=0)

    n = lines.shape[0]
    Xe = np.hstack((vertices[lines, 0], np.full((n, 1), None))).ravel()
    Ye = np.hstack((vertices[lines, 1], np.full((n, 1), None))).ravel()
    Ze = np.hstack((vertices[lines, 2], np.full((n, 1), None))).ravel()
    
    gm = go.Scatter3d(z=Xe, y=Ye, x=Ze, mode='lines', name='', opacity=opacity,
            line=dict(color=color, width=width))  
    
    return gm


def mesh_surface_plot(vertices, faces, color, opacity):

    gm = go.Mesh3d(z=vertices[:,0], y=vertices[:,1], x=vertices[:,2], 
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            color=color, opacity=opacity)
    return gm

   
# EXPERIMENTAL FUCNTIONS

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

    fig.show()