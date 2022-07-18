import networkx as nx
import numpy as np
import pandas as pd


def convert_dataframe_to_coords(
    df, coordinate_cols, frame_col="frame", validate_frame=True
):
    grps = list(df.groupby(frame_col, sort=True))
    if validate_frame:
        assert np.array_equal(np.arange(df[frame_col].max() + 1), [g[0] for g in grps])
    coords = [grp[coordinate_cols].values for _frame, grp in grps]
    return coords


def convert_tree_to_dataframe(tree):
    df_data = []
    node_values = np.array(list(tree.nodes))
    frames = np.unique(node_values[:, 0])
    for frame in frames:
        indices = node_values[node_values[:, 0] == frame, 1]
        df_data.append(
            pd.DataFrame(
                {
                    "frame": [frame] * len(indices),
                    "index": indices,
                }
            )
        )
    df = pd.concat(df_data).set_index(["frame", "index"])
    connected_components = list(nx.connected_components(tree))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            df.loc[(frame, index), "clone_id"] = track_id
            tree.nodes[(frame, index)]["clone_id"] = track_id
    tree2 = tree.copy()
    for node in tree.nodes:
        frame0, index0 = node
        neighbors = list(tree.neighbors(node))
        children = [(frame, index) for (frame, index) in neighbors if frame > frame0]
        parents = [(frame, index) for (frame, index) in neighbors if frame < frame0]
        assert len(children) + len(parents) == len(neighbors)
        if len(children) > 1:
            for child in children:
                tree2.remove_edge(node, child)
        if len(parents) > 1:
            for parent in parents:
                tree2.remove_edge(node, parent)

    connected_components = list(nx.connected_components(tree2))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            df.loc[(frame, index), "cell_id"] = track_id
            tree.nodes[(frame, index)]["cell_id"] = track_id

    for k in ["clone_id", "cell_id"]:
        df[k] = df[k].astype(int)
    return df, None, None
