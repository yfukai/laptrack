import networkx as nx
import numpy as np
import pandas as pd


def convert_dataframe_to_coords(
    df, coordinate_cols, frame_col="frame", validate_frame=True
):
    """Convert a track dataframe to a list of coordinates for input.

    Parameters
    ----------
    df : pd.DataFrame
        the input dataframe
    coordinate_cols : List[str]
        the list of columns to use for coordinates
    frame_col : str, optional
        The column name to use for the frame index. Defaults to "frame".
    validate_frame : bool, optional
        whether to validate the frame. Defaults to True.

    Returns
    -------
    coords : List[np.ndarray]
        the list of coordinates
    """

    grps = list(df.groupby(frame_col, sort=True))
    if validate_frame:
        assert np.array_equal(np.arange(df[frame_col].max() + 1), [g[0] for g in grps])
    coords = [grp[coordinate_cols].values for _frame, grp in grps]
    return coords


def convert_tree_to_dataframe(tree):
    """Convert the track tree to dataframes

    Parameters
    ----------
    tree : nx.Graph
        The track tree, resulted from the traking

    Returns
    -------
    df : pd.DataFrame
        the track dataframe, with the following columns:
        - "frame" : the frame index
        - "index" : the coordinate index
        - "track_id" : the track id
        - "tree_id" : the tree id
    split_df : pd.DataFrame
        the splitting dataframe, with the following columns:
        - "parent_track_id" : the track id of the parent
        - "child_track_id" : the track id of the parent
    merge_df : pd.DataFrame
        the splitting dataframe, with the following columns:
        - "parent_track_id" : the track id of the parent
        - "child_track_id" : the track id of the parent
    """
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
            df.loc[(frame, index), "tree_id"] = track_id
    #            tree.nodes[(frame, index)]["tree_id"] = track_id
    tree2 = tree.copy()

    splits = []
    merges = []
    for node in tree.nodes:
        frame0, _index0 = node
        neighbors = list(tree.neighbors(node))
        children = [(frame, index) for (frame, index) in neighbors if frame > frame0]
        parents = [(frame, index) for (frame, index) in neighbors if frame < frame0]
        assert len(children) + len(parents) == len(neighbors)
        if len(children) > 1:
            for child in children:
                if tree2.has_edge(node, child):
                    tree2.remove_edge(node, child)
            if node not in [p[0] for p in splits]:
                splits.append([node, children])
        if len(parents) > 1:
            for parent in parents:
                if tree2.has_edge(node, parent):
                    tree2.remove_edge(node, parent)
            if node not in [p[0] for p in merges]:
                merges.append([node, parents])

    connected_components = list(nx.connected_components(tree2))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            df.loc[(frame, index), "track_id"] = track_id
    #            tree.nodes[(frame, index)]["track_id"] = track_id

    for k in ["tree_id", "track_id"]:
        df[k] = df[k].astype(int)

    split_df_data = []
    for (node, children) in splits:
        for child in children:
            split_df_data.append(
                {
                    "parent_track_id": df.loc[node, "track_id"],
                    "child_track_id": df.loc[child, "track_id"],
                }
            )
    split_df = pd.DataFrame.from_records(split_df_data).astype(int)

    merge_df_data = []
    for (node, parents) in merges:
        for parent in parents:
            merge_df_data.append(
                {
                    "parent_track_id": df.loc[parent, "track_id"],
                    "child_track_id": df.loc[node, "track_id"],
                }
            )
    merge_df = pd.DataFrame.from_records(merge_df_data).astype(int)

    return df, split_df, merge_df
