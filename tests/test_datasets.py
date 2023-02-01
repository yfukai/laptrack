from laptrack import datasets


def test_fetch() -> None:
    for k in datasets.TEST_DATA_PROPS.keys():
        datasets.fetch(k)
    df = datasets.simple_tracks()
    assert set(df.keys()) == {"frame", "position_x", "position_y"}
    im = datasets.bright_brownian_particles()
    assert im.ndim == 3
    im, label = datasets.cell_segmentation()
    assert im.ndim == 3
    assert label.ndim == 3
    label = datasets.mouse_epidermis()
    assert label.ndim == 3
