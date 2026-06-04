def color_print(msg, bg_color=None, **kwargs):
    print(msg)


def read_csv_tqdm(path, **kwargs):
    import pandas as pd
    return pd.read_csv(path, **kwargs)
