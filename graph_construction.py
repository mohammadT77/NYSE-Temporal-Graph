# %%
# !pip install -q kagglehub numpy pandas networkx matplotlib
# !python -m pip install -q pygraphviz

# %%
# %cd NYSE-Temporal-Graph-Construction

# %%
import kagglehub
import kagglehub.datasets
import numpy as np
import pandas as pd
import networkx as nx
from os.path import join as join_path
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
SIMILARITY_THRESHOLD = 2
TARGET_SECTORS = ['Real Estate', 'Information Technology', 'Materials', 'Telecommunications Services']
DIR_NAME = 'rimt_thresh2'

# %% [markdown]
# # Download Dataset

# %%
dataset_path = kagglehub.dataset_download("dgawlik/nyse")

# %%
fundamentals_pd = pd.read_csv(join_path(dataset_path,"fundamentals.csv"))
prices_pd = pd.read_csv(join_path(dataset_path, "prices.csv"))
prices_split_adjusted_pd = pd.read_csv(join_path(dataset_path, "prices-split-adjusted.csv"))
securities_pd = pd.read_csv(join_path(dataset_path, "securities.csv"))

# %%
all_symbols = prices_pd['symbol'].unique()

def get_symbol_by_sector(sector):
    return securities_pd[securities_pd['GICS Sector'] == sector]['Ticker symbol'].to_list()

securities_pd.groupby('GICS Sector')['Ticker symbol'].count()

# %%
prices_pd['date'] = pd.to_datetime(prices_pd['date'], format='mixed').dt.date
dates = prices_pd['date'].sort_values().unique()


def any_to_date(date):
    if not isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
        date = pd.to_datetime(date, format='mixed').date()
    return date


def date_to_int(date):
    date = any_to_date(date)
    return dates.tolist().index(date)

def int_to_date(idx):
    return dates[idx]

print(dates)
print(int_to_date(0))
print(date_to_int(int_to_date(len(dates)-1)))

# %%
def add_features(target_prices_df = prices_pd):
    target_prices_df['day_diff'] = ((target_prices_df['close']) - (o:=target_prices_df['open'])) / o
    # target_prices_df['close_1d'] = target_prices_df['close'].pct_change(1)
    # target_prices_df['close_3d'] = target_prices_df['close'].pct_change(3)


def similarity_score(record1: pd.Series, record2: pd.Series) -> float:
    norm_factor = 1e-5
    
    if record1['symbol'] == record2['symbol']:
        return 0
    
    abs_diff = abs((record2['day_diff'])-(record1['day_diff'])) + norm_factor
    sim = -np.log(abs_diff)
    if sim > SIMILARITY_THRESHOLD:
        return np.round(sim, 3)
    else:
        return 0
        

def build_temporal_graphs(target_prices_df: pd.DataFrame, similarity_score = similarity_score):
    add_features(target_prices_df)
    target_prices_df.fillna(0, inplace=True)
    target_prices_df.sort_values(by=['date'], inplace=True)

    for date, group in target_prices_df.groupby('date'):
        group.reset_index(inplace=True)
        graph = nx.Graph()
        dateIdx = date_to_int(date)
        for i, record1 in group.iterrows():
            group['sim_score'] = group.apply(lambda record2: similarity_score(record1, record2), axis=1)

            edges = group[group['sim_score'] > 0]
            
            for j, edge_row in edges.iterrows():
                graph.add_edge(record1['symbol'], edge_row['symbol'], weight=edge_row['sim_score'], time=dateIdx)

        yield dateIdx, date, graph


def save_graph(prefix: str, dateIdx: int, date, graph: nx.Graph, base_dir='graph_data/train'):
    dir = join_path(base_dir, prefix)

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    file_name = f"{dateIdx}_{date}.edgelist"
    file_path = join_path(dir, file_name)
    nx.write_weighted_edgelist(graph, file_path)

# %%
target_symbols = []
for sector in TARGET_SECTORS:
    target_symbols += get_symbol_by_sector(sector)

train_end_date_idx = int(len(dates)*0.7)
test_prices_df = prices_pd[prices_pd['date'] >= int_to_date(0)][prices_pd['date'] < int_to_date(train_end_date_idx)]

target_graphs = build_temporal_graphs(test_prices_df[test_prices_df['symbol'].isin(target_symbols)])

pbar = tqdm(range(train_end_date_idx))
for dateIdx, date, graph in target_graphs:
    save_graph(DIR_NAME, dateIdx, date, graph)
    pbar.update(1)
    


