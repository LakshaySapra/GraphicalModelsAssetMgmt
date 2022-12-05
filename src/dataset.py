import numpy as np
import pandas as pd
import torch
from itertools import combinations
from torch.utils.data import Dataset

def get_mapper(mapper_df,date,permno=None,gvkey=None):
    # Map between permno and gvkey
    mapper = mapper_df[mapper_df['date']<= date]
    mapper = mapper[mapper['date'] == mapper['date'].max()].drop('date',axis=1)
    assert len(mapper) > 0
    if permno is not None:
        mapper = mapper.set_index('permno')
        return mapper.loc[permno,'gvkey'].astype(int).tolist()
    else:
        assert gvkey is not None
        mapper = mapper.set_index('gvkey')
        return mapper.loc[gvkey,'permno'].astype(int).tolist()

def get_permnogvkey_toidx_mapper(mapper_df,date,permno=None,gvkey=None):
    # Map permno/gvkey to idx
    assert (permno is None) or (gvkey is None)
    mapper = mapper_df[mapper_df['date']<= date]
    mapper = mapper[mapper['date'] == mapper['date'].max()].drop('date',axis=1).reset_index(drop=True)
    mapper['index'] = mapper.index
    assert len(mapper) > 0
    if permno is not None:
        mapper = mapper.set_index('permno',drop=True)
        return mapper.loc[permno,'index'].tolist()
    else:
        assert gvkey is not None
        mapper = mapper.set_index('gvkey',drop=True)
        return mapper.loc[gvkey,'index'].tolist()

def get_idxto_permnogvkey_mapper(mapper_df,date,idx,permno=None,gvkey=None):
    # Map idx to permno/gvkey
    assert (permno and not gvkey) or (gvkey and not permno)
    mapper = mapper_df[mapper_df['date']<= date]
    mapper = mapper[mapper['date'] == mapper['date'].max()].drop('date',axis=1).reset_index(drop=True)
    assert len(mapper) > 0
    if permno:
        return mapper.loc[idx,'permno'].tolist()
    else:
        return mapper.loc[idx,'gvkey'].tolist()

def get_gvkey_universe(universe,date):
    cur_universe = universe[universe.index<= date]
    cur_universe = cur_universe.loc[cur_universe.index.max()]
    return cur_universe.dropna().astype(int).tolist()
    
def get_permno_universe(permno_universe,date):
    cur_universe = permno_universe[permno_universe.index<= date]
    cur_universe = cur_universe.loc[cur_universe.index.max()]
    return cur_universe.dropna().astype(int).tolist()

def get_sector_edges(sector_df,all_sectors,date,gvkey_to_idx_mapper,mapper_df,gvkey_universe):
    cur_sector_df = sector_df[sector_df['datadate']<= date]
    cur_sector_df = cur_sector_df[cur_sector_df['datadate'] == cur_sector_df['datadate'].max()].drop('datadate',axis=1)
    cur_sector_df = cur_sector_df[cur_sector_df['GVKEY'].isin(gvkey_universe)]
    total = 0
    edge_lst = []
    for sector in all_sectors:
        gvkeys = cur_sector_df.loc[cur_sector_df['gsector']==sector,'GVKEY'].unique()
        gvkeys = gvkey_to_idx_mapper(mapper_df,date,gvkey=gvkeys)
        pairs = list(combinations(gvkeys,2))
        total += len(pairs)
        edge_lst.extend(pairs)
    # print(total)
    return np.array(edge_lst)

def get_supply_chain_edges(supchain_df,date,gvkey_to_idx_mapper,mapper_df,gvkey_universe):
    cur_df = supchain_df[(supchain_df['srcdate'] <= date) & (supchain_df['srcdate'] >= date -pd.DateOffset(years=3)) ]
    cur_df = cur_df[cur_df['gvkey'].isin(gvkey_universe) & cur_df['cgvkey'].isin(gvkey_universe)]
    c_edge_list = cur_df[['gvkey','cgvkey']].rename({'gvkey':'sup','cgvkey':'con'},axis=1).apply(lambda x: gvkey_to_idx_mapper(mapper_df,date,gvkey=x),axis=0)
    s_edge_list = c_edge_list[['con','sup']].rename({'sup':'con','con':'sup'},axis=1)
    
    return c_edge_list.values,s_edge_list.values

def get_hist_ret_df(weekly_hist_ret_df,date,gvkey_to_idx_mapper,permno_to_gvkey_mapper,mapper_df,gvkey_universe,permno_universe):
    cur_df = weekly_hist_ret_df.loc[date]
    cur_df = cur_df[cur_df.index.isin(permno_universe)]
    cur_df.index = permno_to_gvkey_mapper(mapper_df,date,permno=cur_df.index)
    cur_df = cur_df[cur_df.index.isin(gvkey_universe)]
    cur_df.index = gvkey_to_idx_mapper(mapper_df,date,gvkey=cur_df.index)
    cur_df = cur_df.reindex(np.arange(500))
    return cur_df.values

def get_weekly_ret_df(weekly_ret_df,date,gvkey_to_idx_mapper,permno_to_gvkey_mapper,mapper_df,gvkey_universe,permno_universe):
    cur_df = weekly_ret_df.loc[date]
    cur_df = cur_df[cur_df.index.isin(permno_universe)]
    cur_df.index = permno_to_gvkey_mapper(mapper_df,date,permno=cur_df.index)
    cur_df = cur_df[cur_df.index.isin(gvkey_universe)]
    cur_df.index = gvkey_to_idx_mapper(mapper_df,date,gvkey=cur_df.index)
    cur_df = cur_df.reindex(np.arange(500))
    return cur_df.values

class GNNDataset(Dataset):
    def __init__(self, data_path='../data/',device='cpu'):
        super().__init__()
        self.universe = pd.read_pickle(data_path+'universe.pkl')
        self.universe.index = pd.to_datetime(self.universe.index)
        self.permno_universe = pd.read_pickle(data_path+'permno_universe.pkl')
        self.permno_universe.index = pd.to_datetime(self.permno_universe.index)
        self.mapper_df = pd.read_pickle(data_path+'mapper_df.pkl')
        self.hist_ret_df = pd.read_pickle(data_path+'hist_ret_df.pkl.gz')
        self.weekly_ret_df = pd.read_pickle(data_path+'weekly_ret_df.pkl')
        self.sector_df = pd.read_pickle(data_path+'sector_df.pkl')
        self.all_sectors = self.sector_df['gsector'].dropna().astype(int).unique()
        self.supchain_df = pd.read_pickle(data_path+'supchain_df.pkl')
        self.date_lst = list(self.weekly_ret_df.loc['2012-01-01':'2022-01-01'].index.unique())
        concat_lst = []
        for date in self.date_lst:
            cur_hist_ret_df = self.hist_ret_df[self.hist_ret_df.index.get_level_values(0)<= date]
            cur_hist_ret_df = cur_hist_ret_df[cur_hist_ret_df.index.get_level_values(0) == cur_hist_ret_df.index.get_level_values(0).max()].droplevel(0,axis=0)
            concat_lst.append(cur_hist_ret_df.stack())
        self.weekly_hist_ret_df = pd.DataFrame(concat_lst,index=self.date_lst).stack(level=0)
        self.device=device
    
    def __len__(self):
        return len(self.date_lst)-1

    def __getitem__(self,idx):
        cur_date = self.date_lst[idx]
        next_date = self.date_lst[idx+1]
        cur_universe = get_gvkey_universe(self.universe,cur_date)
        cur_permno_universe = get_permno_universe(self.permno_universe,cur_date)

        # Sector edges
        sector_edge_lst = get_sector_edges(self.sector_df,self.all_sectors,cur_date,get_permnogvkey_toidx_mapper,self.mapper_df,cur_universe)
        # Consumer and supplier edges
        c_edge_lst, s_edge_lst = get_supply_chain_edges(self.supchain_df,cur_date,get_permnogvkey_toidx_mapper,self.mapper_df,cur_universe)

        # Individual stock features
        cur_hist_ret_df = get_hist_ret_df(self.weekly_hist_ret_df,cur_date,get_permnogvkey_toidx_mapper,get_mapper,self.mapper_df,cur_universe,cur_permno_universe)
        # Labels
        cur_weekly_ret_df = get_weekly_ret_df(self.weekly_ret_df,next_date,get_permnogvkey_toidx_mapper,get_mapper,self.mapper_df,cur_universe,cur_permno_universe)

        mask = (np.isnan(cur_hist_ret_df).any(axis=1)) | (np.isnan(cur_weekly_ret_df))
        mask = ~mask

        return torch.Tensor(sector_edge_lst).to(self.device),\
               torch.Tensor(c_edge_lst).to(self.device),s_edge_lst,\
               torch.Tensor(cur_hist_ret_df).to(self.device),\
               torch.Tensor(cur_weekly_ret_df).to(self.device),\
               torch.Tensor(mask).to(self.device)