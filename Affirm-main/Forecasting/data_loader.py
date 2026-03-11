import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

def _normalize_pandas_freq(freq):
    if not freq:
        return None
    freq = str(freq)
    freq_map = {
        "s": "S",
        "t": "T",
        "h": "H",
        "d": "D",
        "b": "B",
        "w": "W",
        "m": "M",
    }
    return freq_map.get(freq, freq)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, args, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.flag = flag
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.patch_size = args.patch_size

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data = data.astype(np.float32, copy=False)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.astype(np.float32, copy=False)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if hasattr(self, "_index_map") and self._index_map:
            s_begin = self._index_map[index]
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # seq_x = zero_pad_sequence(torch.from_numpy(seq_x).permute(1,0), self.pad_len).permute(1,0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if hasattr(self, "_index_map") and self._index_map:
            return len(self._index_map)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, args, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data = data.astype(np.float32, copy=False)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.astype(np.float32, copy=False)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if hasattr(self, "_index_map") and self._index_map:
            s_begin = self._index_map[index]
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if hasattr(self, "_index_map") and self._index_map:
            return len(self._index_map)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None,
                 gap_interp=False, gap_days=3, interp_limit=3, interp_freq="",
                 window_split=False, train_ratio=0.8, val_ratio=0.0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.gap_interp = gap_interp
        self.gap_days = gap_days
        self.interp_limit = interp_limit
        self.interp_freq = interp_freq
        self.window_split = window_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        date_col = None
        if 'date' in df_raw.columns:
            date_col = 'date'
        else:
            for col in df_raw.columns:
                if col.lower() == 'date':
                    date_col = col
                    break
        if date_col is None:
            raise ValueError("No date column found. Expected a column named 'date' (case-insensitive).")

        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found. Available: {df_raw.columns.tolist()}")

        if not pd.api.types.is_numeric_dtype(df_raw[self.target]):
            df_raw[self.target] = pd.to_numeric(df_raw[self.target], errors='coerce')
            if df_raw[self.target].isna().all():
                raise ValueError(f"Target column '{self.target}' is not numeric.")

        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw = df_raw.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

        numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        exclude_numeric = {"location_id"}
        numeric_cols = [c for c in numeric_cols if c not in exclude_numeric]
        if self.target not in numeric_cols:
            raise ValueError(f"Target column '{self.target}' must be numeric.")

        if self.features in ['M', 'MS']:
            cols = [c for c in numeric_cols if c != self.target]
        else:
            cols = [c for c in df_raw.columns if c not in [date_col, self.target]]
        df_raw = df_raw[[date_col] + cols + [self.target]]
        interp_cols = [c for c in df_raw.columns if c != date_col]

        if self.gap_interp:
            freq = _normalize_pandas_freq(self.interp_freq) or _normalize_pandas_freq(self.freq) or "D"
            diffs = df_raw[date_col].diff().dt.days.fillna(1).astype(int)
            gap_idx = np.where(diffs.values[1:] > self.gap_days)[0] + 1
            segments = []
            start = 0
            for idx in gap_idx:
                segments.append(df_raw.iloc[start:idx].copy())
                start = idx
            segments.append(df_raw.iloc[start:].copy())

            filled_segments = []
            for seg in segments:
                if seg.empty:
                    continue
                diffs_seg = seg[date_col].diff().dt.days.fillna(1).astype(int)
                if diffs_seg.max() <= self.gap_days:
                    full_dates = pd.date_range(seg[date_col].min(), seg[date_col].max(), freq=freq)
                    seg = seg.set_index(date_col).reindex(full_dates)
                    seg.index.name = date_col
                    seg[interp_cols] = seg[interp_cols].interpolate(
                        limit=self.interp_limit, limit_direction='both'
                    )
                    seg = seg.reset_index()
                filled_segments.append(seg)
            if filled_segments:
                df_raw = pd.concat(filled_segments, ignore_index=True)

        self.raw_target = df_raw[self.target].values.astype(np.float32)
        all_index_map = []

        # Build valid windows by filtering NaNs (FreEformer-style)
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        data_vals = df_data.values.astype(np.float32, copy=False)
        valid_row = ~np.isnan(data_vals).any(axis=1)
        valid_window = np.convolve(
            valid_row.astype(int),
            np.ones(self.seq_len + self.pred_len, dtype=int),
            mode='valid'
        ) == (self.seq_len + self.pred_len)
        all_index_map = np.where(valid_window)[0].tolist()
        max_start = len(df_raw) - self.seq_len - self.pred_len
        if max_start < 0:
            max_start = -1
        all_index_map = [i for i in all_index_map if 0 <= i <= max_start]

        if not self.window_split:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1 = 0
            border2 = len(df_raw)

        if self.scale:
            data_for_scale = data_vals.copy()
            data_for_scale[np.isnan(data_for_scale)] = 0.0
            if self.window_split:
                total = len(all_index_map)
                train_end = int(total * self.train_ratio)
                train_starts = all_index_map[:train_end]
                train_rows = set()
                for s in train_starts:
                    for r in range(s, min(s + self.seq_len, len(df_data))):
                        train_rows.add(r)
                train_rows = sorted(train_rows)
                if not train_rows:
                    train_rows = list(range(len(df_data)))
                train_data = data_for_scale[train_rows]
            else:
                train_data = data_for_scale[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_for_scale)
        else:
            data = data_vals
        data = data.astype(np.float32, copy=False)

        df_stamp = df_raw[[date_col]].copy()
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[date_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[date_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[date_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[date_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.astype(np.float32, copy=False)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        self.data_dates = df_raw[date_col].reset_index(drop=True)

        # Final safety: drop any start index that would create empty/short sequences
        if hasattr(self, "_index_map") and self._index_map is not None:
            max_start = len(self.data_x) - self.seq_len - self.pred_len
            if max_start < 0:
                self._index_map = []
            else:
                self._index_map = [i for i in self._index_map if 0 <= i <= max_start]

        total = len(all_index_map)
        train_end = int(total * self.train_ratio)
        val_end = int(total * (self.train_ratio + self.val_ratio))
        if self.window_split:
            if self.flag == 'train':
                self._index_map = all_index_map[:train_end]
            elif self.flag == 'val':
                if self.val_ratio > 0:
                    self._index_map = all_index_map[train_end:val_end]
                else:
                    self._index_map = all_index_map[:train_end]
            else:
                if self.val_ratio > 0:
                    self._index_map = all_index_map[val_end:]
                else:
                    self._index_map = all_index_map[train_end:]
        else:
            max_start = border2 - self.seq_len - self.pred_len + 1
            self._index_map = [i for i in all_index_map if border1 <= i < max_start]

    def __getitem__(self, index):
        if hasattr(self, "_index_map") and self._index_map is not None:
            s_begin = self._index_map[index]
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if seq_x.shape[0] != self.seq_len or seq_y.shape[0] != (self.label_len + self.pred_len):
            raise ValueError(
                f"Bad window length at index {index}: "
                f"s_begin={s_begin}, s_end={s_end}, r_end={r_end}, "
                f"seq_x={seq_x.shape}, seq_y={seq_y.shape}, "
                f"data_len={len(self.data_x)}"
            )

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if hasattr(self, "_index_map") and self._index_map is not None:
            return len(self._index_map)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
