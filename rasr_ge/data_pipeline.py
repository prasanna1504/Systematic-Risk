import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from tqdm import tqdm
from datetime import datetime
from nifty50_tickers import NIFTY50_TICKERS

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

START_DATE = config['data']['start_date']
END_DATE = config['data']['end_date']
WAVELET = config['wavelet']['wavelet_name']
WAVELET_LEVEL = config['wavelet']['decomposition_level']
CORR_WINDOW = config['graph']['window']
CORR_THRESHOLD = config['graph']['edge_threshold']

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
GRAPHS_DIR = "data/graphs"
LABELS_DIR = "data/labels"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

def step1_download_data():
    print("Step 1.1: Downloading NIFTY 50 OHLCV data...")
    all_data = []
    
    for ticker in tqdm(NIFTY50_TICKERS):
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            time.sleep(0.5) # respect rate limits
            df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
            if df.empty:
                print(f"Warning: No data for {ticker}")
                continue
            
            # yfinance creates multi-index columns in recent versions if we aren't careful, ensure single index
            if isinstance(df.columns, pd.MultiIndex):
                # get only the columns for this ticker, drop ticker level
                df.columns = df.columns.droplevel(1)
                
            df.to_csv(csv_path)
            
        df = df.reindex(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        # Give column a pre-fix for uniqueness later if needed, but not required yet
        all_data.append((ticker, df))
        
    return all_data

def align_and_clean_data(all_data):
    print("Aligning data and cleaning NaNs...")
    close_df = pd.DataFrame()
    
    for ticker, df in all_data:
        close_df[ticker] = df['Close']
    
    # Forward fill then backward fill
    close_df.ffill(inplace=True)
    close_df.bfill(inplace=True)
    
    # Filter tickers that are all NaN
    valid_tickers = close_df.columns[close_df.notna().any()].tolist()
    if len(valid_tickers) < len(NIFTY50_TICKERS):
        print(f"Dropped {len(NIFTY50_TICKERS) - len(valid_tickers)} invalid tickers.")
    
    # Create aligned tensors
    dates = close_df.index
    aligned_data = {}
    
    for ticker, df in all_data:
        if ticker in valid_tickers:
            df_aligned = df.reindex(dates).ffill().bfill()
            aligned_data[ticker] = df_aligned
            
    return aligned_data, dates, valid_tickers

def step2_feature_engineering(aligned_data, valid_tickers, dates):
    print("Step 1.2: Feature Engineering...")
    # shape: (N_stocks, T_days, 5)
    N = len(valid_tickers)
    T = len(dates)
    F = 5
    
    features = np.zeros((N, T, F))
    
    for i, ticker in enumerate(valid_tickers):
        df = aligned_data[ticker]
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # 1. log_return
        log_ret = np.zeros(T)
        log_ret[1:] = np.log(close[1:] / close[:-1])
        log_ret[0] = log_ret[1] # handle first day
        
        # 2. realized_vol (20-day rolling std of log_returns)
        log_ret_s = pd.Series(log_ret)
        real_vol = log_ret_s.rolling(window=20, min_periods=1).std().bfill().values
        
        # 3. normalized_volume
        vol_s = pd.Series(volume)
        vol_mean_20 = vol_s.rolling(window=20, min_periods=1).mean().bfill().values.copy()
        # avoid division by zero
        vol_mean_20[vol_mean_20 == 0] = 1 
        norm_vol = volume / vol_mean_20
        
        # 4. high_low_range
        hl_range = (high - low) / close
        
        # 5. close_to_sma20
        sma20 = pd.Series(close).rolling(window=20, min_periods=1).mean().bfill().values.copy()
        sma20[sma20 == 0] = 1 # avoid div by zero
        c_sma20 = (close / sma20) - 1.0
        
        features[i, :, 0] = log_ret
        features[i, :, 1] = real_vol
        features[i, :, 2] = norm_vol
        features[i, :, 3] = hl_range
        features[i, :, 4] = c_sma20
        
    return features

def step3_wavelet_denoising(features, valid_tickers):
    print("Step 1.3: Wavelet Denoising...")
    N, T, F = features.shape
    denoised_features = np.zeros((N, T, F))
    
    for i in tqdm(range(N), desc="Denoising stocks"):
        for f in range(F):
            signal = features[i, :, f]
            
            # Apply DWT
            coeffs = pywt.wavedec(signal, wavelet=WAVELET, level=WAVELET_LEVEL)
            
            # Zero out detail coefficients (all except the first one coeffs[0])
            for j in range(1, len(coeffs)):
                coeffs[j] = np.zeros_like(coeffs[j])
                
            # Reconstruct
            denoised_signal = pywt.waverec(coeffs, wavelet=WAVELET)
            
            # Trim in case waverec adds 1 sample at boundaries due to padding
            denoised_signal = denoised_signal[:T]
            denoised_features[i, :, f] = denoised_signal
            
    # Save tensor
    tensor_path = os.path.join(PROCESSED_DATA_DIR, "features.pt")
    X_tensor = torch.tensor(denoised_features, dtype=torch.float32)
    torch.save(X_tensor, tensor_path)
    print(f"Saved denoised features to {tensor_path}: shape {X_tensor.shape}")
    
    return X_tensor, denoised_features

def step4_dynamic_graphs(features, dates, valid_tickers):
    print("Step 1.4: Dynamic Correlation Graph Construction...")
    # We use features[:, :, 0] which is log_returns
    # Since features is already denoised, maybe we should use raw log_returns for correlations.
    # The proposal says: "correlations between asset returns". Let's use denoised log_returns.
    returns = features[:, :, 0] # N x T
    
    N, T = returns.shape
    # T must be > 60
    
    # We start from day CORR_WINDOW
    for t in tqdm(range(CORR_WINDOW, T), desc="Building daily graphs"):
        # Window: t-60 to t
        ret_window = returns[:, t-CORR_WINDOW:t] # N x 60
        
        # Compute correlation
        # Pearson correlation matrix
        C = np.corrcoef(ret_window) # N x N
        
        # Thresholding
        rows, cols = np.where(np.abs(C) > CORR_THRESHOLD)
        
        # Keep edges and weights
        edge_index = []
        edge_weight = []
        for r, c in zip(rows, cols):
            if r != c: # no self loops
                edge_index.append([r, c])
                edge_weight.append(np.abs(C[r, c]))
                
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t() # 2 x E
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32) # E
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
            
        date_str = dates[t].strftime('%Y-%m-%d')
        graph_path = os.path.join(GRAPHS_DIR, f"adj_{date_str}.pt")
        
        torch.save({'edge_index': edge_index, 'edge_weight': edge_weight}, graph_path)
    
    print(f"Generated sparse graphs from day {CORR_WINDOW} onwards.")

def step5_label_generation(aligned_data, dates, valid_tickers):
    print("Step 1.5: Label Generation...")
    N = len(valid_tickers)
    T = len(dates)
    
    labels = np.zeros((N, T), dtype=np.int8)
    
    for i, ticker in enumerate(valid_tickers):
        close = aligned_data[ticker]['Close'].values
        
        # future return 5 days
        for t in range(T - 5):
            fut_ret = (close[t+5] - close[t]) / close[t]
            if fut_ret < -0.05:
                labels[i, t] = 1
                
        # for last 5 days we cannot look into future, labels remain 0 or NaN logic (we'll just leave 0)
        
    y_tensor = torch.tensor(labels, dtype=torch.int8)
    labels_path = os.path.join(LABELS_DIR, "labels.pt")
    torch.save(y_tensor, labels_path)
    
    pos_rate = labels.sum() / (N * (T - 5))
    print(f"Saved labels to {labels_path}: shape {y_tensor.shape}, Global Pos Rate = {pos_rate*100:.2f}%")

if __name__ == '__main__':
    all_data = step1_download_data()
    aligned_data, dates, valid_tickers = align_and_clean_data(all_data)
    
    # Save valid tickers and dates metadata
    torch.save({'valid_tickers': valid_tickers, 'dates': dates}, os.path.join(PROCESSED_DATA_DIR, "meta.pt"))
    
    features = step2_feature_engineering(aligned_data, valid_tickers, dates)
    X_tensor, denoised_features = step3_wavelet_denoising(features, valid_tickers)
    
    step4_dynamic_graphs(denoised_features, dates, valid_tickers)
    step5_label_generation(aligned_data, dates, valid_tickers)
    print("Phase 1 Data Pipeline completed successfully!")
