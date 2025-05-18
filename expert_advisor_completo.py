# -*- coding: utf-8 -*-
# AdaptiveSAERenkoRL_EA - Expert Advisor com Renko Otimizado, SAE e RL (SAC)
# v5.8.1: Melhorado diagnóstico em obter_ticks_mt5, alerta sobre INITIAL_TRAIN_ON_STARTUP=False com modelos ausentes.

# --- 1. Importações ---
import os
import json 
import time
import math
import random
import logging
import logging.config
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict 
import threading
from typing import Optional, Union, List, Tuple, Dict, Any, cast 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Categorical # Normal não usada diretamente por SAC discreto, mas pode ser útil para extensões.
from sklearn.preprocessing import StandardScaler 
import joblib 
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import MetaTrader5 as mt5
import requests 
import zipfile  
import io       

# --- 2. Configuração do Logging ---
LOG_CONFIG = {
    "version": 1, "disable_existing_loggers": False,
    "formatters": {"detailed": {"format": "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s"},
                   "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "simple", "stream": "ext://sys.stdout"},
                 "file": {"class": "logging.handlers.RotatingFileHandler", "level": "DEBUG", "formatter": "detailed",
                          "filename": "AdaptiveSAERenkoRL_EA_v5_8_1.log", "maxBytes": 30*1024*1024, "backupCount": 10, "encoding": "utf-8"}}, # Nome do log atualizado
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    "loggers": { "AdaptiveEA": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False} }
}
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("AdaptiveEA.Main")

# --- Função para Deep Merge de Dicionários ---
def deep_update(source: Dict[Any, Any], overrides: Dict[Any, Any]) -> Dict[Any, Any]:
    updated = source.copy() 
    for key, value in overrides.items():
        if isinstance(value, dict) and key in updated and isinstance(updated[key], dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated

# --- 3. Configurações Globais ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "MT5_SETTINGS": {
        "LOGIN": 0, "PASSWORD": "", "SERVER": "", "PATH": "", 
        "SYMBOL": "EURUSD", "TIMEFRAME_FETCH_MT5": mt5.TIMEFRAME_M1, "TIMEFRAME_STR": "M1", "TIMEFRAME_SECONDS": 60,
        "MAGIC_NUMBER": 789127, "MAX_SPREAD_POINTS": 20, "DEVIATION_SLIPPAGE": 10,
        "DATA_SOURCE": "MT5_TICKS", 
        "EXNESS_TICKS_CONFIG": {
            "SYMBOL_MAPPING": {"EURUSD": "EURUSD", "BTCUSD": "BTCUSD_Raw_Spread", "XAUUSD": "XAUUSD"},
            "TICKS_FOLDER": "exness_tick_data_v5_rl", "CSV_FOLDER": "exness_csv_data_v5_rl",
            "LOOKBACK_MONTHS_RETRAIN": 6, "MAX_DOWNLOAD_ATTEMPTS": 3, "DOWNLOAD_TIMEOUT_SECONDS": 450,
            "HISTORICAL_DATA_YEARS_FOR_INITIAL_TRAIN": 2 
        },
        "MT5_TICKS_CONFIG": {"LOOKBACK_TICKS_LIVE_RENKO": 200000, "LOOKBACK_TICKS_OPTIMIZE_BRICK": 250000, "LOOKBACK_TICKS_INITIAL_TRAIN": 5000000}, 
        "MT5_BARS_CONFIG": {"LOOKBACK_BARS_M1_OPERATIONAL": 5000, "LOOKBACK_BARS_M1_FOR_FEATURES_INIT": 7000, "LOOKBACK_BARS_M1_INITIAL_TRAIN": 262800 * 2 } 
    },
    "GENERAL_SETTINGS": {
        "MODEL_DIR": "AdaptiveSAERenkoRL_Models_v5", "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "RANDOM_SEED": 42, "INITIAL_TRAIN_ON_STARTUP": False, 
        "RETRAIN_INTERVAL_HOURS": 24 * 7 
    },
    "RENKO_SETTINGS": {
        "OPTIMIZER_ENABLED": True,
        "OPTIMIZATION_LOOKBACK_TICKS_RETRAIN": 400000, "OPTIMIZATION_LOOKBACK_TICKS_LIVE": 200000,
        "OPTIMIZATION_MIN_BRICK_ABS_POINTS": 10, "OPTIMIZATION_MAX_BRICK_ABS_POINTS": 3500,
        "OPTIMIZATION_BRICK_TEST_STEPS": 35, "OPTIMIZATION_REVERSAL_PENALTY_FACTOR": 1.4,
        "BRICK_CALC_INTERVAL_MINUTES": 15, "RENKO_HISTORY_FOR_FEATURES": 2000,
        "FALLBACK_RENKO_BRICK_ATR_FACTOR": 0.30, 
        "FALLBACK_RENKO_ATR_PERIOD": 14      
    },
    "FEATURE_ENGINEERING_SETTINGS": {
        "SCALER_RENKO_FEATURES_PATH": "renko_features_scaler_v5.pkl",
        "FEATURE_COLUMNS": [
            'renko_ema_5', 'renko_ema_10', 'renko_ema_20', 'renko_ema_50', 'renko_ema_100', 'renko_ema_200',
            'renko_rsi_7', 'renko_rsi_14', 'renko_rsi_21', 'renko_rsi_28',
            'renko_stoch_k_14_3_3', 'renko_stoch_d_14_3_3', 'renko_stoch_k_21_5_5', 'renko_stoch_d_21_5_5',
            'renko_will_r_14', 'renko_will_r_28',
            'renko_macd', 'renko_macd_signal', 'renko_macd_diff',
            'renko_adx_14', 'renko_adx_pos_14', 'renko_adx_neg_14', 'renko_cci_14', 'renko_cci_20',
            'renko_bb_hband_20_2', 'renko_bb_lband_20_2', 'renko_bb_mavg_20_2', 'renko_bb_wband_20_2', 'renko_bb_pband_20_2',
            'renko_atr_7', 'renko_atr_14', 'renko_atr_21', 'renko_atr_28',
            'renko_momentum_1', 'renko_momentum_3', 'renko_momentum_5', 'renko_momentum_10', 'renko_momentum_20',
            'renko_brick_duration_log', 'renko_volume_sma_20', 'renko_volume_zscore_20', 'renko_consecutive_bricks',
            'renko_price_vs_ema50_norm_atr14', 'renko_price_vs_ema200_norm_atr14',
            'renko_ema_ratio_10_50', 'renko_ema_ratio_20_100', 'renko_ema_ratio_50_200',
            'renko_volatility_atr_ratio_7_21', 'renko_volatility_atr_ratio_14_28',
            'renko_hour_sin', 'renko_hour_cos', 'renko_dayofweek_sin', 'renko_dayofweek_cos'
        ] 
    },
    "SAE_SETTINGS": {
        "ENABLED": True, "MODEL_PATH": "sae_model_v5.pth", "SCALER_PRE_SAE_PATH": "scaler_pre_sae_v5.joblib",
        "LAYER_DIMS": [None, 200, 100, 50], 
        "ACTIVATION_FN": "gelu", "FINAL_ACTIVATION_ENCODER": "tanh", "DROPOUT_P": 0.2,
        "EPOCHS": 120, "BATCH_SIZE": 256, "LEARNING_RATE": 7e-4, "WEIGHT_DECAY": 5e-6,
        "USE_BATCH_NORM": True,
        "TRAIN_DATA_MIN_RENKO_BARS": 5000 
    },
    "RL_AGENT_SETTINGS": {
        "ENABLED": True, "ACTOR_MODEL_PATH": "sac_actor_v5.pth", "CRITIC_MODEL_PATH": "sac_critic_v5.pth",
        "STATE_HISTORY_LEN": 30, 
        "N_ACTIONS": 8, 
        "MAX_TOTAL_POSITION_LOTS": 0.1, 
        "PARTIAL_LOT_SIZES": [0.01, 0.02, 0.03], 
        "ACTION_MAPPING": {0: "HOLD", 1: "BUY_S", 2: "BUY_M", 3: "BUY_L", 4: "SELL_S", 5: "SELL_M", 6: "SELL_L", 7:"CLOSE_ALL"},
        "GAMMA": 0.995, "TAU": 0.005, "ALPHA": 0.15, "LR_ACTOR": 1.5e-4, "LR_CRITIC": 1.5e-4, "LR_ALPHA": 1.5e-4,
        "REPLAY_BUFFER_SIZE": int(1.5e5), "SAC_BATCH_SIZE": 512, "HIDDEN_DIM": 384,
        "TARGET_UPDATE_INTERVAL": 1, "AUTOMATIC_ENTROPY_TUNING": True, "ACTOR_UPDATE_FREQ": 2,
        "INITIAL_SL_PIP_DISTANCE_RL": 200, "INITIAL_TP_PIP_DISTANCE_RL": 400,
        "REWARD_CONFIG": {
            "INITIAL_SIM_BALANCE": 10000.0, 
            "REALIZED_PNL_FACTOR": 1.0, "UNREALIZED_PNL_CHANGE_FACTOR": 0.3,
            "SHARPE_RATIO_WINDOW": 50, "SHARPE_FACTOR": 0.1,
            "MAX_DRAWDOWN_WINDOW": 100, "DRAWDOWN_PENALTY_FACTOR": -0.8,
            "HOLDING_TIME_OPEN_POS_PENALTY": -0.0001, 
            "TRADE_EXECUTION_COST_FIXED": 0.5, 
            "SPREAD_COST_FACTOR": 0.5, 
            "ACTION_CHANGE_PENALTY": -0.005, 
            "TARGET_RISK_REWARD_RATIO": 1.5, "RISK_REWARD_BONUS_FACTOR": 0.05,
            "TRADE_COUNT_PENALTY_FACTOR": -0.1, 
            "COMMISSION_PER_LOT_CURRENCY": 3.5, 
            "SLIPPAGE_ESTIMATED_PIPS": 0.5 
        },
        "TRAINING_EPISODES": 1000, "MAX_STEPS_PER_EPISODE": 1500, "LEARNING_STARTS_AFTER_STEPS": 20000, 
        "ENV_STATE_NORMALIZATION_PARAMS_PATH": "rl_env_state_norm_params_v5.joblib",
        "TRAIN_DATA_MIN_RENKO_BARS_RL": 10000, 
        "MAX_PARTIAL_POSITIONS": 3 
    },
    "TRADING_RISK_SETTINGS": { 
        "GLOBAL_MAX_DRAWDOWN_STOP_PERCENT": 15.0 
    }
}
CONFIG: Dict[str, Any] = DEFAULT_CONFIG.copy()

# --- 4. Variáveis Globais de Estado ---
mt5_initialized_flag: bool = False
last_retrain_time_global: Optional[datetime] = None
retraining_in_progress_flag: bool = False
symbol_point_global: float = 0.00001
symbol_digits_global: int = 5
last_calculated_brick_size_global: Optional[float] = None
last_brick_calc_time_global: Optional[datetime] = None
feature_engineer_instance: Optional['FeatureEngineeringAdaptive'] = None
sae_handler_instance: Optional['SAEHandler'] = None
rl_agent_instance: Optional['RLAgentSAC'] = None
renko_builder_instance: Optional['DynamicRenkoBuilder'] = None
renko_brick_optimizer_instance: Optional['RenkoBrickOptimizer'] = None
comm_manager_instance: Optional['GerenciadorComunicacaoMT5Adaptativo'] = None
rl_live_balance: float = CONFIG["RL_AGENT_SETTINGS"].get("REWARD_CONFIG",{}).get("INITIAL_SIM_BALANCE", 10000.0) 
rl_live_equity: float = rl_live_balance
rl_live_open_positions: List[Dict[str, Any]] = [] 
rl_live_next_ticket: int = 1
rl_live_trade_history: List[Dict[str, Any]] = []

# --- 5. Funções Utilitárias de Dados ---
def get_exness_mapped_symbol(mt5_symbol_map_key: str) -> str:
    return CONFIG["MT5_SETTINGS"]["EXNESS_TICKS_CONFIG"]["SYMBOL_MAPPING"].get(mt5_symbol_map_key, mt5_symbol_map_key)

def baixar_e_extrair_ticks_exness(simbolo_mt5_key: str, ano: int, mes: int) -> Optional[str]:
    cfg_exness = CONFIG["MT5_SETTINGS"]["EXNESS_TICKS_CONFIG"]
    simbolo_exness_nome = get_exness_mapped_symbol(simbolo_mt5_key)
    mes_fmt = f"{mes:02d}"
    nome_zip = f"Exness_{simbolo_exness_nome}_{ano}_{mes_fmt}.zip"
    url = f"https://ticks.ex2archive.com/ticks/{simbolo_exness_nome}/{ano}/{mes_fmt}/{nome_zip}"
    
    os.makedirs(cfg_exness["TICKS_FOLDER"], exist_ok=True)
    os.makedirs(cfg_exness["CSV_FOLDER"], exist_ok=True)

    path_zip = os.path.join(cfg_exness["TICKS_FOLDER"], nome_zip)
    nome_csv = f"Exness_{simbolo_exness_nome}_{ano}_{mes_fmt}_ticks.csv"
    path_csv = os.path.join(cfg_exness["CSV_FOLDER"], nome_csv)

    if os.path.exists(path_csv): 
        logger.debug(f"CSV Exness já existe: {path_csv}")
        return path_csv
        
    logger.info(f"Baixando ticks Exness de: {url}")
    for attempt_dl in range(cfg_exness["MAX_DOWNLOAD_ATTEMPTS"]):
        try:
            response = requests.get(url, stream=True, timeout=cfg_exness["DOWNLOAD_TIMEOUT_SECONDS"])
            response.raise_for_status()
            with open(path_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=262144): 
                    f.write(chunk) 
            logger.info(f"Download completo: {path_zip}")
            
            with zipfile.ZipFile(path_zip, 'r') as z_ref:
                csv_files_in_zip = [f_name for f_name in z_ref.namelist() if f_name.lower().endswith('.csv')]
                if not csv_files_in_zip: 
                    logger.error(f"Nenhum arquivo CSV encontrado no ZIP: {path_zip}")
                    if os.path.exists(path_zip): os.remove(path_zip) 
                    return None
                with z_ref.open(csv_files_in_zip[0]) as src, open(path_csv, 'wb') as dst: 
                    dst.write(src.read())
            logger.info(f"CSV extraído para: {path_csv}")
            if os.path.exists(path_zip): os.remove(path_zip) 
            return path_csv
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404: 
                logger.warning(f"Arquivo Exness não encontrado (404): {url}")
                return None 
            logger.error(f"Erro HTTP (tentativa {attempt_dl+1}/{cfg_exness['MAX_DOWNLOAD_ATTEMPTS']}) ao baixar dados Exness: {http_err}")
        except Exception as e: 
            logger.error(f"Erro (tentativa {attempt_dl+1}/{cfg_exness['MAX_DOWNLOAD_ATTEMPTS']}) durante download/extração Exness: {e}")
        
        if attempt_dl < cfg_exness["MAX_DOWNLOAD_ATTEMPTS"] - 1: 
            time.sleep(10 * (attempt_dl + 1)) 
            
    if os.path.exists(path_zip): os.remove(path_zip) 
    return None

def ler_ticks_exness_csv(caminho_csv: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(caminho_csv): 
        logger.warning(f"Arquivo CSV Exness não encontrado para leitura: {caminho_csv}")
        return None
    logger.debug(f"Lendo CSV Exness: {caminho_csv}")
    try:
        with open(caminho_csv, 'r', encoding='utf-8') as f_test_sep:
            first_line_test = f_test_sep.readline()
            sep_to_use = ';' if first_line_test.count(';') > first_line_test.count(',') else ','

        chunks = []
        for chunk in pd.read_csv(caminho_csv, sep=sep_to_use, 
                                 usecols=['Timestamp', 'Bid', 'Ask', 'Volume'], 
                                 dtype={'Timestamp': str, 'Bid': np.float32, 'Ask': np.float32, 'Volume': np.float32}, 
                                 chunksize=10_000_000, 
                                 engine='c', 
                                 on_bad_lines='warn', 
                                 low_memory=False):
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce', utc=True)
            chunk = chunk.dropna(subset=['Timestamp', 'Bid', 'Ask']) 
            chunk['Volume'] = chunk['Volume'].fillna(1.0) 
            if not chunk.empty: 
                chunks.append(chunk)
        
        if not chunks: 
            logger.warning(f"Nenhum dado válido encontrado após processar chunks do CSV: {caminho_csv}")
            return None
            
        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values(by='Timestamp').drop_duplicates(subset=['Timestamp', 'Bid', 'Ask'], keep='last').reset_index(drop=True)
        
        if not df['Timestamp'].dt.tz:
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
        else:
            df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC')
            
        logger.info(f"CSV Exness {caminho_csv} lido com {len(df)} linhas.")
        return df
    except Exception as e: 
        logger.error(f"Erro crítico ao ler ou processar CSV Exness {caminho_csv}: {e}", exc_info=True)
        return None

# --- 6. Renko: Otimizador e Builder ---
class RenkoBrickOptimizer:
    def __init__(self, symbol_point: float, symbol_digits: int, min_abs: float, max_abs: float, steps: int, penalty: float):
        self.symbol_point = symbol_point; self.symbol_digits = symbol_digits
        self.min_brick_abs = max(min_abs, symbol_point * 2) 
        self.max_brick_abs = max(max_abs, self.min_brick_abs + symbol_point * 20)
        self.brick_test_steps = max(steps, 5)
        self.reversal_penalty_factor = penalty
        self.renko_builder_internal = DynamicRenkoBuilder() 

    def _calculate_stats(self, df_ticks: pd.DataFrame, brick_size: float) -> Tuple[int, int]:
        df_r = self.renko_builder_internal.calculate_renko_from_ticks(df_ticks.copy(), brick_size)
        if df_r.empty or len(df_r) < 3: return 0, 0 
        n_bars = len(df_r)
        reversals = (df_r['direction'].diff().fillna(0).abs() == 2).sum() 
        return n_bars, reversals

    def find_optimal_brick_size(self, df_ticks_opt: pd.DataFrame) -> Optional[float]:
        if df_ticks_opt is None or df_ticks_opt.empty or len(df_ticks_opt) < 500: 
            fallback_brick_val = round(((self.min_brick_abs + self.max_brick_abs) / 2) / self.symbol_point) * self.symbol_point
            logger.warning(f"RenkoOpt: Dados insuficientes ({len(df_ticks_opt) if df_ticks_opt is not None else 0}), usando fallback: {fallback_brick_val}")
            return max(fallback_brick_val, self.min_brick_abs)
        
        test_bricks = np.logspace(np.log10(max(self.min_brick_abs, self.symbol_point)), np.log10(self.max_brick_abs), self.brick_test_steps)
        test_bricks = np.round(test_bricks / self.symbol_point) * self.symbol_point
        test_bricks = np.unique(test_bricks[test_bricks >= self.symbol_point * 0.5]) 
        if len(test_bricks) == 0: 
            logger.warning("RenkoOpt: Nenhum brick de teste válido gerado. Usando min_brick_abs.")
            return self.min_brick_abs

        best_brick, best_score, best_n_b, best_n_r = None, -float('inf'), 0, 0
        logger.info(f"RenkoOpt: Testando {len(test_bricks)} tamanhos de brick de {test_bricks[0]:.{self.symbol_digits}f} a {test_bricks[-1]:.{self.symbol_digits}f}")
        for brick in test_bricks:
            if brick < self.symbol_point * 0.1 : continue 
            n_b, n_r = self._calculate_stats(df_ticks_opt, brick)
            score = (n_b - (self.reversal_penalty_factor * n_r)) * math.log1p(n_b + 1e-6) / (math.log1p(brick / self.symbol_point) + 1e-3) if n_b > 10 else -float('inf')
            
            if score > best_score: 
                best_score, best_brick, best_n_b, best_n_r = score, brick, n_b, n_r
            elif score == best_score and best_brick is not None and brick < best_brick: 
                best_brick, best_n_b, best_n_r = brick, n_b, n_r
        
        if best_brick is None: 
            logger.warning("RenkoOpt: Nenhum brick ótimo encontrado. Usando brick médio da faixa de teste.")
            best_brick = test_bricks[len(test_bricks)//2] if len(test_bricks)>0 else self.min_brick_abs
            
        final_brick = round(best_brick / self.symbol_point) * self.symbol_point
        final_brick = max(final_brick, self.symbol_point) 
        logger.info(f"RenkoOpt: Melhor Brick={final_brick:.{self.symbol_digits}f} (Score={best_score:.2f}, Barras={best_n_b}, Rev={best_n_r})")
        return final_brick

class DynamicRenkoBuilder:
    def __init__(self):
        self._point: float = symbol_point_global
        self._digits: int = symbol_digits_global
        self._update_symbol_details() 

    def _update_symbol_details(self):
        global symbol_point_global, symbol_digits_global
        if mt5_initialized_flag: 
            try:
                info = mt5.symbol_info(CONFIG["MT5_SETTINGS"]["SYMBOL"])
                if info: 
                    self._point, self._digits = info.point, info.digits
                    symbol_point_global, symbol_digits_global = self._point, self._digits 
                else:
                    logger.warning("RenkoBuilder: Falha ao obter symbol_info para atualizar point/digits.")
            except Exception as e:
                logger.error(f"RenkoBuilder: Erro ao atualizar symbol_info: {e}")

    def calculate_renko_from_ticks(self, df_ticks: pd.DataFrame, brick_size: float) -> pd.DataFrame:
        self._update_symbol_details() 
        
        if df_ticks is None or df_ticks.empty:
            logger.debug("RenkoBuilder: df_ticks vazio ou None.")
            return pd.DataFrame()
        if brick_size <= self._point / 100.0 or brick_size <= 0: 
             logger.warning(f"RenkoBuilder: brick_size ({brick_size}) inválido ou muito pequeno vs point ({self._point}). Usando fallback mínimo de 2 * point.")
             brick_size = max(self._point * 2.0, 1e-9) 

        df_t = df_ticks.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_t['Timestamp']):
            df_t['Timestamp'] = pd.to_datetime(df_t['Timestamp'], errors='coerce', utc=True)
        df_t = df_t.sort_values(by='Timestamp').dropna(subset=['Timestamp', 'Bid', 'Ask']).reset_index(drop=True)
        if df_t.empty: 
            logger.debug("RenkoBuilder: df_ticks vazio após limpeza e ordenação.")
            return pd.DataFrame()

        bricks = []
        current_ref = (df_t['Bid'].iloc[0] + df_t['Ask'].iloc[0]) / 2.0
        current_ref = round(current_ref / (self._point / 2)) * (self._point / 2)

        last_dir = 0 
        vol_acc = 0.0 
        last_brick_close_price = current_ref 

        for _, row in df_t.iterrows():
            t, bid, ask, vol = row['Timestamp'], row['Bid'], row['Ask'], row.get('Volume', 1.0)
            if pd.isna(vol) or vol < 1e-9 : vol = 1e-6 
            
            price_to_check = (bid + ask) / 2.0 
            
            formed_brick_in_loop = True
            while formed_brick_in_loop:
                formed_brick_in_loop = False
                vol_acc_for_this_brick = vol if not formed_brick_in_loop else 0 

                if price_to_check >= last_brick_close_price + brick_size:
                    if last_dir == -1 and price_to_check < last_brick_close_price + 2 * brick_size: 
                        break 

                    open_price = last_brick_close_price if last_dir != -1 else last_brick_close_price + brick_size 
                    if last_dir == -1: 
                        open_price = last_brick_close_price + brick_size
                    
                    close_price = open_price + brick_size
                    
                    if price_to_check < close_price and last_dir != -1 : 
                         if price_to_check < open_price + brick_size: break
                    if price_to_check < close_price and last_dir == -1 : 
                         if price_to_check < open_price + brick_size : break

                    bricks.append({'time':t, 'open':open_price, 'high':close_price, 'low':open_price, 'close':close_price, 
                                   'direction':1, 'brick_size':brick_size, 'tick_volume': vol_acc + vol_acc_for_this_brick})
                    last_brick_close_price = close_price
                    last_dir = 1
                    vol_acc = 0.0 
                    formed_brick_in_loop = True
                
                elif price_to_check <= last_brick_close_price - brick_size:
                    if last_dir == 1 and price_to_check > last_brick_close_price - 2 * brick_size: 
                        break

                    open_price = last_brick_close_price if last_dir != 1 else last_brick_close_price - brick_size 
                    if last_dir == 1: 
                        open_price = last_brick_close_price - brick_size

                    close_price = open_price - brick_size

                    if price_to_check > close_price and last_dir != 1: 
                        if price_to_check > open_price - brick_size: break
                    if price_to_check > close_price and last_dir == 1: 
                        if price_to_check > open_price - brick_size: break

                    bricks.append({'time':t, 'open':open_price, 'high':open_price, 'low':close_price, 'close':close_price, 
                                   'direction':-1, 'brick_size':brick_size, 'tick_volume':vol_acc + vol_acc_for_this_brick})
                    last_brick_close_price = close_price
                    last_dir = -1
                    vol_acc = 0.0
                    formed_brick_in_loop = True
            
            if not formed_brick_in_loop : 
                vol_acc += vol

        if not bricks: 
            logger.debug("RenkoBuilder: Nenhum brick Renko gerado.")
            return pd.DataFrame()
            
        df_r = pd.DataFrame(bricks)
        df_r['time'] = pd.to_datetime(df_r['time'], utc=True)
        df_r = df_r.drop_duplicates(subset=['time','close', 'direction'],keep='last').reset_index(drop=True) 
        
        df_r['original_close_price'] = df_r['close']
        df_r['original_open_price'] = df_r['open']
        df_r['original_high_price'] = df_r['high']
        df_r['original_low_price'] = df_r['low']

        return df_r[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'direction', 'brick_size', 
                     'original_open_price', 'original_high_price', 'original_low_price', 'original_close_price']]

# --- 7. Feature Engineering ---
class FeatureEngineeringAdaptive:
    def __init__(self, scaler_path: str, feature_cols_list: List[str]):
        self.scaler_path_base_name = scaler_path # Nome base do arquivo
        self.scaler_path_full = "" # Será definido em _update_scaler_path
        self.base_feature_columns = feature_cols_list
        self.scaler: StandardScaler = StandardScaler() 
        self.final_feature_columns: List[str] = [] 
        self.is_scaler_fitted = False
        self._update_scaler_path() # Atualiza o caminho completo na inicialização
        self.load_scaler()

    def _update_scaler_path(self):
        """Atualiza o caminho completo do scaler com base no símbolo da configuração."""
        symbol = CONFIG["MT5_SETTINGS"]["SYMBOL"]
        model_dir_symbol = os.path.join(CONFIG["GENERAL_SETTINGS"]["MODEL_DIR"], symbol)
        os.makedirs(model_dir_symbol, exist_ok=True)
        self.scaler_path_full = os.path.join(model_dir_symbol, self.scaler_path_base_name)


    def add_technical_indicators(self, df_renko: pd.DataFrame) -> pd.DataFrame:
        if df_renko is None or df_renko.empty or len(df_renko) < 60: 
            logger.debug(f"FeatureEng: Dados Renko insuficientes ({len(df_renko) if df_renko is not None else 0}) para indicadores.")
            return pd.DataFrame() 
        df = df_renko.copy()
        
        for col_ohlc in ['open', 'high', 'low', 'close']:
            if col_ohlc not in df.columns:
                logger.error(f"FeatureEng: Coluna {col_ohlc} ausente nos dados Renko.")
                return pd.DataFrame()
            df[col_ohlc] = pd.to_numeric(df[col_ohlc], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        if len(df) < 60: 
             logger.debug(f"FeatureEng: Dados Renko insuficientes ({len(df)}) após limpeza de NaNs em OHLC.")
             return pd.DataFrame()

        c = df['close']; h = df['high']; l = df['low']
        v = df.get('tick_volume', pd.Series(1.0, index=df.index)).fillna(1e-6) 
        if v.sum() < 1e-5 : v = pd.Series(1.0, index=df.index) 

        calculated_features_temp: Dict[str, pd.Series] = {} 
        
        if 'renko_ema_5' in self.base_feature_columns: calculated_features_temp['renko_ema_5'] = EMAIndicator(c, 5, False).ema_indicator()
        if 'renko_ema_10' in self.base_feature_columns: calculated_features_temp['renko_ema_10'] = EMAIndicator(c, 10, False).ema_indicator()
        if 'renko_ema_20' in self.base_feature_columns: calculated_features_temp['renko_ema_20'] = EMAIndicator(c, 20, False).ema_indicator()
        if 'renko_ema_50' in self.base_feature_columns: calculated_features_temp['renko_ema_50'] = EMAIndicator(c, 50, False).ema_indicator()
        if 'renko_ema_100' in self.base_feature_columns: calculated_features_temp['renko_ema_100'] = EMAIndicator(c, 100, False).ema_indicator()
        if 'renko_ema_200' in self.base_feature_columns: calculated_features_temp['renko_ema_200'] = EMAIndicator(c, 200, False).ema_indicator()
        
        if 'renko_atr_7' in self.base_feature_columns: calculated_features_temp['renko_atr_7'] = AverageTrueRange(h,l,c,7,False).average_true_range()
        if 'renko_atr_14' in self.base_feature_columns: calculated_features_temp['renko_atr_14'] = AverageTrueRange(h,l,c,14,False).average_true_range()
        if 'renko_atr_21' in self.base_feature_columns: calculated_features_temp['renko_atr_21'] = AverageTrueRange(h,l,c,21,False).average_true_range()
        if 'renko_atr_28' in self.base_feature_columns: calculated_features_temp['renko_atr_28'] = AverageTrueRange(h,l,c,28,False).average_true_range()

        for col_name in self.base_feature_columns:
            if col_name in calculated_features_temp: continue 
            try:
                if col_name == 'renko_rsi_7': calculated_features_temp[col_name] = RSIIndicator(c, 7, False).rsi()
                elif col_name == 'renko_rsi_14': calculated_features_temp[col_name] = RSIIndicator(c, 14, False).rsi()
                elif col_name == 'renko_rsi_21': calculated_features_temp[col_name] = RSIIndicator(c, 21, False).rsi()
                elif col_name == 'renko_rsi_28': calculated_features_temp[col_name] = RSIIndicator(c, 28, False).rsi()
                elif col_name == 'renko_stoch_k_14_3_3': stoch14 = StochasticOscillator(h,l,c,14,3,False); calculated_features_temp[col_name] = stoch14.stoch()
                elif col_name == 'renko_stoch_d_14_3_3': stoch14d = StochasticOscillator(h,l,c,14,3,False); calculated_features_temp[col_name] = stoch14d.stoch_signal()
                elif col_name == 'renko_stoch_k_21_5_5': stoch21 = StochasticOscillator(h,l,c,21,5,False); calculated_features_temp[col_name] = stoch21.stoch()
                elif col_name == 'renko_stoch_d_21_5_5': stoch21d = StochasticOscillator(h,l,c,21,5,False); calculated_features_temp[col_name] = stoch21d.stoch_signal()
                elif col_name == 'renko_will_r_14': calculated_features_temp[col_name] = WilliamsRIndicator(h,l,c,14,False).williams_r()
                elif col_name == 'renko_will_r_28': calculated_features_temp[col_name] = WilliamsRIndicator(h,l,c,28,False).williams_r()
                elif col_name == 'renko_macd': calculated_features_temp[col_name] = MACD(c,26,12,9,False).macd()
                elif col_name == 'renko_macd_signal': calculated_features_temp[col_name] = MACD(c,26,12,9,False).macd_signal()
                elif col_name == 'renko_macd_diff': calculated_features_temp[col_name] = MACD(c,26,12,9,False).macd_diff()
                elif col_name == 'renko_adx_14': adx14 = ADXIndicator(h,l,c,14,False); calculated_features_temp[col_name] = adx14.adx()
                elif col_name == 'renko_adx_pos_14': adx14p = ADXIndicator(h,l,c,14,False); calculated_features_temp[col_name] = adx14p.adx_pos()
                elif col_name == 'renko_adx_neg_14': adx14n = ADXIndicator(h,l,c,14,False); calculated_features_temp[col_name] = adx14n.adx_neg()
                elif col_name == 'renko_cci_14': calculated_features_temp[col_name] = CCIIndicator(h,l,c,14,0.015,False).cci()
                elif col_name == 'renko_cci_20': calculated_features_temp[col_name] = CCIIndicator(h,l,c,20,0.015,False).cci()
                elif col_name == 'renko_bb_hband_20_2': calculated_features_temp[col_name] = BollingerBands(c,20,2,False).bollinger_hband()
                elif col_name == 'renko_bb_lband_20_2': calculated_features_temp[col_name] = BollingerBands(c,20,2,False).bollinger_lband()
                elif col_name == 'renko_bb_mavg_20_2': calculated_features_temp[col_name] = BollingerBands(c,20,2,False).bollinger_mavg()
                elif col_name == 'renko_bb_wband_20_2': calculated_features_temp[col_name] = BollingerBands(c,20,2,False).bollinger_wband()
                elif col_name == 'renko_bb_pband_20_2': calculated_features_temp[col_name] = BollingerBands(c,20,2,False).bollinger_pband()
                elif col_name == 'renko_momentum_1': calculated_features_temp[col_name] = c.diff(1)
                elif col_name == 'renko_momentum_3': calculated_features_temp[col_name] = c.diff(3)
                elif col_name == 'renko_momentum_5': calculated_features_temp[col_name] = c.diff(5)
                elif col_name == 'renko_momentum_10': calculated_features_temp[col_name] = c.diff(10)
                elif col_name == 'renko_momentum_20': calculated_features_temp[col_name] = c.diff(20)
                elif col_name == 'renko_brick_duration_log': 
                    if 'time' in df and pd.api.types.is_datetime64_any_dtype(df['time']):
                        calculated_features_temp[col_name] = np.log1p(df['time'].diff().dt.total_seconds().fillna(1).clip(lower=1)) 
                    else: logger.warning("FeatureEng: Coluna 'time' ausente ou tipo inválido para renko_brick_duration_log")
                elif col_name == 'renko_volume_sma_20': calculated_features_temp[col_name] = v.rolling(20,min_periods=1).mean()
                elif col_name == 'renko_volume_zscore_20': 
                    vol_mean = v.rolling(20,min_periods=1).mean()
                    vol_std = v.rolling(20,min_periods=1).std().replace(0,1e-7) + 1e-7
                    calculated_features_temp[col_name] = (v - vol_mean) / vol_std
                elif col_name == 'renko_consecutive_bricks' and 'direction' in df:
                    s_dir = df['direction'].fillna(0); calculated_features_temp[col_name] = (s_dir.groupby((s_dir != s_dir.shift()).cumsum()).cumcount()+1) * s_dir
                elif col_name == 'renko_price_vs_ema50_norm_atr14':
                    if 'renko_ema_50' in calculated_features_temp and 'renko_atr_14' in calculated_features_temp and calculated_features_temp['renko_atr_14'] is not None:
                        atr14_safe = calculated_features_temp['renko_atr_14'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = (c - calculated_features_temp['renko_ema_50']) / atr14_safe
                elif col_name == 'renko_price_vs_ema200_norm_atr14':
                    if 'renko_ema_200' in calculated_features_temp and 'renko_atr_14' in calculated_features_temp and calculated_features_temp['renko_atr_14'] is not None:
                        atr14_safe = calculated_features_temp['renko_atr_14'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = (c - calculated_features_temp['renko_ema_200']) / atr14_safe
                elif col_name == 'renko_ema_ratio_10_50':
                    if 'renko_ema_10' in calculated_features_temp and 'renko_ema_50' in calculated_features_temp and calculated_features_temp['renko_ema_50'] is not None:
                        ema50_safe = calculated_features_temp['renko_ema_50'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = calculated_features_temp['renko_ema_10'] / ema50_safe
                elif col_name == 'renko_ema_ratio_20_100':
                    if 'renko_ema_20' in calculated_features_temp and 'renko_ema_100' in calculated_features_temp and calculated_features_temp['renko_ema_100'] is not None:
                        ema100_safe = calculated_features_temp['renko_ema_100'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = calculated_features_temp['renko_ema_20'] / ema100_safe
                elif col_name == 'renko_ema_ratio_50_200':
                    if 'renko_ema_50' in calculated_features_temp and 'renko_ema_200' in calculated_features_temp and calculated_features_temp['renko_ema_200'] is not None:
                        ema200_safe = calculated_features_temp['renko_ema_200'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = calculated_features_temp['renko_ema_50'] / ema200_safe
                elif col_name == 'renko_volatility_atr_ratio_7_21':
                    if 'renko_atr_7' in calculated_features_temp and 'renko_atr_21' in calculated_features_temp and calculated_features_temp['renko_atr_21'] is not None:
                        atr21_safe = calculated_features_temp['renko_atr_21'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = calculated_features_temp['renko_atr_7'] / atr21_safe
                elif col_name == 'renko_volatility_atr_ratio_14_28':
                    if 'renko_atr_14' in calculated_features_temp and 'renko_atr_28' in calculated_features_temp and calculated_features_temp['renko_atr_28'] is not None:
                        atr28_safe = calculated_features_temp['renko_atr_28'].replace(0,1e-7) + 1e-7
                        calculated_features_temp[col_name] = calculated_features_temp['renko_atr_14'] / atr28_safe
                elif col_name == 'renko_hour_sin' and 'time' in df and pd.api.types.is_datetime64_any_dtype(df['time']): 
                    calculated_features_temp[col_name] = np.sin(2 * np.pi * df['time'].dt.hour/23.0) 
                elif col_name == 'renko_hour_cos' and 'time' in df and pd.api.types.is_datetime64_any_dtype(df['time']): 
                    calculated_features_temp[col_name] = np.cos(2 * np.pi * df['time'].dt.hour/23.0)
                elif col_name == 'renko_dayofweek_sin' and 'time' in df and pd.api.types.is_datetime64_any_dtype(df['time']): 
                    calculated_features_temp[col_name] = np.sin(2 * np.pi * df['time'].dt.dayofweek/6.0) 
                elif col_name == 'renko_dayofweek_cos' and 'time' in df and pd.api.types.is_datetime64_any_dtype(df['time']): 
                    calculated_features_temp[col_name] = np.cos(2 * np.pi * df['time'].dt.dayofweek/6.0)
            except Exception as e_feat_calc: 
                logger.debug(f"FeatureEng: Erro ao calcular {col_name}: {e_feat_calc}")
                calculated_features_temp[col_name] = pd.Series(np.nan, index=df.index) 

        for col_name_calc, data_series_calc in calculated_features_temp.items():
            df[col_name_calc] = data_series_calc
        
        self.final_feature_columns = [col for col in self.base_feature_columns if col in df.columns and df[col].notna().any()]
        if not self.final_feature_columns:
            logger.error("FeatureEng: Nenhuma feature final pôde ser calculada ou todas são NaN.")
            return pd.DataFrame()

        df_final_features = df[self.final_feature_columns].copy()
        df_final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_final_features.fillna(method='ffill', inplace=True)
        df_final_features.fillna(method='bfill', inplace=True)
        df_final_features.fillna(0, inplace=True)
        
        cols_to_keep_original = [col for col in df.columns if col not in self.final_feature_columns]
        df_out = pd.concat([df[cols_to_keep_original], df_final_features], axis=1)
        
        return df_out

    def prepare_features_for_model(self, df_with_indicators: pd.DataFrame, for_training: bool = True) -> Optional[np.ndarray]:
        self._update_scaler_path() # Garante que o caminho do scaler está atualizado com o símbolo
        if df_with_indicators.empty or not self.final_feature_columns:
            logger.debug("FeatureEng Prep: DF vazio ou colunas finais não definidas.")
            return None
        
        missing_cols = [col for col in self.final_feature_columns if col not in df_with_indicators.columns]
        if missing_cols:
            logger.error(f"FeatureEng Prep: Colunas finais ausentes no DataFrame de entrada: {missing_cols}")
            return None

        df_feat = df_with_indicators[self.final_feature_columns].copy() 
        if df_feat.isnull().values.any():
            logger.warning("FeatureEng Prep: NaNs encontrados nas features antes do escalonamento. Preenchendo com 0.")
            df_feat.fillna(0, inplace=True) 
        
        if df_feat.empty: 
            logger.debug("FeatureEng Prep: DF de features vazio após seleção de colunas.")
            return None
        
        if for_training:
            if len(df_feat) < max(20, len(self.final_feature_columns) + 1): 
                logger.warning(f"FeatureEng Prep: Amostras insuficientes ({len(df_feat)}) para treinar scaler. Necessário pelo menos {max(20, len(self.final_feature_columns) + 1)}.")
                return None 
            self.scaler = StandardScaler() 
            scaled_data = self.scaler.fit_transform(df_feat)
            self.is_scaler_fitted = True
            self.save_scaler()
            logger.info(f"FeatureEng Prep: Scaler treinado e salvo. {len(df_feat)} amostras. Dimensões: {scaled_data.shape}")
            return scaled_data
        else:
            if not self.is_scaler_fitted: 
                logger.warning("FeatureEng Prep: Scaler não treinado (is_scaler_fitted=False). Carregando novamente...")
                self.load_scaler() 
                if not self.is_scaler_fitted:
                    logger.error("FeatureEng Prep: Scaler ainda não treinado após tentativa de recarga. Não é possível escalar features.")
                    return None # Não pode prosseguir sem scaler ajustado
            
            if df_feat.shape[1] != self.scaler.n_features_in_:
                logger.error(f"FeatureEng Prep: Discrepância de features. Scaler espera {self.scaler.n_features_in_}, dados têm {df_feat.shape[1]}.")
                return None
            return self.scaler.transform(df_feat)

    def save_scaler(self):
        self._update_scaler_path() # Garante que o caminho está correto
        if self.is_scaler_fitted and hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            try: 
                joblib.dump(self.scaler, self.scaler_path_full) # Usa o caminho completo
                logger.info(f"FeatureEng: Scaler salvo em {self.scaler_path_full}")
            except Exception as e: 
                logger.error(f"FeatureEng: Erro ao salvar scaler: {e}")
        else:
            logger.warning(f"FeatureEng: Scaler não ajustado (fitted). Não foi salvo em {self.scaler_path_full}.")

    def load_scaler(self):
        self._update_scaler_path() # Garante que o caminho está correto
        if os.path.exists(self.scaler_path_full):
            try: 
                self.scaler = joblib.load(self.scaler_path_full)
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None and hasattr(self.scaler, 'n_features_in_'):
                    self.is_scaler_fitted = True
                    # Atualizar final_feature_columns com base no scaler carregado, se possível e consistente
                    if hasattr(self.scaler, 'feature_names_in_') and list(self.scaler.feature_names_in_) == self.base_feature_columns[:len(self.scaler.feature_names_in_)]:
                        self.final_feature_columns = list(self.scaler.feature_names_in_)
                    elif len(self.base_feature_columns) == self.scaler.n_features_in_:
                         self.final_feature_columns = self.base_feature_columns # Assumir que são as mesmas se o número bater
                    else:
                        logger.warning(f"FeatureEng: Scaler carregado tem {self.scaler.n_features_in_} features, mas base_feature_columns tem {len(self.base_feature_columns)}. Pode haver inconsistência.")
                        # Manter self.final_feature_columns como estava ou redefinir para base_feature_columns pode ser uma opção
                        self.final_feature_columns = self.base_feature_columns[:self.scaler.n_features_in_] # Tentativa de alinhar

                    logger.info(f"FeatureEng: Scaler carregado de {self.scaler_path_full}. Features: {self.scaler.n_features_in_}. Colunas finais inferidas: {self.final_feature_columns[:5]}...")
                else:
                    self.is_scaler_fitted = False
                    self.scaler = StandardScaler() 
                    logger.warning(f"FeatureEng: Scaler carregado de {self.scaler_path_full} parece não estar ajustado. Reinstanciado.")
            except Exception as e: 
                self.scaler = StandardScaler()
                self.is_scaler_fitted = False
                logger.error(f"FeatureEng: Erro ao carregar scaler de {self.scaler_path_full}: {e}. Novo scaler instanciado.")
        else: 
            self.scaler = StandardScaler()
            self.is_scaler_fitted = False
            logger.info(f"FeatureEng: Arquivo do scaler {self.scaler_path_full} não encontrado. Novo scaler instanciado.")

# --- 8. Stacked Autoencoder (SAE) ---
class StackedAutoencoder(nn.Module):
    def __init__(self, layer_dims: List[int], activation_fn: str = 'relu', final_enc_act: Optional[str] = 'tanh', dropout: float = 0.1, use_batch_norm: bool = False): 
        super().__init__()
        if not layer_dims or len(layer_dims) < 2:
            raise ValueError("SAE layer_dims deve conter pelo menos input e uma dimensão codificada.")
        self.layer_dims = layer_dims; self.use_batch_norm = use_batch_norm
        
        enc_layers = []
        for i in range(len(layer_dims) - 1):
            enc_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2: 
                if use_batch_norm: enc_layers.append(nn.BatchNorm1d(layer_dims[i+1]))
                if activation_fn == 'relu': enc_layers.append(nn.ReLU())
                elif activation_fn == 'leaky_relu': enc_layers.append(nn.LeakyReLU(0.1))
                elif activation_fn == 'selu': enc_layers.append(nn.SELU())
                elif activation_fn == 'gelu': enc_layers.append(nn.GELU())
                else: enc_layers.append(nn.Tanh()) 
                if dropout > 0: enc_layers.append(nn.Dropout(dropout))
            elif final_enc_act: 
                if final_enc_act == 'tanh': enc_layers.append(nn.Tanh())
                elif final_enc_act == 'sigmoid': enc_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*enc_layers)
        
        dec_layers = []
        rev_dims = layer_dims[::-1] 
        for i in range(len(rev_dims) - 1):
            dec_layers.append(nn.Linear(rev_dims[i], rev_dims[i+1]))
            if i < len(rev_dims) - 2: 
                if use_batch_norm: dec_layers.append(nn.BatchNorm1d(rev_dims[i+1]))
                if activation_fn == 'relu': dec_layers.append(nn.ReLU())
                elif activation_fn == 'leaky_relu': dec_layers.append(nn.LeakyReLU(0.1))
                elif activation_fn == 'selu': dec_layers.append(nn.SELU())
                elif activation_fn == 'gelu': dec_layers.append(nn.GELU())
                else: dec_layers.append(nn.Tanh())
                if dropout > 0 : dec_layers.append(nn.Dropout(dropout)) 
        self.decoder = nn.Sequential(*dec_layers)
        
        init_gain = nn.init.calculate_gain(activation_fn if activation_fn not in ['tanh', 'sigmoid', 'selu', 'gelu'] else 'relu')
        self.apply(lambda m: init_weights_rl(m, gain=init_gain))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class SAEHandler:
    def __init__(self, model_path: str, scaler_pre_sae_path: str, layer_dims_cfg: List[Optional[int]], device: str): 
        self.model_base_path = model_path 
        self.scaler_base_path = scaler_pre_sae_path
        self.model_path_full: str = "" 
        self.scaler_path_full: str = "" 
        
        self.layer_dimensions_config_template = layer_dims_cfg 
        self.device = torch.device(device)
        self.model: Optional[StackedAutoencoder] = None 
        self.scaler_pre_sae: StandardScaler = StandardScaler() 
        self.is_scaler_pre_sae_fitted = False
        self.input_dim_actual: Optional[int] = None 
        self.output_dim_actual: Optional[int] = None 
        
        self._update_paths_for_symbol() 
        self.load_scaler_pre_sae() 
        self.load_sae_model()    

    def _update_paths_for_symbol(self):
        symbol = CONFIG["MT5_SETTINGS"]["SYMBOL"]
        model_dir_symbol = os.path.join(CONFIG["GENERAL_SETTINGS"]["MODEL_DIR"], symbol)
        os.makedirs(model_dir_symbol, exist_ok=True)
        self.model_path_full = os.path.join(model_dir_symbol, os.path.basename(self.model_base_path))
        self.scaler_path_full = os.path.join(model_dir_symbol, os.path.basename(self.scaler_base_path))

    def load_scaler_pre_sae(self):
        self._update_paths_for_symbol() 
        if os.path.exists(self.scaler_path_full):
            try: 
                self.scaler_pre_sae = joblib.load(self.scaler_path_full)
                if hasattr(self.scaler_pre_sae, 'mean_') and self.scaler_pre_sae.mean_ is not None and hasattr(self.scaler_pre_sae, 'n_features_in_'):
                    self.is_scaler_pre_sae_fitted = True
                    self.input_dim_actual = self.scaler_pre_sae.n_features_in_ 
                    logger.info(f"SAE: Scaler pré-SAE carregado de {self.scaler_path_full}. Input dim inferido: {self.input_dim_actual}")
                else:
                    self.is_scaler_pre_sae_fitted = False
                    self.scaler_pre_sae = StandardScaler()
                    logger.warning(f"SAE: Scaler pré-SAE carregado de {self.scaler_path_full} parece não estar ajustado. Reinstanciado.")
            except Exception as e: 
                logger.error(f"SAE: Erro ao carregar scaler pré-SAE de {self.scaler_path_full}: {e}. Novo scaler instanciado.")
                self.scaler_pre_sae = StandardScaler()
                self.is_scaler_pre_sae_fitted = False
        else:
            self.scaler_pre_sae = StandardScaler() 
            self.is_scaler_pre_sae_fitted = False
            logger.info(f"SAE: Arquivo do scaler pré-SAE {self.scaler_path_full} não encontrado. Novo scaler instanciado.")

    def save_scaler_pre_sae(self):
        self._update_paths_for_symbol() 
        if self.is_scaler_pre_sae_fitted and hasattr(self.scaler_pre_sae, 'mean_'):
            try:
                joblib.dump(self.scaler_pre_sae, self.scaler_path_full)
                logger.info(f"SAE: Scaler pré-SAE salvo em {self.scaler_path_full}")
            except Exception as e:
                logger.error(f"SAE: Erro ao salvar scaler pré-SAE: {e}")
        else:
            logger.warning("SAE: Scaler pré-SAE não ajustado. Não foi salvo.")

    def _build_model_with_input_dim(self, input_dim: int):
        if input_dim <= 0:
            logger.error(f"SAE: Tentativa de construir modelo com input_dim inválido: {input_dim}")
            return False
            
        self.input_dim_actual = input_dim
        concrete_layer_dims = [self.input_dim_actual] + [ld for ld in self.layer_dimensions_config_template[1:] if isinstance(ld, int) and ld > 0]
        
        if len(concrete_layer_dims) < 2 or concrete_layer_dims[0] != self.input_dim_actual:
             logger.error(f"SAE: Configuração de LAYER_DIMS inválida após concretização. Dims: {concrete_layer_dims} vs input_dim: {self.input_dim_actual}")
             self.model = None
             return False

        try:
            self.model = StackedAutoencoder(concrete_layer_dims, 
                                            CONFIG["SAE_SETTINGS"]["ACTIVATION_FN"],
                                            CONFIG["SAE_SETTINGS"]["FINAL_ACTIVATION_ENCODER"],
                                            CONFIG["SAE_SETTINGS"]["DROPOUT_P"],
                                            CONFIG["SAE_SETTINGS"]["USE_BATCH_NORM"]).to(self.device)
            self.output_dim_actual = concrete_layer_dims[-1]
            logger.info(f"SAE: Modelo construído/reconstruído com dims: {concrete_layer_dims}. Output dim: {self.output_dim_actual}")
            return True
        except ValueError as ve:
            logger.error(f"SAE: Erro ao construir StackedAutoencoder (ValueError): {ve}. Dims: {concrete_layer_dims}")
            self.model = None
            return False
        except Exception as e:
            logger.error(f"SAE: Erro desconhecido ao construir StackedAutoencoder: {e}. Dims: {concrete_layer_dims}")
            self.model = None
            return False

    def train_sae(self, X_input_features_unscaled: np.ndarray): 
        if X_input_features_unscaled is None or X_input_features_unscaled.ndim != 2 or X_input_features_unscaled.shape[0] < CONFIG["SAE_SETTINGS"]["BATCH_SIZE"]:
            logger.error(f"SAE: Dados de entrada para treino inválidos ou insuficientes. Shape: {X_input_features_unscaled.shape if X_input_features_unscaled is not None else 'None'}")
            return False

        current_data_input_dim = X_input_features_unscaled.shape[1]

        logger.info(f"SAE: Ajustando scaler_pre_sae com dados de shape {X_input_features_unscaled.shape}")
        self.scaler_pre_sae = StandardScaler() 
        X_input_features_scaled = self.scaler_pre_sae.fit_transform(X_input_features_unscaled)
        self.is_scaler_pre_sae_fitted = True
        self.input_dim_actual = current_data_input_dim 
        self.save_scaler_pre_sae() 

        if self.model is None or (self.model and self.model.layer_dims[0] != self.input_dim_actual):
            logger.info(f"SAE: Modelo precisa ser construído ou reconstruído. Input dim atual: {self.input_dim_actual}, Modelo existente input dim: {self.model.layer_dims[0] if self.model else 'N/A'}")
            if not self._build_model_with_input_dim(self.input_dim_actual):
                logger.error("SAE: Falha ao construir modelo durante o treinamento.")
                return False
        
        if self.model is None: 
            logger.error("SAE: Modelo ainda é None após tentativa de construção. Não pode treinar.")
            return False

        X_t = torch.FloatTensor(X_input_features_scaled).to(self.device)
        actual_batch_size = min(CONFIG["SAE_SETTINGS"]["BATCH_SIZE"], len(X_t))
        if actual_batch_size <= 0:
            logger.error(f"SAE: Tamanho do batch inválido ({actual_batch_size}) para o número de amostras ({len(X_t)}).")
            return False
            
        loader = DataLoader(TensorDataset(X_t, X_t), batch_size=actual_batch_size, shuffle=True, drop_last=True if len(X_t) > actual_batch_size else False)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG["SAE_SETTINGS"]["LEARNING_RATE"], weight_decay=CONFIG["SAE_SETTINGS"].get("WEIGHT_DECAY", 1e-5))
        
        logger.info(f"SAE: Iniciando treinamento com {len(X_input_features_scaled)} amostras. Input dim: {self.input_dim_actual}, Output dim (encoded): {self.output_dim_actual}")
        self.model.train()
        for epoch in range(CONFIG["SAE_SETTINGS"]["EPOCHS"]):
            epoch_loss = 0.0
            if not loader: 
                logger.warning(f"SAE Epoch {epoch+1}: DataLoader vazio, pulando época.")
                continue

            for data_batch, target_batch in loader:
                optimizer.zero_grad()
                _, reconstructed_batch = self.model(data_batch)
                loss = criterion(reconstructed_batch, target_batch)
                loss.backward(); optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0
            if (epoch + 1) % 10 == 0 or epoch == CONFIG["SAE_SETTINGS"]["EPOCHS"] -1 : 
                logger.info(f"SAE Epoch {epoch+1}/{CONFIG['SAE_SETTINGS']['EPOCHS']}: Loss={avg_loss:.7f}")
        
        self.model.eval() 
        self.save_sae_model() 
        return True

    def get_encoded_representation(self, X_input_features_unscaled: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None: 
            logger.warning("SAE Enc: Modelo não carregado/treinado. Tentando carregar...")
            if not self.load_sae_model(): 
                 logger.error("SAE Enc: Falha ao carregar modelo. Não é possível codificar.")
                 return None
        
        if not self.is_scaler_pre_sae_fitted:
            logger.error("SAE Enc: Scaler pré-SAE não ajustado. Não é possível escalar e codificar.")
            return None

        if X_input_features_unscaled.shape[1] != self.input_dim_actual: 
            logger.error(f"SAE Enc: Dimensão de entrada dos dados ({X_input_features_unscaled.shape[1]}) != dimensão esperada pelo scaler/modelo ({self.input_dim_actual}).")
            return None
        
        X_input_features_scaled = self.scaler_pre_sae.transform(X_input_features_unscaled)
        
        self.model.eval() # type: ignore
        X_t = torch.FloatTensor(X_input_features_scaled).to(self.device)
        with torch.no_grad(): 
            encoded_output, _ = self.model(X_t) # type: ignore
        return encoded_output.cpu().numpy()

    def save_sae_model(self):
        self._update_paths_for_symbol() 
        if self.model and self.input_dim_actual and self.output_dim_actual and hasattr(self.model, 'layer_dims'): 
            try:
                torch.save({'state_dict': self.model.state_dict(), 
                            'layer_dims': self.model.layer_dims, 
                            'input_dim_actual': self.input_dim_actual,
                            'output_dim_actual': self.output_dim_actual}, 
                           self.model_path_full)
                logger.info(f"SAE: Modelo salvo em {self.model_path_full}")
            except Exception as e_ssm: 
                logger.error(f"SAE: Erro ao salvar modelo: {e_ssm}")
        else:
            logger.warning("SAE: Modelo, input_dim_actual, output_dim_actual ou layer_dims não definidos. Não foi possível salvar.")

    def load_sae_model(self) -> bool:
        self._update_paths_for_symbol() 
        if not os.path.exists(self.model_path_full):
            logger.warning(f"SAE: Arquivo do modelo não encontrado em {self.model_path_full}. Modelo não carregado.")
            self.model = None
            return False
        try:
            data = torch.load(self.model_path_full, map_location=self.device)
            
            loaded_input_dim = data.get('input_dim_actual') 
            loaded_output_dim = data.get('output_dim_actual')
            loaded_layer_dims = data.get('layer_dims')

            if not loaded_input_dim or not loaded_output_dim or not loaded_layer_dims:
                logger.error(f"SAE Load: Informações de dimensão ausentes no arquivo do modelo {self.model_path_full}.")
                self.model = None
                return False

            self.input_dim_actual = loaded_input_dim
            self.output_dim_actual = loaded_output_dim
            
            self.model = StackedAutoencoder(loaded_layer_dims, 
                                            CONFIG["SAE_SETTINGS"]["ACTIVATION_FN"],
                                            CONFIG["SAE_SETTINGS"]["FINAL_ACTIVATION_ENCODER"],
                                            CONFIG["SAE_SETTINGS"]["DROPOUT_P"],
                                            CONFIG["SAE_SETTINGS"]["USE_BATCH_NORM"]).to(self.device)
            self.model.load_state_dict(data['state_dict'])
            self.model.eval()
            logger.info(f"SAE: Modelo carregado de {self.model_path_full}. Input: {self.input_dim_actual}, Output (Encoded): {self.output_dim_actual}, Dims Usadas: {loaded_layer_dims}")
            return True
        except Exception as e_lsm: 
            logger.error(f"SAE: Erro ao carregar modelo de {self.model_path_full}: {e_lsm}", exc_info=True)
            self.model = None
            return False

# --- 9. Aprendizado por Reforço (RL - SAC) ---
def init_weights_rl(m, gain=1.0): 
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None: torch.nn.init.constant_(m.bias, 0)

class ActorNetworkDiscreteSAC(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) 
        )
        self.net.apply(lambda m: init_weights_rl(m, gain=nn.init.calculate_gain('relu')))
        self.net[-1].apply(lambda m: init_weights_rl(m, gain=0.01)) 

    def forward(self, state):
        return self.net(state) 

    def sample_action(self, state, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs) 
        
        if deterministic: 
            action_taken = torch.argmax(action_probs, dim=-1)
        else: 
            action_taken = dist.sample()
            
        log_prob_taken_action = dist.log_prob(action_taken)
        return action_taken, log_prob_taken_action, action_probs

class CriticNetworkDiscreteSAC(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_dim): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) 
        )
        self.net.apply(lambda m: init_weights_rl(m, gain=nn.init.calculate_gain('relu')))

    def forward(self, state): 
        return self.net(state) 

class ReplayBufferRL:
    def __init__(self, capacity: int): 
        self.buffer: deque = deque(maxlen=capacity) 

    def add(self, state: np.ndarray, action_idx: int, reward: float, next_state: np.ndarray, done: bool): 
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size == 0:
            raise ValueError("ReplayBufferRL: Tentativa de amostrar de um buffer vazio ou com tamanho de lote inválido.")

        batch = random.sample(self.buffer, actual_batch_size)
        state, action_idx, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (torch.FloatTensor(state).to(device),
                torch.LongTensor(action_idx).unsqueeze(1).to(device), 
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.BoolTensor(done).unsqueeze(1).to(device)) 

    def __len__(self): return len(self.buffer)

class RLAgentSAC:
    def __init__(self, state_dim: int, action_dim: int, cfg_rl_agent: Dict[str, Any], device_str: str): 
        self.state_dim = state_dim; self.action_dim = action_dim; self.cfg = cfg_rl_agent; self.device = torch.device(device_str)
        
        self.actor: ActorNetworkDiscreteSAC = ActorNetworkDiscreteSAC(state_dim, action_dim, cfg_rl_agent["HIDDEN_DIM"]).to(self.device) 
        self.critic1: CriticNetworkDiscreteSAC = CriticNetworkDiscreteSAC(state_dim, action_dim, cfg_rl_agent["HIDDEN_DIM"]).to(self.device) 
        self.critic2: CriticNetworkDiscreteSAC = CriticNetworkDiscreteSAC(state_dim, action_dim, cfg_rl_agent["HIDDEN_DIM"]).to(self.device) 
        self.target_critic1: CriticNetworkDiscreteSAC = CriticNetworkDiscreteSAC(state_dim, action_dim, cfg_rl_agent["HIDDEN_DIM"]).to(self.device) 
        self.target_critic2: CriticNetworkDiscreteSAC = CriticNetworkDiscreteSAC(state_dim, action_dim, cfg_rl_agent["HIDDEN_DIM"]).to(self.device) 
        
        self.target_critic1.load_state_dict(self.critic1.state_dict()); self.target_critic1.eval()
        self.target_critic2.load_state_dict(self.critic2.state_dict()); self.target_critic2.eval()

        self.actor_optimizer: optim.Adam = optim.Adam(self.actor.parameters(), lr=cfg_rl_agent["LR_ACTOR"]) 
        self.critic_optimizer: optim.Adam = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=cfg_rl_agent["LR_CRITIC"]) 

        self.log_alpha: torch.Tensor = torch.tensor(np.log(cfg_rl_agent["ALPHA"]), dtype=torch.float32, requires_grad=True, device=self.device) 
        self.alpha_optimizer: optim.Adam = optim.Adam([self.log_alpha], lr=cfg_rl_agent["LR_ALPHA"]) 
        self.target_entropy: float = -math.log(1.0 / action_dim) * 0.98  
        
        self.replay_buffer: ReplayBufferRL = ReplayBufferRL(cfg_rl_agent["REPLAY_BUFFER_SIZE"]) 
        self.total_steps_trained: int = 0 
        self.env_state_normalizer: Optional[StandardScaler] = None 
        self._load_env_state_normalizer()
        self.load_models() 

    def _get_model_file_path(self, base_filename: str) -> str:
        symbol = CONFIG["MT5_SETTINGS"]["SYMBOL"]
        model_dir_symbol = os.path.join(CONFIG["GENERAL_SETTINGS"]["MODEL_DIR"], symbol)
        os.makedirs(model_dir_symbol, exist_ok=True)
        return os.path.join(model_dir_symbol, base_filename)

    def _load_env_state_normalizer(self):
        path = self._get_model_file_path(self.cfg["ENV_STATE_NORMALIZATION_PARAMS_PATH"])
        if os.path.exists(path):
            try: 
                self.env_state_normalizer = joblib.load(path)
                logger.info(f"RLAgentSAC: Normalizador de estado do ambiente carregado de {path}")
            except Exception as e: 
                logger.error(f"RLAgentSAC: Erro ao carregar normalizador de estado do ambiente: {e}")
                self.env_state_normalizer = None 
        else:
            logger.info(f"RLAgentSAC: Arquivo do normalizador de estado do ambiente {path} não encontrado. Será criado no treino se necessário.")
            self.env_state_normalizer = None

    def _save_env_state_normalizer(self):
        if self.env_state_normalizer:
            path = self._get_model_file_path(self.cfg["ENV_STATE_NORMALIZATION_PARAMS_PATH"])
            try:
                joblib.dump(self.env_state_normalizer, path)
                logger.info(f"RLAgentSAC: Normalizador de estado do ambiente salvo em {path}")
            except Exception as e:
                logger.error(f"RLAgentSAC: Erro ao salvar normalizador de estado do ambiente: {e}")
        else:
            logger.warning("RLAgentSAC: Normalizador de estado do ambiente não definido. Não foi salvo.")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if self.actor is None: 
            logger.warning("RLAgentSAC: Ator não inicializado, selecionando ação aleatória.")
            return random.randint(0, self.action_dim - 1)
        
        if self.env_state_normalizer and hasattr(self.env_state_normalizer, 'mean_'):
            try:
                state = self.env_state_normalizer.transform(state.reshape(1, -1)).flatten()
            except Exception as e:
                logger.warning(f"RLAgentSAC: Erro ao normalizar estado para seleção de ação: {e}. Usando estado não normalizado.")
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval() 
        with torch.no_grad():
            action, _, _ = self.actor.sample_action(state_t, deterministic=deterministic)
        self.actor.train() 
        return action.item()

    def add_experience(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.add(state, action_idx, reward, next_state, done)

    def train_agent_step(self) -> Optional[Dict[str, float]]: 
        if len(self.replay_buffer) < self.cfg["SAC_BATCH_SIZE"]: 
            return None
        self.total_steps_trained +=1

        state_b, action_idx_b, reward_b, next_state_b, done_b = self.replay_buffer.sample(self.cfg["SAC_BATCH_SIZE"], self.device)
        
        if self.env_state_normalizer and hasattr(self.env_state_normalizer, 'mean_'):
            try:
                state_b_np = state_b.cpu().numpy()
                next_state_b_np = next_state_b.cpu().numpy()
                
                state_b_norm_np = self.env_state_normalizer.transform(state_b_np)
                next_state_b_norm_np = self.env_state_normalizer.transform(next_state_b_np)
                
                state_b = torch.FloatTensor(state_b_norm_np).to(self.device)
                next_state_b = torch.FloatTensor(next_state_b_norm_np).to(self.device)
            except Exception as e:
                logger.warning(f"RLAgentSAC: Erro ao normalizar batch de estados durante treino: {e}. Usando estados não normalizados.")

        with torch.no_grad():
            _, next_log_pi_b, next_action_probs_b = self.actor.sample_action(next_state_b) 
            
            q1_target_next_all_actions = self.target_critic1(next_state_b)
            q2_target_next_all_actions = self.target_critic2(next_state_b)
            q_target_next_min_all_actions = torch.min(q1_target_next_all_actions, q2_target_next_all_actions)
            
            v_target_next = torch.sum(next_action_probs_b * (q_target_next_min_all_actions - self.log_alpha.exp() * next_log_pi_b.unsqueeze(1)), dim=1, keepdim=True)
            
            q_target = reward_b + (~done_b) * self.cfg["GAMMA"] * v_target_next 
            
        q1_current_all_actions = self.critic1(state_b)
        q2_current_all_actions = self.critic2(state_b)
        
        q1_current = q1_current_all_actions.gather(1, action_idx_b)
        q2_current = q2_current_all_actions.gather(1, action_idx_b)

        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        critic_loss_total = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad(); critic_loss_total.backward(); self.critic_optimizer.step()

        actor_loss_val = torch.tensor(0.0, device=self.device) 
        alpha_loss_val = torch.tensor(0.0, device=self.device) 

        if self.total_steps_trained % self.cfg.get("ACTOR_UPDATE_FREQ", 2) == 0:
            _, log_pi_b_actor, action_probs_b_actor = self.actor.sample_action(state_b)
            
            q1_for_actor_all = self.critic1(state_b).detach() 
            q_for_actor_all = q1_for_actor_all 

            actor_loss_val = torch.sum(action_probs_b_actor * (self.log_alpha.exp().detach() * log_pi_b_actor.unsqueeze(1) - q_for_actor_all), dim=1).mean()
            
            self.actor_optimizer.zero_grad(); actor_loss_val.backward(); self.actor_optimizer.step()

            if self.cfg["AUTOMATIC_ENTROPY_TUNING"]:
                alpha_loss_val = -(self.log_alpha.exp() * (log_pi_b_actor.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(); alpha_loss_val.backward(); self.alpha_optimizer.step()

        if self.total_steps_trained % self.cfg["TARGET_UPDATE_INTERVAL"] == 0:
            with torch.no_grad():
                for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(self.cfg["TAU"] * param.data + (1.0 - self.cfg["TAU"]) * target_param.data)
                for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(self.cfg["TAU"] * param.data + (1.0 - self.cfg["TAU"]) * target_param.data)
        
        return {"c_loss": critic_loss_total.item(), "a_loss": actor_loss_val.item(), 
                "alpha_loss": alpha_loss_val.item() if self.cfg["AUTOMATIC_ENTROPY_TUNING"] else 0.0,
                "alpha": self.log_alpha.exp().item()}

    def save_models(self):
        actor_p = self._get_model_file_path(self.cfg["ACTOR_MODEL_PATH"])
        critic_p = self._get_model_file_path(self.cfg["CRITIC_MODEL_PATH"])
        
        if self.actor: torch.save(self.actor.state_dict(), actor_p)
        if self.critic1 and self.critic2:
            torch.save({"critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict()}, critic_p)
        logger.info(f"RL (SAC) Modelos Ator e Crítico salvos.")
        self._save_env_state_normalizer() 

    def load_models(self):
        actor_p = self._get_model_file_path(self.cfg["ACTOR_MODEL_PATH"])
        critic_p = self._get_model_file_path(self.cfg["CRITIC_MODEL_PATH"])
        loaded_actor, loaded_critic = False, False

        if os.path.exists(actor_p): 
            try: 
                self.actor.load_state_dict(torch.load(actor_p, map_location=self.device))
                loaded_actor=True
                logger.info(f"RL (SAC) Modelo Ator carregado de {actor_p}")
            except Exception as e: 
                logger.error(f"Erro ao carregar Ator SAC de {actor_p}: {e}. Pode ser necessário retreinar.")
        else:
            logger.warning(f"RL (SAC) Arquivo do modelo Ator {actor_p} não encontrado.")

        if os.path.exists(critic_p): 
            try:
                critics_state = torch.load(critic_p, map_location=self.device)
                self.critic1.load_state_dict(critics_state["critic1"])
                self.critic2.load_state_dict(critics_state["critic2"])
                self.target_critic1.load_state_dict(self.critic1.state_dict()); self.target_critic1.eval()
                self.target_critic2.load_state_dict(self.critic2.state_dict()); self.target_critic2.eval()
                loaded_critic=True
                logger.info(f"RL (SAC) Modelos Críticos carregados de {critic_p}")
            except Exception as e: 
                logger.error(f"Erro ao carregar Críticos SAC de {critic_p}: {e}. Pode ser necessário retreinar.")
        else:
            logger.warning(f"RL (SAC) Arquivo dos modelos Críticos {critic_p} não encontrado.")

        if not loaded_actor or not loaded_critic:
             logger.warning(f"RL (SAC) Falha ao carregar um ou ambos os modelos (Ator e/ou Crítico). O agente pode não funcionar corretamente sem retreinamento.")
        
        self._load_env_state_normalizer() 
        return loaded_actor and loaded_critic

# --- 10. Ambiente de Trading para RL (`TradingEnvRL`) ---
class TradingEnvRL: 
    def __init__(self, initial_balance: float, 
                 df_historical_renko_features_with_price: pd.DataFrame, 
                 rl_config: Dict[str, Any], 
                 env_state_normalizer_rl_agent: Optional[StandardScaler] = None, 
                 train_normalizer: bool = False): 
        
        self.initial_balance = initial_balance
        self.df_data = df_historical_renko_features_with_price.copy() 
        if not all(col in self.df_data.columns for col in ['time', 'original_open_price', 'original_high_price', 'original_low_price', 'original_close_price', 'renko_atr_14', 'direction']):
            raise ValueError("TradingEnvRL: df_data deve conter colunas de preço original (OHLC), tempo, renko_atr_14 e direction.")

        self.rl_cfg = rl_config
        self.reward_cfg = rl_config["REWARD_CONFIG"]
        self.action_mapping = rl_config["ACTION_MAPPING"]
        self.n_actions = rl_config["N_ACTIONS"]
        self.state_history_len = rl_config["STATE_HISTORY_LEN"]
        self.partial_lot_sizes = rl_config["PARTIAL_LOT_SIZES"] 
        
        self.point_val = symbol_point_global 
        self.digits_val = symbol_digits_global
        self.commission_per_lot_currency = self.reward_cfg.get("COMMISSION_PER_LOT_CURRENCY", 3.5)
        self.slippage_pips_sim = self.reward_cfg.get("SLIPPAGE_ESTIMATED_PIPS", 0.5)

        self.balance: float = 0.0; self.equity: float = 0.0 
        self.open_positions_env: List[Dict[str, Any]] = []  
        self.current_step_idx: int = 0 
        self.trade_history_env: List[Dict[str, Any]] = []  
        self.max_drawdown_percent_episode: float = 0.0 
        self.peak_equity_episode: float = 0.0 

        self.market_feature_columns: List[str] = [
            col for col in self.df_data.columns 
            if col not in ['time', 'open', 'high', 'low', 'close', 
                           'original_open_price', 'original_high_price', 'original_low_price', 'original_close_price', 
                           'tick_volume', 'direction', 'brick_size', 'renko_atr_14'] 
        ]
        if not self.market_feature_columns:
            raise ValueError("TradingEnvRL: Nenhuma coluna de feature de mercado encontrada no df_data.")
        logger.info(f"TradingEnvRL: Colunas de features de mercado identificadas: {self.market_feature_columns}")

        self.env_state_normalizer = env_state_normalizer_rl_agent 
        if train_normalizer and self.env_state_normalizer is None:
            self.env_state_normalizer = StandardScaler() 
            logger.info("TradingEnvRL: Novo StandardScaler instanciado para normalização de estado do ambiente (será treinado).")
        elif train_normalizer and self.env_state_normalizer is not None:
             logger.info("TradingEnvRL: StandardScaler existente fornecido, será re-treinado (fit).")

        if train_normalizer:
            self._fit_env_state_normalizer()

    def _fit_env_state_normalizer(self):
        if not self.env_state_normalizer:
            logger.warning("TradingEnvRL: Normalizador de estado não instanciado. Não pode ser ajustado.")
            return

        logger.info("TradingEnvRL: Ajustando (fit) o normalizador de estado do ambiente...")
        num_samples_for_fit = min(len(self.df_data) - self.state_history_len -1, 50000) 
        if num_samples_for_fit < 100 : 
            logger.warning(f"TradingEnvRL: Dados insuficientes ({num_samples_for_fit}) para ajustar normalizador de estado. Pelo menos 100 são recomendados.")
            self.env_state_normalizer = None 
            return

        sample_states = []
        for i in range(self.state_history_len -1, self.state_history_len -1 + num_samples_for_fit):
            if i >= len(self.df_data): break 
            
            original_current_step_idx = self.current_step_idx
            self.current_step_idx = i 
            
            original_open_positions = self.open_positions_env
            self.open_positions_env = [] 
            
            raw_state_sample = self._get_current_full_state_raw() 
            
            self.open_positions_env = original_open_positions 
            self.current_step_idx = original_current_step_idx 

            if raw_state_sample is not None:
                sample_states.append(raw_state_sample)
        
        if sample_states:
            sample_states_np = np.array(sample_states)
            self.env_state_normalizer.fit(sample_states_np)
            logger.info(f"TradingEnvRL: Normalizador de estado ajustado com {len(sample_states_np)} amostras. Mean: {self.env_state_normalizer.mean_[:3]}..., Scale: {self.env_state_normalizer.scale_[:3]}...")
        else:
            logger.warning("TradingEnvRL: Nenhuma amostra de estado coletada para ajustar o normalizador.")
            self.env_state_normalizer = None 

    def _normalize_state_if_needed(self, state_features: np.ndarray) -> np.ndarray:
        if self.env_state_normalizer and hasattr(self.env_state_normalizer, 'mean_'): 
            try: 
                if state_features.ndim == 1:
                    return self.env_state_normalizer.transform(state_features.reshape(1, -1)).flatten()
                else: 
                    return self.env_state_normalizer.transform(state_features)
            except Exception as e: 
                logger.warning(f"TradingEnvRL: Erro ao normalizar estado: {e}. Usando estado não normalizado."); 
                return state_features
        return state_features

    def _get_market_features_for_state(self) -> Optional[np.ndarray]:
        if self.current_step_idx < self.state_history_len -1 : return None
        
        start_idx = self.current_step_idx - self.state_history_len + 1
        end_idx = self.current_step_idx + 1 
        
        market_features_seq_df = self.df_data.iloc[start_idx : end_idx]
        
        if market_features_seq_df.empty or len(market_features_seq_df) < self.state_history_len: 
            return None
            
        market_features_values = market_features_seq_df[self.market_feature_columns].values
        return market_features_values.flatten() 

    def _get_position_features_for_state(self, current_market_price: float) -> np.ndarray:
        total_buy_lots = sum(p['volume'] for p in self.open_positions_env if p['type'] == 'buy')
        total_sell_lots = sum(p['volume'] for p in self.open_positions_env if p['type'] == 'sell')
        
        avg_buy_entry = sum(p['entry_price'] * p['volume'] for p in self.open_positions_env if p['type'] == 'buy') / (total_buy_lots + 1e-9) if total_buy_lots > 0 else 0
        avg_sell_entry = sum(p['entry_price'] * p['volume'] for p in self.open_positions_env if p['type'] == 'sell') / (total_sell_lots + 1e-9) if total_sell_lots > 0 else 0

        unrealized_pnl_buy = (current_market_price - avg_buy_entry) * total_buy_lots * 100000 * self.point_val if total_buy_lots > 0 else 0
        unrealized_pnl_sell = (avg_sell_entry - current_market_price) * total_sell_lots * 100000 * self.point_val if total_sell_lots > 0 else 0
        
        pos_features_vec = np.zeros(6, dtype=np.float32) 
        max_total_lots_config = self.rl_cfg.get("MAX_TOTAL_POSITION_LOTS", 0.1)
        
        pos_features_vec[0] = total_buy_lots / (max_total_lots_config + 1e-9)
        pos_features_vec[1] = np.clip(unrealized_pnl_buy / (self.balance * 0.05 + 1e-6), -3, 3) if total_buy_lots > 0 else 0
        pos_features_vec[2] = total_sell_lots / (max_total_lots_config + 1e-9)
        pos_features_vec[3] = np.clip(unrealized_pnl_sell / (self.balance * 0.05 + 1e-6), -3, 3) if total_sell_lots > 0 else 0
        pos_features_vec[4] = len(self.open_positions_env) / (self.rl_cfg.get("MAX_PARTIAL_POSITIONS", 3) * 2.0 + 1e-6) 
        
        current_equity_val = self.balance + unrealized_pnl_buy + unrealized_pnl_sell
        pos_features_vec[5] = np.clip((current_equity_val - self.balance) / (self.balance * 0.2 + 1e-6), -2, 2) 
        return pos_features_vec

    def _get_current_full_state_raw(self) -> Optional[np.ndarray]:
        market_features_flat = self._get_market_features_for_state()
        if market_features_flat is None: 
            return None
        
        current_price = self.df_data['original_close_price'].iloc[self.current_step_idx]
        position_features_vec = self._get_position_features_for_state(current_price)
        
        raw_state = np.concatenate((market_features_flat, position_features_vec)).astype(np.float32)
        return raw_state

    def get_current_full_state(self) -> Optional[np.ndarray]: 
        raw_state = self._get_current_full_state_raw()
        if raw_state is None: return None
        return self._normalize_state_if_needed(raw_state)

    def reset(self, start_index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]: 
        self.balance = self.initial_balance; self.equity = self.initial_balance
        self.open_positions_env = []; self.trade_history_env = []
        self.peak_equity_episode = self.initial_balance
        self.max_drawdown_percent_episode = 0.0
        
        min_start_idx = self.state_history_len -1
        max_possible_start_idx = len(self.df_data) - self.rl_cfg.get("MAX_STEPS_PER_EPISODE", 1000) - 5 

        if start_index is not None:
            self.current_step_idx = max(min_start_idx, min(start_index, max_possible_start_idx))
        else: 
            if max_possible_start_idx <= min_start_idx: 
                 self.current_step_idx = min_start_idx
                 if self.current_step_idx < 0 or self.current_step_idx >= len(self.df_data): 
                     logger.error(f"TradingEnvRL: Dataset histórico muito pequeno ({len(self.df_data)}) para iniciar episódio. Min idx: {min_start_idx}")
                     return None, {"status": "error_dataset_too_small", "balance": self.balance, "equity": self.equity}
            else:
                 self.current_step_idx = random.randint(min_start_idx, max_possible_start_idx)
        
        initial_state = self.get_current_full_state()
        info = {"balance": self.balance, "equity": self.equity, "status": "reset_ok" if initial_state is not None else "error_initial_state"}
        return initial_state, info

    def _apply_trade_costs(self, price: float, order_type_str: str, volume: float) -> Tuple[float, float]: 
        slippage_val = self.slippage_pips_sim * self.point_val * random.uniform(0.5, 1.5)
        if order_type_str.startswith('buy'): 
            exec_price = price + slippage_val if order_type_str == 'buy_open' else price - slippage_val
        elif order_type_str.startswith('sell'): 
            exec_price = price - slippage_val if order_type_str == 'sell_open' else price + slippage_val
        else: exec_price = price 

        commission_cost_val = (volume / 1.0) * self.commission_per_lot_currency 
        return exec_price, commission_cost_val

    def _calculate_reward(self, pnl_realized_step: float, unrealized_pnl_change_step: float, action_idx: int, num_trades_this_step: int) -> float:
        reward = 0.0
        reward += pnl_realized_step * self.reward_cfg.get("REALIZED_PNL_FACTOR", 1.0)
        reward += unrealized_pnl_change_step * self.reward_cfg.get("UNREALIZED_PNL_CHANGE_FACTOR", 0.3)
        
        action_str = self.action_mapping.get(action_idx, "INVALID")
        if action_str == "HOLD" and self.open_positions_env:
            reward += self.reward_cfg.get("HOLDING_TIME_OPEN_POS_PENALTY", -0.0001) * len(self.open_positions_env)
        
        if num_trades_this_step > 0: 
            reward += self.reward_cfg.get("TRADE_COUNT_PENALTY_FACTOR", -0.1) * num_trades_this_step 

        self.peak_equity_episode = max(self.peak_equity_episode, self.equity)
        current_dd_abs = self.peak_equity_episode - self.equity
        if current_dd_abs > 0 and self.peak_equity_episode > 0:
            current_dd_percent = current_dd_abs / self.peak_equity_episode
            self.max_drawdown_percent_episode = max(self.max_drawdown_percent_episode, current_dd_percent)
            if current_dd_percent > 0.001: 
                 reward += self.reward_cfg.get("DRAWDOWN_PENALTY_FACTOR", -0.8) * current_dd_percent * self.initial_balance 
        return reward

    def step(self, action_idx: int) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict[str, Any]]: 
        if self.current_step_idx >= len(self.df_data) - 2: 
            next_raw_state = self._get_current_full_state_raw() 
            return self._normalize_state_if_needed(next_raw_state) if next_raw_state is not None else None, 0, True, False, {"status": "end_of_data_early", "balance": self.balance, "equity": self.equity}

        price_before_action = self.df_data['original_close_price'].iloc[self.current_step_idx]
        unrealized_pnl_before = sum(
            (price_before_action - p['entry_price']) * p['volume'] * 100000 * self.point_val if p['type'] == 'buy'
            else (p['entry_price'] - price_before_action) * p['volume'] * 100000 * self.point_val
            for p in self.open_positions_env
        )

        pnl_realized_this_step = 0.0; num_trades_made = 0
        action_str = self.action_mapping.get(action_idx, "INVALID_ACTION")
        
        current_price_for_exec = self.df_data['original_open_price'].iloc[self.current_step_idx + 1] 
        
        lot_to_trade = 0.0 
        is_buy_action = False; is_sell_action = False
        if action_str == "BUY_S": lot_to_trade = self.partial_lot_sizes[0]; is_buy_action = True
        elif action_str == "BUY_M": lot_to_trade = self.partial_lot_sizes[1]; is_buy_action = True
        elif action_str == "BUY_L": lot_to_trade = self.partial_lot_sizes[2]; is_buy_action = True
        elif action_str == "SELL_S": lot_to_trade = self.partial_lot_sizes[0]; is_sell_action = True
        elif action_str == "SELL_M": lot_to_trade = self.partial_lot_sizes[1]; is_sell_action = True
        elif action_str == "SELL_L": lot_to_trade = self.partial_lot_sizes[2]; is_sell_action = True

        total_current_lots_open = sum(p['volume'] for p in self.open_positions_env)

        if (is_buy_action or is_sell_action) and lot_to_trade > 0:
            if total_current_lots_open + lot_to_trade <= self.rl_cfg.get("MAX_TOTAL_POSITION_LOTS", 0.1) + 1e-7: 
                order_type_sim = 'buy_open' if is_buy_action else 'sell_open'
                exec_price, comm_cost = self._apply_trade_costs(current_price_for_exec, order_type_sim, lot_to_trade)
                
                self.balance -= comm_cost
                pnl_realized_this_step -= comm_cost 
                num_trades_made +=1
                
                sl_dist_pips = self.rl_cfg.get("INITIAL_SL_PIP_DISTANCE_RL", 200)
                tp_dist_pips = self.rl_cfg.get("INITIAL_TP_PIP_DISTANCE_RL", 400)
                
                sl_price = exec_price - sl_dist_pips * self.point_val if is_buy_action else exec_price + sl_dist_pips * self.point_val
                tp_price = exec_price + tp_dist_pips * self.point_val if is_buy_action else exec_price - tp_dist_pips * self.point_val
                
                self.open_positions_env.append({
                    'type': 'buy' if is_buy_action else 'sell', 
                    'entry_price': exec_price, 
                    'volume': lot_to_trade, 
                    'sl': sl_price, 'tp': tp_price, 
                    'entry_step_idx': self.current_step_idx,
                    'ticket_sim': time.time_ns() 
                })
            else:
                pass 

        elif action_str == "CLOSE_ALL": 
            if self.open_positions_env:
                for pos in list(self.open_positions_env): 
                    close_type_sim = 'sell_close' if pos['type'] == 'buy' else 'buy_close'
                    exec_price_close, comm_cost_close = self._apply_trade_costs(current_price_for_exec, close_type_sim, pos['volume'])
                    
                    pnl_pos = 0
                    if pos['type'] == 'buy':
                        pnl_pos = (exec_price_close - pos['entry_price']) * pos['volume'] * 100000 * self.point_val
                    else: 
                        pnl_pos = (pos['entry_price'] - exec_price_close) * pos['volume'] * 100000 * self.point_val
                    
                    pnl_realized_this_step += pnl_pos - comm_cost_close
                    self.balance += pnl_pos - comm_cost_close
                    num_trades_made +=1 
                    self.trade_history_env.append({'pnl': pnl_pos - comm_cost_close, 'type': f"{pos['type']}_closed_by_agent", 'ticket_sim': pos['ticket_sim']})
                    self.open_positions_env.remove(pos)
        
        self.current_step_idx += 1
        if self.current_step_idx >= len(self.df_data) -1: 
             next_raw_state_eof = self._get_current_full_state_raw()
             final_reward_eof = self._calculate_reward(pnl_realized_this_step, 0, action_idx, num_trades_made) 
             self.equity = self.balance 
             return self._normalize_state_if_needed(next_raw_state_eof) if next_raw_state_eof is not None else None, final_reward_eof, True, False, {"status":"end_of_data", "balance": self.balance, "equity": self.equity}

        price_after_action_and_step = self.df_data['original_close_price'].iloc[self.current_step_idx] 
        low_price_current_step = self.df_data['original_low_price'].iloc[self.current_step_idx]
        high_price_current_step = self.df_data['original_high_price'].iloc[self.current_step_idx]

        new_open_positions_after_sltp_env = []
        equity_pnl_open_after_action = 0.0
        
        for pos in self.open_positions_env:
            closed_by_sltp_env = False
            cost_to_close_sltp = (pos['volume'] / 1.0) * self.commission_per_lot_currency 

            if pos['type'] == 'buy':
                if pos.get('sl', 0.0) > 0 and low_price_current_step <= pos['sl']:  
                    pnl_realized = (pos['sl'] - pos['entry_price']) * pos['volume'] * 100000 * self.point_val
                    pnl_realized_this_step += pnl_realized - cost_to_close_sltp
                    self.balance += pnl_realized - cost_to_close_sltp
                    closed_by_sltp_env = True; num_trades_made +=1
                    self.trade_history_env.append({'pnl': pnl_realized - cost_to_close_sltp, 'type': 'buy_sl_hit', 'ticket_sim': pos['ticket_sim']})
                elif pos.get('tp', 0.0) > 0 and high_price_current_step >= pos['tp']: 
                    pnl_realized = (pos['tp'] - pos['entry_price']) * pos['volume'] * 100000 * self.point_val
                    pnl_realized_this_step += pnl_realized - cost_to_close_sltp
                    self.balance += pnl_realized - cost_to_close_sltp
                    closed_by_sltp_env = True; num_trades_made +=1
                    self.trade_history_env.append({'pnl': pnl_realized - cost_to_close_sltp, 'type': 'buy_tp_hit', 'ticket_sim': pos['ticket_sim']})
            
            elif pos['type'] == 'sell':
                if pos.get('sl', 0.0) > 0 and high_price_current_step >= pos['sl']: 
                    pnl_realized = (pos['entry_price'] - pos['sl']) * pos['volume'] * 100000 * self.point_val
                    pnl_realized_this_step += pnl_realized - cost_to_close_sltp
                    self.balance += pnl_realized - cost_to_close_sltp
                    closed_by_sltp_env = True; num_trades_made +=1
                    self.trade_history_env.append({'pnl': pnl_realized - cost_to_close_sltp, 'type': 'sell_sl_hit', 'ticket_sim': pos['ticket_sim']})
                elif pos.get('tp', 0.0) > 0 and low_price_current_step <= pos['tp']: 
                    pnl_realized = (pos['entry_price'] - pos['tp']) * pos['volume'] * 100000 * self.point_val
                    pnl_realized_this_step += pnl_realized - cost_to_close_sltp
                    self.balance += pnl_realized - cost_to_close_sltp
                    closed_by_sltp_env = True; num_trades_made +=1
                    self.trade_history_env.append({'pnl': pnl_realized - cost_to_close_sltp, 'type': 'sell_tp_hit', 'ticket_sim': pos['ticket_sim']})
            
            if not closed_by_sltp_env:
                new_open_positions_after_sltp_env.append(pos)
                if pos['type'] == 'buy':
                    equity_pnl_open_after_action += (price_after_action_and_step - pos['entry_price']) * pos['volume'] * 100000 * self.point_val
                else: 
                    equity_pnl_open_after_action += (pos['entry_price'] - price_after_action_and_step) * pos['volume'] * 100000 * self.point_val
        
        self.open_positions_env = new_open_positions_after_sltp_env
        self.equity = self.balance + equity_pnl_open_after_action
        
        unrealized_pnl_change_step = equity_pnl_open_after_action - unrealized_pnl_before 
        
        reward = self._calculate_reward(pnl_realized_this_step, unrealized_pnl_change_step, action_idx, num_trades_made)
        
        done = self.equity <= self.initial_balance * 0.5 

        next_full_state_raw = self._get_current_full_state_raw() if not done else None
        next_full_state_normalized = self._normalize_state_if_needed(next_full_state_raw) if next_full_state_raw is not None else None
        
        if next_full_state_normalized is None and not done: 
            done = True 

        info = {"balance": self.balance, "equity": self.equity, 
                "pnl_realized_step": pnl_realized_this_step, 
                "unrealized_pnl_change": unrealized_pnl_change_step, 
                "action_taken_str": action_str, "num_trades_made_step": num_trades_made,
                "open_positions_count": len(self.open_positions_env)}
        
        return next_full_state_normalized, reward, done, False, info 

# --- 11. Gerenciador de Comunicação MT5 ---
class GerenciadorComunicacaoMT5Adaptativo:
    def __init__(self, cfg_mt5_conn: Dict[str, Any], symbol_principal_str: str): 
        self.cfg = cfg_mt5_conn; self.symbol = symbol_principal_str; self.connected = False
        self.max_retries = self.cfg.get("MAX_CONNECTION_RETRIES", 5) 
        self.retry_delay_sec = self.cfg.get("CONNECTION_RETRY_DELAY_SEC", 20) 

    def conectar(self) -> bool:
        global mt5_initialized_flag 
        if self.connected and mt5.terminal_info() and mt5.terminal_info().connected: 
            mt5_initialized_flag = True
            return True 
        logger.debug("Tentando conectar ao MT5...")
        for attempt in range(self.max_retries):
            login_val = self.cfg.get("LOGIN", 0); password_val = self.cfg.get("PASSWORD", ""); 
            server_val = self.cfg.get("SERVER", ""); path_val = self.cfg.get("PATH", "")
            
            init_params: Dict[str, Any] = {"timeout":30000, "portable": False} 
            if path_val and os.path.exists(path_val): init_params["path"] = path_val
            # Removido o else que logava warning sobre path, pois o MT5 pode encontrar o terminal sem path.

            if login_val != 0 and password_val and server_val: 
                init_params.update({"login":int(login_val), "password":password_val, "server":server_val}) 
            
            if mt5.initialize(**init_params):
                term_info = mt5.terminal_info()
                acc_info = mt5.account_info()
                if term_info and term_info.connected and acc_info: 
                    if mt5.symbol_select(self.symbol, True):
                        logger.info(f"MT5 Conectado: {acc_info.login}@{acc_info.server}. Versão: {mt5.version()}, Símbolo: {self.symbol}"); 
                        self.connected = True
                        mt5_initialized_flag = True
                        return True
                    else: 
                        logger.error(f"MT5 Conectado, mas falha ao selecionar {self.symbol}: {mt5.last_error()}"); 
                        mt5.shutdown()
                        mt5_initialized_flag = False
                else: 
                    logger.error(f"MT5 init OK, mas não conectado. Terminal: {term_info}, Conta: {acc_info}. Erro: {mt5.last_error()}"); 
                    mt5.shutdown()
                    mt5_initialized_flag = False
            else: 
                logger.error(f"MT5 Conexão Falhou (tentativa {attempt+1}/{self.max_retries}): {mt5.last_error()}")
            
            if attempt < self.max_retries -1: time.sleep(self.retry_delay_sec * (attempt + 1)) 
        
        self.connected = False
        mt5_initialized_flag = False
        return False

    def desconectar(self):
        global mt5_initialized_flag
        if self.connected or mt5_initialized_flag : 
            mt5.shutdown()
            self.connected = False
            mt5_initialized_flag = False
            logger.info("MT5 Desconectado.")

    def obter_ticks_mt5(self, count: int, from_datetime_utc: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        if not self.conectar(): 
            logger.error("obter_ticks_mt5: Falha ao conectar ao MT5.")
            return None
        try:
            start_time = from_datetime_utc if from_datetime_utc else datetime.now(timezone.utc)
            logger.debug(f"MT5: Solicitando {count} ticks para {self.symbol} a partir de {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            ticks = mt5.copy_ticks_from(self.symbol, start_time, count, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0: 
                last_err = mt5.last_error()
                logger.warning(f"MT5: Nenhum tick retornado para {self.symbol} (count={count}). Erro MT5: {last_err}")
                return pd.DataFrame() 
            
            df = pd.DataFrame(ticks)
            df['Timestamp'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
            logger.debug(f"MT5: {len(df)} ticks recebidos para {self.symbol}.")
            return df[['Timestamp', 'bid', 'ask', 'volume']].rename(columns={'bid':'Bid', 'ask':'Ask', 'volume':'Volume'})
        except Exception as e: 
            logger.error(f"Erro obter ticks MT5: {e}", exc_info=True)
            return None
    
    def obter_barras_m1_mt5(self, count: int, from_pos: int = 0) -> Optional[pd.DataFrame]:
        if not self.conectar(): 
            logger.error("obter_barras_m1_mt5: Falha ao conectar ao MT5.")
            return None
        try:
            tf_mt5 = CONFIG["MT5_SETTINGS"]["TIMEFRAME_FETCH_MT5"]
            logger.debug(f"MT5: Solicitando {count} barras {CONFIG['MT5_SETTINGS']['TIMEFRAME_STR']} para {self.symbol} a partir da posição {from_pos}.")
            rates = mt5.copy_rates_from_pos(self.symbol, tf_mt5, from_pos, count)
            if rates is None or len(rates) == 0: 
                last_err = mt5.last_error()
                logger.warning(f"MT5: Nenhum rate retornado para {self.symbol} (count={count}, timeframe={tf_mt5}). Erro MT5: {last_err}")
                return pd.DataFrame() 
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            logger.debug(f"MT5: {len(df)} barras recebidas para {self.symbol}.")
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(
                columns={'open':'original_open_price', 'high':'original_high_price', 
                         'low':'original_low_price', 'close':'original_close_price'}
            )
            return df
        except Exception as e: 
            logger.error(f"Erro obter barras M1 MT5: {e}", exc_info=True)
            return None

    def get_current_prices_mt5(self) -> Tuple[Optional[float], Optional[float]]: 
        if not self.conectar(): return None, None
        tick = mt5.symbol_info_tick(self.symbol)
        return (tick.bid, tick.ask) if tick and tick.bid > 0 and tick.ask > 0 else (None, None)

    def get_symbol_info_mt5(self) -> Optional[Any]: 
        if not self.conectar(): return None
        return mt5.symbol_info(self.symbol)

    def get_account_info_mt5(self) -> Optional[Any]: 
        if not self.conectar(): return None
        return mt5.account_info()

    def get_open_positions_mt5(self) -> List[Any]: 
        if not self.conectar(): return []
        try:
            positions = mt5.positions_get(symbol=self.symbol, magic=CONFIG["MT5_SETTINGS"]["MAGIC_NUMBER"])
            return list(positions) if positions else []
        except Exception as e:
            logger.error(f"Erro ao obter posições abertas do MT5: {e}")
            return []

    def send_order_mt5(self, request: Dict[str, Any]) -> Optional[Any]: 
        if not self.conectar(): 
            logger.error("MT5 não conectado. Não é possível enviar ordem.")
            return None
        logger.info(f"Enviando Ordem MT5: {request}")
        try: 
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"Falha ao enviar ordem MT5 (resultado None). Erro MT5: {mt5.last_error()}. Request: {request}")
                return None
            if result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != mt5.TRADE_RETCODE_PLACED:
                 logger.error(f"Ordem MT5 não executada com sucesso. Retcode: {result.retcode}, Comentário: {result.comment}. Request: {request}")
            return result
        except Exception as e: 
            logger.error(f"Exceção ao enviar ordem MT5: {e}. Request: {request}"); return None
    
    def close_position_mt5(self, ticket: int, volume: float, position_type: int, deviation: int = 10) -> Optional[Any]: 
        if not self.conectar(): return None
        s_info = self.get_symbol_info_mt5()
        current_bid, current_ask = self.get_current_prices_mt5()
        if not s_info or current_bid is None or current_ask is None : 
            logger.error("Não foi possível obter symbol_info ou preços atuais para fechar posição.")
            return None 
        
        price = current_bid if position_type == mt5.ORDER_TYPE_BUY else current_ask 
        order_type_close = mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": volume, 
                   "type": order_type_close, "position": ticket, "price": price, 
                   "deviation": deviation, "magic": CONFIG["MT5_SETTINGS"]["MAGIC_NUMBER"],
                   "comment": f"RL Close Pos {ticket}", "type_time": mt5.ORDER_TIME_GTC, 
                   "type_filling": s_info.filling_mode} 
        return self.send_order_mt5(request)

    def modify_position_mt5(self, ticket: int, sl_price: Optional[float], tp_price: Optional[float]) -> Optional[Any]: 
        if not self.conectar(): return None
        s_info = self.get_symbol_info_mt5();
        if not s_info: 
            logger.error("Não foi possível obter symbol_info para modificar posição.")
            return None
            
        request: Dict[str, Any] = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": self.symbol} 
        if sl_price is not None and sl_price > 1e-9 : request["sl"] = round(sl_price, s_info.digits)
        if tp_price is not None and tp_price > 1e-9 : request["tp"] = round(tp_price, s_info.digits)
        
        if "sl" not in request and "tp" not in request: 
            logger.info(f"Modificar Posição {ticket}: Nenhum SL ou TP válido fornecido.")
            return None 
        return self.send_order_mt5(request)

# --- 12. Lógica de Treinamento Inicial ---
def obter_dados_historicos_para_treinamento_inicial() -> Optional[pd.DataFrame]:
    global feature_engineer_instance, renko_builder_instance, renko_brick_optimizer_instance, last_calculated_brick_size_global
    
    cfg_mt5 = CONFIG["MT5_SETTINGS"]
    data_source = cfg_mt5["DATA_SOURCE"]
    symbol = cfg_mt5["SYMBOL"]
    logger.info(f"Iniciando obtenção de dados históricos para treinamento inicial. Fonte: {data_source}, Símbolo: {symbol}")

    df_ticks_historicos = pd.DataFrame()

    if data_source == "EXNESS_TICKS":
        anos_lookback = CONFIG["MT5_SETTINGS"]["EXNESS_TICKS_CONFIG"].get("HISTORICAL_DATA_YEARS_FOR_INITIAL_TRAIN", 1)
        num_meses_total = anos_lookback * 12
        data_atual = datetime.now(timezone.utc)
        dfs_mensais = []
        for i in range(num_meses_total):
            data_alvo = data_atual - timedelta(days=i * 30.5) 
            ano, mes = data_alvo.year, data_alvo.month
            logger.info(f"Tentando baixar/ler dados Exness para {ano}-{mes:02d}...")
            caminho_csv = baixar_e_extrair_ticks_exness(symbol, ano, mes)
            if caminho_csv:
                df_mes = ler_ticks_exness_csv(caminho_csv)
                if df_mes is not None and not df_mes.empty:
                    dfs_mensais.append(df_mes)
                    logger.info(f"Dados de {ano}-{mes:02d} carregados: {len(df_mes)} ticks.")
                else: logger.warning(f"Nenhum dado em {caminho_csv} ou falha na leitura.")
            else: logger.warning(f"Falha ao baixar/localizar CSV para {ano}-{mes:02d}.")
            if len(dfs_mensais) > 0 and sum(len(df) for df in dfs_mensais) > CONFIG["MT5_SETTINGS"]["MT5_TICKS_CONFIG"].get("LOOKBACK_TICKS_INITIAL_TRAIN", 2000000) * 1.5 : 
                logger.info(f"Volume de ticks suficiente ({sum(len(df) for df in dfs_mensais)}) alcançado para treino inicial com Exness.")
                break 
        if dfs_mensais:
            df_ticks_historicos = pd.concat(dfs_mensais, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
    
    elif data_source == "MT5_TICKS":
        if comm_manager_instance:
            num_ticks_treino = CONFIG["MT5_SETTINGS"]["MT5_TICKS_CONFIG"]["LOOKBACK_TICKS_INITIAL_TRAIN"]
            logger.info(f"Buscando {num_ticks_treino} ticks do MT5 para treino inicial...")
            df_ticks_historicos = comm_manager_instance.obter_ticks_mt5(num_ticks_treino)
    
    elif data_source == "MT5_BARS":
        if comm_manager_instance:
            num_barras_treino = CONFIG["MT5_SETTINGS"]["MT5_BARS_CONFIG"]["LOOKBACK_BARS_M1_INITIAL_TRAIN"]
            logger.info(f"Buscando {num_barras_treino} barras M1 do MT5 para treino inicial (serão convertidas para pseudo-ticks)...")
            df_barras_m1 = comm_manager_instance.obter_barras_m1_mt5(num_barras_treino)
            if df_barras_m1 is not None and not df_barras_m1.empty:
                pseudo_ticks_list = []
                for _, row in df_barras_m1.iterrows():
                    ts_base = row['time']
                    vol_pseudo = row['tick_volume'] / 4 if row['tick_volume'] > 0 else 1.0
                    pseudo_ticks_list.append({'Timestamp': ts_base, 'Bid': row['original_open_price'], 'Ask': row['original_open_price'], 'Volume': vol_pseudo})
                    pseudo_ticks_list.append({'Timestamp': ts_base + timedelta(seconds=15), 'Bid': row['original_high_price'], 'Ask': row['original_high_price'], 'Volume': vol_pseudo})
                    pseudo_ticks_list.append({'Timestamp': ts_base + timedelta(seconds=30), 'Bid': row['original_low_price'], 'Ask': row['original_low_price'], 'Volume': vol_pseudo})
                    pseudo_ticks_list.append({'Timestamp': ts_base + timedelta(seconds=45), 'Bid': row['original_close_price'], 'Ask': row['original_close_price'], 'Volume': vol_pseudo})
                df_ticks_historicos = pd.DataFrame(pseudo_ticks_list).sort_values(by="Timestamp").reset_index(drop=True)
    else:
        logger.error(f"Fonte de dados para treinamento inicial desconhecida: {data_source}")
        return None

    if df_ticks_historicos is None or df_ticks_historicos.empty:
        logger.error("Nenhum dado tick histórico pôde ser obtido para o treinamento inicial.")
        return None
    logger.info(f"Total de {len(df_ticks_historicos)} ticks brutos obtidos para treinamento.")

    if renko_brick_optimizer_instance:
        logger.info("Otimizando tamanho do brick Renko para dados históricos...")
        sample_size_opt = min(len(df_ticks_historicos), CONFIG["RENKO_SETTINGS"]["OPTIMIZATION_LOOKBACK_TICKS_RETRAIN"])
        sample_ticks_for_opt = df_ticks_historicos.sample(n=sample_size_opt, random_state=CONFIG["GENERAL_SETTINGS"]["RANDOM_SEED"]) if len(df_ticks_historicos) > sample_size_opt else df_ticks_historicos
        
        optimal_brick = renko_brick_optimizer_instance.find_optimal_brick_size(sample_ticks_for_opt)
        if optimal_brick and optimal_brick > 0:
            last_calculated_brick_size_global = optimal_brick
            logger.info(f"Brick Renko otimizado para treino: {last_calculated_brick_size_global:.{symbol_digits_global}f}")
        else:
            logger.warning("Falha ao otimizar brick Renko para treino. Usando fallback ou valor anterior.")
            if last_calculated_brick_size_global is None or last_calculated_brick_size_global <=0: 
                 last_calculated_brick_size_global = symbol_point_global * 20 
                 logger.info(f"Brick Renko (Fallback para treino): {last_calculated_brick_size_global:.{symbol_digits_global}f}")

    if renko_builder_instance is None: renko_builder_instance = DynamicRenkoBuilder()
    if last_calculated_brick_size_global is None or last_calculated_brick_size_global <= 0:
        logger.error("Tamanho do brick Renko inválido ou não definido antes de gerar Renko para treino.")
        return None
        
    logger.info(f"Gerando barras Renko para treinamento com brick: {last_calculated_brick_size_global:.{symbol_digits_global}f}...")
    df_renko_treino = renko_builder_instance.calculate_renko_from_ticks(df_ticks_historicos, last_calculated_brick_size_global)
    if df_renko_treino.empty:
        logger.error("Nenhuma barra Renko gerada a partir dos dados históricos.")
        return None
    logger.info(f"{len(df_renko_treino)} barras Renko geradas para treinamento.")

    if feature_engineer_instance is None: 
        logger.error("Instância de FeatureEngineeringAdaptive não encontrada. Não pode processar features.")
        return None
        
    logger.info("Adicionando indicadores técnicos às barras Renko de treinamento...")
    df_renko_com_features_treino = feature_engineer_instance.add_technical_indicators(df_renko_treino)
    if df_renko_com_features_treino.empty or feature_engineer_instance.final_feature_columns is None or not feature_engineer_instance.final_feature_columns :
        logger.error("Falha ao adicionar indicadores ou nenhuma feature final foi definida.")
        return None
    logger.info(f"{len(df_renko_com_features_treino)} barras Renko com features. {len(feature_engineer_instance.final_feature_columns)} features finais: {feature_engineer_instance.final_feature_columns[:5]}...")

    logger.info("Treinando o scaler principal do FeatureEngineeringAdaptive com os dados de treinamento...")
    scaled_features_for_fe_scaler_training = feature_engineer_instance.prepare_features_for_model(df_renko_com_features_treino, for_training=True)
    if scaled_features_for_fe_scaler_training is None:
        logger.error("Falha ao treinar o scaler do FeatureEngineeringAdaptive.")
        return None
    logger.info("Scaler do FeatureEngineeringAdaptive treinado e salvo.")
    
    return df_renko_com_features_treino 

def orchestrate_initial_models_training():
    global feature_engineer_instance, sae_handler_instance, rl_agent_instance
    logger.info("Orquestrando treinamento inicial dos modelos...")

    df_dados_completos_para_treino = obter_dados_historicos_para_treinamento_inicial()
    if df_dados_completos_para_treino is None or df_dados_completos_para_treino.empty:
        logger.error("Treinamento inicial abortado: Falha ao obter ou processar dados históricos.")
        return False
    
    scaled_renko_features_array = feature_engineer_instance.prepare_features_for_model(df_dados_completos_para_treino, for_training=False)
    if scaled_renko_features_array is None:
        logger.error("Treinamento inicial abortado: Falha ao obter features escaladas do FeatureEngineer.")
        return False
    
    logger.info(f"Shape das features escaladas pelo FeatureEngineer para treino: {scaled_renko_features_array.shape}")

    sae_foi_treinado_agora = False
    if CONFIG["SAE_SETTINGS"]["ENABLED"]:
        if sae_handler_instance is None:
            logger.error("SAE habilitado, mas instância do SAEHandler é None. Abortando treino SAE.")
            return False 

        sae_model_path_full = sae_handler_instance.model_path_full 
        if not os.path.exists(sae_model_path_full) or not os.path.exists(sae_handler_instance.scaler_path_full):
            logger.info(f"Modelo SAE ({sae_model_path_full}) ou scaler pré-SAE ({sae_handler_instance.scaler_path_full}) não encontrado. Iniciando treinamento do SAE...")
            
            min_bars_sae = CONFIG["SAE_SETTINGS"]["TRAIN_DATA_MIN_RENKO_BARS"]
            if len(scaled_renko_features_array) < min_bars_sae:
                logger.error(f"Dados insuficientes para treinar SAE. Necessário: {min_bars_sae}, Disponível: {len(scaled_renko_features_array)}")
            else:
                if sae_handler_instance.train_sae(scaled_renko_features_array): 
                    logger.info("Treinamento do SAE concluído com sucesso.")
                    sae_foi_treinado_agora = True
                    sae_handler_instance.load_sae_model()
                    sae_handler_instance.load_scaler_pre_sae() 
                else:
                    logger.error("Falha no treinamento do SAE.")
        else:
            logger.info(f"Modelo SAE e scaler pré-SAE já existem em {sae_model_path_full} e {sae_handler_instance.scaler_path_full}. Pulando treinamento do SAE.")
            if sae_handler_instance.model is None: sae_handler_instance.load_sae_model()
            if not sae_handler_instance.is_scaler_pre_sae_fitted: sae_handler_instance.load_scaler_pre_sae()

    features_para_rl_env = scaled_renko_features_array 
    if CONFIG["SAE_SETTINGS"]["ENABLED"] and sae_handler_instance and sae_handler_instance.model and sae_handler_instance.is_scaler_pre_sae_fitted:
        logger.info("SAE está habilitado e modelo/scaler carregados. Gerando representações codificadas para o RL.")
        encoded_features = sae_handler_instance.get_encoded_representation(scaled_renko_features_array)
        if encoded_features is not None:
            features_para_rl_env = encoded_features
            logger.info(f"Usando features codificadas pelo SAE para o ambiente RL. Shape: {features_para_rl_env.shape}")
        else:
            logger.warning("Falha ao obter features codificadas do SAE. Usando features escaladas do FeatureEngineer para RL.")
    else:
        logger.info("SAE não habilitado ou modelo/scaler não disponível. Usando features escaladas do FeatureEngineer para RL.")

    if features_para_rl_env.shape[0] != len(df_dados_completos_para_treino):
        logger.error(f"Discrepância no número de amostras entre features para RL ({features_para_rl_env.shape[0]}) e dados originais ({len(df_dados_completos_para_treino)}). Abortando treino RL.")
        return False

    num_market_features_rl = features_para_rl_env.shape[1]
    market_feature_cols_rl_env = [f"mkt_feat_{i}" for i in range(num_market_features_rl)]
    df_features_para_rl_env = pd.DataFrame(features_para_rl_env, columns=market_feature_cols_rl_env, index=df_dados_completos_para_treino.index)

    cols_essenciais_para_env = ['time', 'original_open_price', 'original_high_price', 'original_low_price', 'original_close_price', 'renko_atr_14', 'direction']
    df_para_rl_env_final = pd.concat([df_dados_completos_para_treino[cols_essenciais_para_env], df_features_para_rl_env], axis=1)
    df_para_rl_env_final = df_para_rl_env_final.dropna() 

    if df_para_rl_env_final.empty:
        logger.error("DataFrame final para o ambiente RL está vazio após processamento. Abortando treino RL.")
        return False

    if CONFIG["RL_AGENT_SETTINGS"]["ENABLED"]:
        if rl_agent_instance is None: 
            logger.info("Instância do RLAgentSAC é None. Tentando criar agora com dimensões atualizadas...")
            market_features_dim_rl = features_para_rl_env.shape[1]
            pos_features_dim_rl = 6 
            state_dim_rl_recalc = CONFIG["RL_AGENT_SETTINGS"]["STATE_HISTORY_LEN"] * market_features_dim_rl + pos_features_dim_rl
            
            if state_dim_rl_recalc <= pos_features_dim_rl:
                 logger.error(f"Dimensão do estado RL recalculada é inválida: {state_dim_rl_recalc}. Abortando.")
                 return False

            rl_agent_instance = RLAgentSAC(state_dim_rl_recalc, CONFIG["RL_AGENT_SETTINGS"]["N_ACTIONS"], CONFIG["RL_AGENT_SETTINGS"], CONFIG["GENERAL_SETTINGS"]["DEVICE"])
            logger.info(f"Nova instância RLAgentSAC criada com state_dim: {state_dim_rl_recalc}")

        actor_path = rl_agent_instance._get_model_file_path(CONFIG["RL_AGENT_SETTINGS"]["ACTOR_MODEL_PATH"])
        critic_path = rl_agent_instance._get_model_file_path(CONFIG["RL_AGENT_SETTINGS"]["CRITIC_MODEL_PATH"])
        env_norm_path = rl_agent_instance._get_model_file_path(CONFIG["RL_AGENT_SETTINGS"]["ENV_STATE_NORMALIZATION_PARAMS_PATH"])

        if not os.path.exists(actor_path) or not os.path.exists(critic_path) or not os.path.exists(env_norm_path):
            logger.info("Modelos RL (Ator/Crítico) ou normalizador de ambiente não encontrados. Iniciando treinamento do Agente RL...")
            
            min_bars_rl = CONFIG["RL_AGENT_SETTINGS"]["TRAIN_DATA_MIN_RENKO_BARS_RL"]
            if len(df_para_rl_env_final) < min_bars_rl:
                logger.error(f"Dados insuficientes para treinar Agente RL. Necessário: {min_bars_rl}, Disponível: {len(df_para_rl_env_final)}")
            else:
                trade_env_rl_train = TradingEnvRL(
                    initial_balance=CONFIG["RL_AGENT_SETTINGS"]["REWARD_CONFIG"]["INITIAL_SIM_BALANCE"],
                    df_historical_renko_features_with_price=df_para_rl_env_final,
                    rl_config=CONFIG["RL_AGENT_SETTINGS"],
                    env_state_normalizer_rl_agent=rl_agent_instance.env_state_normalizer, 
                    train_normalizer=True 
                )
                if trade_env_rl_train.env_state_normalizer is not None and hasattr(trade_env_rl_train.env_state_normalizer, 'mean_'):
                    rl_agent_instance.env_state_normalizer = trade_env_rl_train.env_state_normalizer
                    rl_agent_instance._save_env_state_normalizer() 
                    logger.info("Normalizador de estado do ambiente RL treinado e associado ao agente.")
                else:
                    logger.warning("Normalizador de estado do ambiente RL não foi treinado ou é None após a inicialização do TradingEnvRL.")

                logger.info(f"Iniciando loop de treinamento RL para {CONFIG['RL_AGENT_SETTINGS']['TRAINING_EPISODES']} episódios...")
                total_steps_rl_train = 0
                for episode in range(CONFIG["RL_AGENT_SETTINGS"]["TRAINING_EPISODES"]):
                    state_rl, info_reset = trade_env_rl_train.reset()
                    if state_rl is None:
                        logger.error(f"Episódio {episode+1}: Falha ao resetar ambiente. Pulando episódio.")
                        continue
                    
                    episode_reward = 0; episode_steps = 0
                    for step_num in range(CONFIG["RL_AGENT_SETTINGS"]["MAX_STEPS_PER_EPISODE"]):
                        if state_rl is None: 
                            logger.warning(f"Episódio {episode+1}, Step {step_num+1}: Estado é None antes da ação. Interrompendo episódio.")
                            break

                        action_rl = rl_agent_instance.select_action(state_rl, deterministic=False)
                        
                        next_state_rl, reward_rl, done_rl, _, info_step = trade_env_rl_train.step(action_rl)
                        
                        if state_rl is not None and next_state_rl is not None : 
                             rl_agent_instance.add_experience(state_rl, action_rl, reward_rl, next_state_rl, done_rl)
                        elif state_rl is not None and done_rl and next_state_rl is None: 
                             rl_agent_instance.add_experience(state_rl, action_rl, reward_rl, state_rl, done_rl) 

                        state_rl = next_state_rl
                        episode_reward += reward_rl
                        total_steps_rl_train += 1
                        episode_steps += 1

                        if total_steps_rl_train >= CONFIG["RL_AGENT_SETTINGS"]["LEARNING_STARTS_AFTER_STEPS"]:
                            loss_info = rl_agent_instance.train_agent_step()
                        
                        if done_rl or state_rl is None:
                            break
                    
                    logger.info(f"Episódio RL {episode+1}/{CONFIG['RL_AGENT_SETTINGS']['TRAINING_EPISODES']} concluído. Recompensa: {episode_reward:.2f}, Steps: {episode_steps}, Total Steps Treino: {total_steps_rl_train}")
                    if (episode + 1) % 50 == 0: 
                        rl_agent_instance.save_models()
                
                rl_agent_instance.save_models() 
                logger.info("Treinamento do Agente RL concluído.")
                rl_agent_instance.load_models()
        else:
            logger.info("Modelos RL e normalizador de ambiente já existem. Pulando treinamento RL.")
            if rl_agent_instance.actor is None: 
                rl_agent_instance.load_models()
    
    logger.info("Orquestração do treinamento inicial concluída.")
    return True

# --- 13. Loop Principal e Execução ---
def run_adaptive_sae_renko_rl_ea():
    global mt5_initialized_flag, symbol_point_global, symbol_digits_global, last_calculated_brick_size_global, last_brick_calc_time_global
    global feature_engineer_instance, sae_handler_instance, rl_agent_instance, renko_builder_instance, renko_brick_optimizer_instance
    global comm_manager_instance 
    global last_retrain_time_global, retraining_in_progress_flag
    global rl_live_balance, rl_live_equity 

    cfg_main = CONFIG["GENERAL_SETTINGS"]; cfg_mt5_ea = CONFIG["MT5_SETTINGS"]; cfg_renko = CONFIG["RENKO_SETTINGS"]
    cfg_feat = CONFIG["FEATURE_ENGINEERING_SETTINGS"]; cfg_sae = CONFIG["SAE_SETTINGS"]; cfg_rl = CONFIG["RL_AGENT_SETTINGS"]
    
    comm_manager_instance = GerenciadorComunicacaoMT5Adaptativo(cfg_mt5_ea, cfg_mt5_ea["SYMBOL"])
    if not comm_manager_instance.conectar(): 
        logger.critical("Falha Conexão MT5 Inicial. Encerrando EA.")
        return

    s_info = comm_manager_instance.get_symbol_info_mt5()
    if not s_info: 
        logger.critical("Falha obter SymbolInfo. Encerrando EA.")
        comm_manager_instance.desconectar()
        return
    symbol_point_global, symbol_digits_global = s_info.point, s_info.digits
    
    acc_info_init = comm_manager_instance.get_account_info_mt5()
    if acc_info_init:
        rl_live_balance = acc_info_init.balance 
        rl_live_equity = acc_info_init.equity
        logger.info(f"Saldo Inicial da Conta MT5: {rl_live_balance:.2f}, Equity: {rl_live_equity:.2f}")
    else:
        logger.warning("Não foi possível obter informações da conta MT5. Usando saldo/equity simulado inicial da config.")

    renko_builder_instance = DynamicRenkoBuilder()
    if cfg_renko["OPTIMIZER_ENABLED"]:
        renko_brick_optimizer_instance = RenkoBrickOptimizer(symbol_point_global, symbol_digits_global,
                                                             cfg_renko["OPTIMIZATION_MIN_BRICK_ABS_POINTS"] * symbol_point_global,
                                                             cfg_renko["OPTIMIZATION_MAX_BRICK_ABS_POINTS"] * symbol_point_global,
                                                             cfg_renko["OPTIMIZATION_BRICK_TEST_STEPS"],
                                                             cfg_renko["OPTIMIZATION_REVERSAL_PENALTY_FACTOR"])
    
    feature_engineer_instance = FeatureEngineeringAdaptive(
        cfg_feat["SCALER_RENKO_FEATURES_PATH"], 
        cfg_feat["FEATURE_COLUMNS"]
    )
    
    sae_input_dim_from_feat_eng = len(feature_engineer_instance.final_feature_columns if feature_engineer_instance.final_feature_columns else cfg_feat["FEATURE_COLUMNS"])
    if sae_input_dim_from_feat_eng == 0 and feature_engineer_instance.is_scaler_fitted: # Se scaler foi carregado e tem features
         sae_input_dim_from_feat_eng = feature_engineer_instance.scaler.n_features_in_
    elif sae_input_dim_from_feat_eng == 0:
        logger.error("FEATURE_COLUMNS está vazia ou FeatureEngineer não conseguiu definir colunas finais. Não é possível inicializar SAE ou RL Agent.")
        comm_manager_instance.desconectar(); return

    if cfg_sae["ENABLED"]:
        sae_layer_dims_template = cfg_sae["LAYER_DIMS"] 
        sae_handler_instance = SAEHandler(cfg_sae["MODEL_PATH"], cfg_sae["SCALER_PRE_SAE_PATH"], 
                                          sae_layer_dims_template, cfg_main["DEVICE"])
    else:
        sae_handler_instance = None 
    
    market_features_dim_for_rl = sae_input_dim_from_feat_eng 
    if cfg_sae["ENABLED"] and sae_handler_instance and sae_handler_instance.model and sae_handler_instance.output_dim_actual:
        market_features_dim_for_rl = sae_handler_instance.output_dim_actual
        logger.info(f"RL Agent usará dimensão da camada de código SAE ({market_features_dim_for_rl}) para features de mercado.")
    elif cfg_sae["ENABLED"]:
        logger.warning(f"RL Agent: SAE HABILITADO mas modelo/output_dim SAE não disponível inicialmente. Usando dimensão de features base ({market_features_dim_for_rl}). Será atualizado se SAE for treinado.")
    else:
        logger.info(f"RL Agent: SAE DESABILITADO. Usando dimensão de features base ({market_features_dim_for_rl}).")

    if market_features_dim_for_rl <= 0:
        logger.error(f"CRÍTICO: market_features_dim_for_rl é {market_features_dim_for_rl}. Não é possível inicializar o RL Agent.")
        comm_manager_instance.desconectar(); return 

    state_pos_features_dim_rl = 6 
    state_dim_rl_final = cfg_rl["STATE_HISTORY_LEN"] * market_features_dim_for_rl + state_pos_features_dim_rl
    
    if state_dim_rl_final <= state_pos_features_dim_rl : 
        logger.error(f"Dimensão do estado RL calculada como {state_dim_rl_final} (market_features_dim={market_features_dim_for_rl}), o que é inválido.")
        comm_manager_instance.desconectar(); return
    
    if cfg_rl["ENABLED"]:
        rl_agent_instance = RLAgentSAC(state_dim_rl_final, cfg_rl["N_ACTIONS"], cfg_rl, cfg_main["DEVICE"])
    else:
        rl_agent_instance = None

    # Checagem se modelos existem ANTES de decidir sobre o treinamento inicial
    models_exist = True
    if feature_engineer_instance and not feature_engineer_instance.is_scaler_fitted: models_exist = False
    if sae_handler_instance and (not sae_handler_instance.model or not sae_handler_instance.is_scaler_pre_sae_fitted): models_exist = False
    if rl_agent_instance:
        actor_p = rl_agent_instance._get_model_file_path(cfg_rl["ACTOR_MODEL_PATH"])
        critic_p = rl_agent_instance._get_model_file_path(cfg_rl["CRITIC_MODEL_PATH"])
        env_norm_p = rl_agent_instance._get_model_file_path(cfg_rl["ENV_STATE_NORMALIZATION_PARAMS_PATH"])
        if not os.path.exists(actor_p) or not os.path.exists(critic_p) or not os.path.exists(env_norm_p) : models_exist = False


    if cfg_main["INITIAL_TRAIN_ON_STARTUP"]:
        if not models_exist:
            logger.info("INITIAL_TRAIN_ON_STARTUP é True e um ou mais modelos/scalers estão ausentes. Iniciando treinamento...")
            training_successful = orchestrate_initial_models_training()
            if training_successful:
                logger.info("Treinamento inicial concluído. Recarregando componentes para garantir consistência.")
                feature_engineer_instance = FeatureEngineeringAdaptive(cfg_feat["SCALER_RENKO_FEATURES_PATH"], cfg_feat["FEATURE_COLUMNS"])
                if cfg_sae["ENABLED"]:
                    sae_handler_instance = SAEHandler(cfg_sae["MODEL_PATH"], cfg_sae["SCALER_PRE_SAE_PATH"], cfg_sae["LAYER_DIMS"], cfg_main["DEVICE"])
                    if sae_handler_instance.model and sae_handler_instance.output_dim_actual: # Se SAE foi treinado e tem output_dim
                        new_market_features_dim_for_rl = sae_handler_instance.output_dim_actual
                        if new_market_features_dim_for_rl != market_features_dim_for_rl:
                             logger.info(f"Dimensão de output do SAE mudou após treino: {market_features_dim_for_rl} -> {new_market_features_dim_for_rl}. Recriando RL Agent.")
                             market_features_dim_for_rl = new_market_features_dim_for_rl
                             state_dim_rl_final = cfg_rl["STATE_HISTORY_LEN"] * market_features_dim_for_rl + state_pos_features_dim_rl
                if cfg_rl["ENABLED"]:
                    rl_agent_instance = RLAgentSAC(state_dim_rl_final, cfg_rl["N_ACTIONS"], cfg_rl, cfg_main["DEVICE"]) # Recria com a dimensão potencialmente nova
                else: rl_agent_instance = None
            else:
                logger.error("Treinamento inicial falhou. EA pode não funcionar corretamente.")
        else:
            logger.info("INITIAL_TRAIN_ON_STARTUP é True, mas todos os modelos/scalers necessários já existem. Pulando treinamento.")
            # Carregar modelos existentes (já feito na instanciação dos handlers, mas pode reforçar)
            if feature_engineer_instance: feature_engineer_instance.load_scaler()
            if sae_handler_instance: sae_handler_instance.load_sae_model(); sae_handler_instance.load_scaler_pre_sae()
            if rl_agent_instance: rl_agent_instance.load_models()

    elif not models_exist: # INITIAL_TRAIN_ON_STARTUP é False e modelos não existem
        logger.warning("INITIAL_TRAIN_ON_STARTUP é False e um ou mais modelos/scalers estão ausentes.")
        logger.warning("O EA pode não ter a inteligência necessária para operar. Considere habilitar INITIAL_TRAIN_ON_STARTUP=True no arquivo de configuração para treinar os modelos na próxima execução.")
        # Carregar o que puder (os handlers já tentam fazer isso na inicialização)
        if feature_engineer_instance: feature_engineer_instance.load_scaler()
        if sae_handler_instance: sae_handler_instance.load_sae_model(); sae_handler_instance.load_scaler_pre_sae()
        if rl_agent_instance: rl_agent_instance.load_models()
    else: # INITIAL_TRAIN_ON_STARTUP é False e modelos existem
        logger.info("INITIAL_TRAIN_ON_STARTUP é False. Carregando modelos existentes (já tentado na instanciação dos handlers).")
        # Assegurar que estão carregados
        if feature_engineer_instance and not feature_engineer_instance.is_scaler_fitted: feature_engineer_instance.load_scaler()
        if sae_handler_instance and (sae_handler_instance.model is None or not sae_handler_instance.is_scaler_pre_sae_fitted) :
            sae_handler_instance.load_scaler_pre_sae()
            sae_handler_instance.load_sae_model()
        if rl_agent_instance and rl_agent_instance.actor is None:
            rl_agent_instance.load_models()


    df_renko_live_buffer = pd.DataFrame()
    last_brick_calc_time_global = datetime.now(timezone.utc) - timedelta(minutes=cfg_renko["BRICK_CALC_INTERVAL_MINUTES"] + 1)
    last_rl_action_time = time.monotonic() - 10 
    
    logger.info("=== EA SAE-RENKO-RL INICIADO (MODO DE EXECUÇÃO LIVE) ===")
    try: 
        while True: 
            current_time_loop = datetime.now(timezone.utc)
            if not comm_manager_instance or not comm_manager_instance.conectar(): 
                logger.warning("MT5 não conectado no loop principal. Tentando reconectar em 30s...")
                time.sleep(30); continue 

            if cfg_renko["OPTIMIZER_ENABLED"] and renko_brick_optimizer_instance and \
               (current_time_loop - last_brick_calc_time_global) >= timedelta(minutes=cfg_renko["BRICK_CALC_INTERVAL_MINUTES"]):
                logger.info("Iniciando otimização periódica do brick Renko...")
                df_ticks_for_opt = comm_manager_instance.obter_ticks_mt5(cfg_renko["OPTIMIZATION_LOOKBACK_TICKS_LIVE"])
                if df_ticks_for_opt is not None and not df_ticks_for_opt.empty:
                    new_brick = renko_brick_optimizer_instance.find_optimal_brick_size(df_ticks_for_opt)
                    if new_brick and new_brick > 0: 
                        if last_calculated_brick_size_global != new_brick:
                             logger.info(f"Novo brick Renko otimizado: {new_brick:.{symbol_digits_global}f} (anterior: {last_calculated_brick_size_global or 'N/A'})")
                             last_calculated_brick_size_global = new_brick
                             df_renko_live_buffer = pd.DataFrame() 
                else: logger.warning("Não foi possível obter ticks para otimização periódica do brick.")
                last_brick_calc_time_global = current_time_loop
            
            if last_calculated_brick_size_global is None or last_calculated_brick_size_global <= 0:
                df_m1_for_atr_fallback = comm_manager_instance.obter_barras_m1_mt5(cfg_renko.get("FALLBACK_RENKO_ATR_PERIOD", 14) * 3 + 50) 
                if df_m1_for_atr_fallback is not None and not df_m1_for_atr_fallback.empty and len(df_m1_for_atr_fallback) >= cfg_renko.get("FALLBACK_RENKO_ATR_PERIOD", 14):
                    try:
                        atr_val = AverageTrueRange(
                            high=df_m1_for_atr_fallback['original_high_price'],
                            low=df_m1_for_atr_fallback['original_low_price'],
                            close=df_m1_for_atr_fallback['original_close_price'],
                            window=cfg_renko.get("FALLBACK_RENKO_ATR_PERIOD", 14),
                            fillna=False 
                        ).average_true_range().iloc[-1]
                        if atr_val and atr_val > 0:
                            last_calculated_brick_size_global = max(atr_val * cfg_renko.get("FALLBACK_RENKO_BRICK_ATR_FACTOR", 0.1), symbol_point_global * 5)
                        else: last_calculated_brick_size_global = symbol_point_global * 20 
                    except Exception as e_atr_fb:
                        logger.error(f"Erro ao calcular ATR para fallback de brick: {e_atr_fb}")
                        last_calculated_brick_size_global = symbol_point_global * 20
                else: last_calculated_brick_size_global = symbol_point_global * 20 
                logger.info(f"Brick Renko (Fallback/Inicial Live): {last_calculated_brick_size_global:.{symbol_digits_global}f}")
                df_renko_live_buffer = pd.DataFrame() 

            df_ticks_current_live = comm_manager_instance.obter_ticks_mt5(cfg_mt5_ea["MT5_TICKS_CONFIG"]["LOOKBACK_TICKS_LIVE_RENKO"])
            if df_ticks_current_live is None or df_ticks_current_live.empty: 
                time.sleep(max(1, int(cfg_mt5_ea["TIMEFRAME_SECONDS"]/10) )); continue 
            
            if renko_builder_instance is None: renko_builder_instance = DynamicRenkoBuilder() 
            
            new_renko_bars_live = renko_builder_instance.calculate_renko_from_ticks(df_ticks_current_live, last_calculated_brick_size_global)
            if new_renko_bars_live is not None and not new_renko_bars_live.empty:
                df_renko_live_buffer = pd.concat([df_renko_live_buffer, new_renko_bars_live]).drop_duplicates(subset=['time','close', 'direction'],keep='last').tail(cfg_renko["RENKO_HISTORY_FOR_FEATURES"] + cfg_rl["STATE_HISTORY_LEN"] + 50).reset_index(drop=True) 
            
            min_bars_for_state_live = max(cfg_rl["STATE_HISTORY_LEN"] + 60, 100) 
            if df_renko_live_buffer.empty or len(df_renko_live_buffer) < min_bars_for_state_live:
                time.sleep(max(1, int(cfg_mt5_ea["TIMEFRAME_SECONDS"]/5))); continue

            if time.monotonic() - last_rl_action_time < min(5, cfg_mt5_ea["TIMEFRAME_SECONDS"] / 2) : 
                time.sleep(0.5); continue
            
            if feature_engineer_instance is None: 
                 feature_engineer_instance = FeatureEngineeringAdaptive(cfg_feat["SCALER_RENKO_FEATURES_PATH"], cfg_feat["FEATURE_COLUMNS"])
                 feature_engineer_instance.load_scaler() 

            df_renko_with_features_live = feature_engineer_instance.add_technical_indicators(df_renko_live_buffer)
            if df_renko_with_features_live.empty or len(df_renko_with_features_live) < min_bars_for_state_live :
                logger.debug("Features não puderam ser adicionadas ou resultado vazio. Aguardando mais dados Renko.")
                time.sleep(1); continue

            scaled_features_renko_live_arr = feature_engineer_instance.prepare_features_for_model(df_renko_with_features_live, for_training=False)
            
            if scaled_features_renko_live_arr is None or len(scaled_features_renko_live_arr) < cfg_rl["STATE_HISTORY_LEN"]:
                logger.debug(f"Features escaladas insuficientes ({len(scaled_features_renko_live_arr) if scaled_features_renko_live_arr is not None else 0}/{cfg_rl['STATE_HISTORY_LEN']}) para estado RL.")
                time.sleep(1); continue

            market_features_for_rl_input_live = scaled_features_renko_live_arr
            if cfg_sae["ENABLED"] and sae_handler_instance and sae_handler_instance.model and sae_handler_instance.is_scaler_pre_sae_fitted:
                encoded_features_live = sae_handler_instance.get_encoded_representation(scaled_features_renko_live_arr)
                if encoded_features_live is not None: 
                    market_features_for_rl_input_live = encoded_features_live
                else: logger.warning("Falha ao codificar features com SAE para o loop live. Usando features do FeatureEngineer.")
            
            if len(market_features_for_rl_input_live) < cfg_rl["STATE_HISTORY_LEN"]: 
                logger.debug(f"Features de mercado para RL ({len(market_features_for_rl_input_live)}) insuficientes para histórico de estado ({cfg_rl['STATE_HISTORY_LEN']}).")
                continue 

            current_market_state_sequence_live = market_features_for_rl_input_live[-cfg_rl["STATE_HISTORY_LEN"]:, :]
            
            open_positions_mt5_live_loop = comm_manager_instance.get_open_positions_mt5() 
            acc_info_live_ea_loop = comm_manager_instance.get_account_info_mt5()
            
            if acc_info_live_ea_loop is None:
                logger.warning("Não foi possível obter AccountInfo no loop live. Usando valores simulados de balanço/equity.")
                current_balance_live_ea_loop = rl_live_balance 
                current_equity_live_ea_loop = rl_live_equity
            else:
                current_balance_live_ea_loop = acc_info_live_ea_loop.balance
                current_equity_live_ea_loop = acc_info_live_ea_loop.equity
                rl_live_balance = current_balance_live_ea_loop
                rl_live_equity = current_equity_live_ea_loop

            pos_state_vec_live_loop = np.zeros(state_pos_features_dim_rl, dtype=np.float32)
            total_buy_lots_live_loop = sum(p.volume for p in open_positions_mt5_live_loop if p.type == mt5.ORDER_TYPE_BUY)
            total_sell_lots_live_loop = sum(p.volume for p in open_positions_mt5_live_loop if p.type == mt5.ORDER_TYPE_SELL)
            unrealized_pnl_buy_live_loop = sum(p.profit for p in open_positions_mt5_live_loop if p.type == mt5.ORDER_TYPE_BUY)
            unrealized_pnl_sell_live_loop = sum(p.profit for p in open_positions_mt5_live_loop if p.type == mt5.ORDER_TYPE_SELL)
            
            max_lots_cfg = cfg_rl.get("MAX_TOTAL_POSITION_LOTS", 0.1)
            pos_state_vec_live_loop[0] = total_buy_lots_live_loop / (max_lots_cfg + 1e-9)
            pos_state_vec_live_loop[1] = np.clip(unrealized_pnl_buy_live_loop / (current_balance_live_ea_loop * 0.05 + 1e-6), -3, 3) if total_buy_lots_live_loop > 0 else 0
            pos_state_vec_live_loop[2] = total_sell_lots_live_loop / (max_lots_cfg + 1e-9)
            pos_state_vec_live_loop[3] = np.clip(unrealized_pnl_sell_live_loop / (current_balance_live_ea_loop * 0.05 + 1e-6), -3, 3) if total_sell_lots_live_loop > 0 else 0
            pos_state_vec_live_loop[4] = len(open_positions_mt5_live_loop) / (cfg_rl.get("MAX_PARTIAL_POSITIONS", 3) * 2.0 + 1e-6) 
            pos_state_vec_live_loop[5] = np.clip((current_equity_live_ea_loop - current_balance_live_ea_loop) / (current_balance_live_ea_loop * 0.2 + 1e-6), -2, 2)

            rl_state_final_live_loop_raw = np.concatenate((current_market_state_sequence_live.flatten(), pos_state_vec_live_loop)).astype(np.float32)
            
            if rl_agent_instance and rl_state_final_live_loop_raw.shape[0] != rl_agent_instance.state_dim:
                logger.error(f"RL State dim mismatch LIVE! Agente espera: {rl_agent_instance.state_dim}, Estado atual: {rl_state_final_live_loop_raw.shape[0]}. Skipping RL action.")
                time.sleep(1); continue

            action_idx_rl_live_loop = 0 
            if cfg_rl["ENABLED"] and rl_agent_instance and rl_agent_instance.actor:
                action_idx_rl_live_loop = rl_agent_instance.select_action(rl_state_final_live_loop_raw, deterministic=True) 
                last_rl_action_time = time.monotonic() 
            else: 
                logger.warning("RL desabilitado ou agente não pronto. Nenhuma ação RL será tomada.")
                time.sleep(5); continue 

            action_str_rl_live_loop = cfg_rl["ACTION_MAPPING"].get(action_idx_rl_live_loop, "INVALID_ACTION_IDX")
            logger.info(f"RL Decisão Live: Idx={action_idx_rl_live_loop} -> {action_str_rl_live_loop}. Saldo: {current_balance_live_ea_loop:.2f}, Equity: {current_equity_live_ea_loop:.2f}, Posições Abertas: {len(open_positions_mt5_live_loop)}")

            s_info_live_trade_loop = comm_manager_instance.get_symbol_info_mt5() 
            if not s_info_live_trade_loop: 
                logger.error("Falha ao obter SymbolInfo antes de operar. Pulando ciclo de trade.")
                time.sleep(1); continue
            
            lot_size_action_loop = 0.0; is_buy_trade_action = False; is_sell_trade_action = False
            if action_str_rl_live_loop == "BUY_S": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][0]; is_buy_trade_action = True
            elif action_str_rl_live_loop == "BUY_M": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][1]; is_buy_trade_action = True
            elif action_str_rl_live_loop == "BUY_L": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][2]; is_buy_trade_action = True
            elif action_str_rl_live_loop == "SELL_S": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][0]; is_sell_trade_action = True
            elif action_str_rl_live_loop == "SELL_M": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][1]; is_sell_trade_action = True
            elif action_str_rl_live_loop == "SELL_L": lot_size_action_loop = cfg_rl["PARTIAL_LOT_SIZES"][2]; is_sell_trade_action = True

            can_open_new_trade_loop = True
            if lot_size_action_loop > 0:
                current_total_lots_open_mt5 = total_buy_lots_live_loop + total_sell_lots_live_loop
                if current_total_lots_open_mt5 + lot_size_action_loop > max_lots_cfg + 1e-7: 
                    logger.warning(f"RL: Ação {action_str_rl_live_loop} com lote {lot_size_action_loop} excederia MAX_TOTAL_LOTS ({max_lots_cfg}). Abertos: {current_total_lots_open_mt5}. Nenhuma ordem será aberta.")
                    can_open_new_trade_loop = False
            
            if can_open_new_trade_loop and (is_buy_trade_action or is_sell_trade_action):
                price_ask_trade_loop, price_bid_trade_loop = comm_manager_instance.get_current_prices_mt5()
                if price_ask_trade_loop and price_bid_trade_loop and price_ask_trade_loop > 0 and price_bid_trade_loop > 0:
                    entry_price_loop = price_ask_trade_loop if is_buy_trade_action else price_bid_trade_loop
                    order_type_mt5_trade_loop = mt5.ORDER_TYPE_BUY if is_buy_trade_action else mt5.ORDER_TYPE_SELL
                    
                    sl_pips_trade = cfg_rl.get("INITIAL_SL_PIP_DISTANCE_RL", 200)
                    tp_pips_trade = cfg_rl.get("INITIAL_TP_PIP_DISTANCE_RL", 400)
                    
                    sl_val_trade = entry_price_loop - sl_pips_trade * s_info_live_trade_loop.point if is_buy_trade_action else entry_price_loop + sl_pips_trade * s_info_live_trade_loop.point
                    tp_val_trade = entry_price_loop + tp_pips_trade * s_info_live_trade_loop.point if is_buy_trade_action else entry_price_loop - tp_pips_trade * s_info_live_trade_loop.point
                    
                    stops_level_dist_points = s_info_live_trade_loop.trade_stops_level 
                    min_dist_from_price = stops_level_dist_points * s_info_live_trade_loop.point

                    if is_buy_trade_action:
                        sl_val_trade = min(sl_val_trade, entry_price_loop - min_dist_from_price - s_info_live_trade_loop.point) 
                        if tp_pips_trade > 0 : tp_val_trade = max(tp_val_trade, entry_price_loop + min_dist_from_price + s_info_live_trade_loop.point)
                    else: 
                        sl_val_trade = max(sl_val_trade, entry_price_loop + min_dist_from_price + s_info_live_trade_loop.point) 
                        if tp_pips_trade > 0 : tp_val_trade = min(tp_val_trade, entry_price_loop - min_dist_from_price - s_info_live_trade_loop.point)

                    request_trade_loop = {
                        "action": mt5.TRADE_ACTION_DEAL, "symbol": cfg_mt5_ea["SYMBOL"], "volume": round(lot_size_action_loop,2),
                        "type": order_type_mt5_trade_loop, "price": entry_price_loop, 
                        "sl": round(sl_val_trade, s_info_live_trade_loop.digits), 
                        "tp": round(tp_val_trade, s_info_live_trade_loop.digits) if tp_pips_trade > 0 else 0.0,
                        "deviation": cfg_mt5_ea["DEVIATION_SLIPPAGE"], "magic": cfg_mt5_ea["MAGIC_NUMBER"],
                        "comment": f"RL_{action_str_rl_live_loop}_{int(time.time())}", "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": s_info_live_trade_loop.filling_mode 
                    }
                    result_order_live_loop = comm_manager_instance.send_order_mt5(request_trade_loop)
                    if result_order_live_loop and (result_order_live_loop.retcode == mt5.TRADE_RETCODE_DONE or result_order_live_loop.retcode == mt5.TRADE_RETCODE_PLACED) :
                        logger.info(f"RL: Ordem {action_str_rl_live_loop} enviada. Ticket: {result_order_live_loop.order if result_order_live_loop.order > 0 else result_order_live_loop.deal}. Retcode: {result_order_live_loop.retcode}")
                    else: 
                        logger.error(f"RL: Falha ao enviar ordem {action_str_rl_live_loop}. Result: {result_order_live_loop.comment if result_order_live_loop else 'None'}. Retcode: {result_order_live_loop.retcode if result_order_live_loop else 'N/A'}")
                else:
                    logger.warning("RL: Preços Ask/Bid inválidos ou não disponíveis para executar ordem.")

            elif action_str_rl_live_loop == "CLOSE_ALL":
                if open_positions_mt5_live_loop:
                    logger.info(f"RL: Ação CLOSE_ALL. Tentando fechar {len(open_positions_mt5_live_loop)} posições.")
                    closed_any_loop = False
                    for pos_live_mt5_close_loop in open_positions_mt5_live_loop: 
                        logger.info(f"RL: Tentando fechar Ticket MT5: {pos_live_mt5_close_loop.ticket} (Volume: {pos_live_mt5_close_loop.volume}, Tipo: {pos_live_mt5_close_loop.type})")
                        res_close_loop = comm_manager_instance.close_position_mt5(
                            pos_live_mt5_close_loop.ticket, 
                            pos_live_mt5_close_loop.volume, 
                            pos_live_mt5_close_loop.type 
                        )
                        if res_close_loop and (res_close_loop.retcode == mt5.TRADE_RETCODE_DONE or res_close_loop.retcode == mt5.TRADE_RETCODE_PLACED):
                            logger.info(f"RL: Posição {pos_live_mt5_close_loop.ticket} fechada (ou ordem de fechamento colocada).")
                            closed_any_loop = True
                        else: 
                            logger.error(f"RL: Falha ao fechar posição {pos_live_mt5_close_loop.ticket}. Result: {res_close_loop.comment if res_close_loop else 'None'}. Retcode: {res_close_loop.retcode if res_close_loop else 'N/A'}")
                    if closed_any_loop: logger.info("RL: Ação CLOSE_ALL processada.")
                else: logger.info("RL: Ação CLOSE_ALL, mas não há posições abertas.")

            if acc_info_live_ea_loop and CONFIG["TRADING_RISK_SETTINGS"]["GLOBAL_MAX_DRAWDOWN_STOP_PERCENT"] > 0:
                max_dd_abs_val_loop = acc_info_live_ea_loop.balance * (CONFIG["TRADING_RISK_SETTINGS"]["GLOBAL_MAX_DRAWDOWN_STOP_PERCENT"] / 100.0)
                if acc_info_live_ea_loop.equity <= acc_info_live_ea_loop.balance - max_dd_abs_val_loop:
                    logger.critical(f"RISCO GLOBAL: MAX DRAWDOWN ATINGIDO! Equity: {acc_info_live_ea_loop.equity:.2f}, Saldo Inicial DD: {acc_info_live_ea_loop.balance:.2f}, Perda Máx Abs: {max_dd_abs_val_loop:.2f}. Parando EA.")
                    break 

            if last_retrain_time_global is None: last_retrain_time_global = current_time_loop
            if not retraining_in_progress_flag and (current_time_loop - last_retrain_time_global) >= timedelta(hours=cfg_main["RETRAIN_INTERVAL_HOURS"]):
                logger.info("Intervalo de retreinamento atingido. Disparando retreinamento em background (placeholder)...")
                last_retrain_time_global = current_time_loop
            time.sleep(max(0.8, int(cfg_mt5_ea.get("TIMEFRAME_SECONDS", 60) / 10) )) 
    
    except KeyboardInterrupt:
        logger.info("EA Interrompido pelo usuário (KeyboardInterrupt).")
    except Exception as e_main_loop: 
        logger.critical(f"Erro Crítico no Loop Principal do EA: {e_main_loop}", exc_info=True)
        time.sleep(60) 
    
    finally:
        if comm_manager_instance: comm_manager_instance.desconectar()
        logger.info("=== EA SAE-RENKO-RL ENCERRADO ===")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    config_file_path_optional = "config_adaptive_ea_v5.json" 
    if os.path.exists(config_file_path_optional):
        try:
            with open(config_file_path_optional, 'r', encoding='utf-8') as f: 
                CONFIG_FROM_FILE = json.load(f)
                CONFIG = deep_update(DEFAULT_CONFIG.copy(), CONFIG_FROM_FILE) 
                logger.info(f"Configurações carregadas e mescladas de {config_file_path_optional}")
        except Exception as e_cfg: 
            logger.error(f"Erro ao carregar/mesclar config de {config_file_path_optional}: {e_cfg}. Usando defaults internos.")
            CONFIG = DEFAULT_CONFIG.copy()
    else:
        logger.info(f"Arquivo de configuração {config_file_path_optional} não encontrado. Usando defaults internos e tentando salvar.")
        CONFIG = DEFAULT_CONFIG.copy()
        try: 
            with open(config_file_path_optional, 'w', encoding='utf-8') as f_out: 
                json.dump(CONFIG, f_out, indent=4, ensure_ascii=False) 
            logger.info(f"Arquivo de configuração default salvo em: {config_file_path_optional}. Revise e preencha se necessário (LOGIN, PASSWORD, SERVER, PATH).")
        except Exception as e_save_cfg: 
            logger.error(f"Não foi possível salvar config default: {e_save_cfg}")

    mt5_login_script = CONFIG["MT5_SETTINGS"].get("LOGIN") 
    if isinstance(mt5_login_script, str) and mt5_login_script.isdigit(): 
        mt5_login_script = int(mt5_login_script)
    
    valid_mt5_config = False
    if isinstance(mt5_login_script, int) and mt5_login_script >= 0:
        if mt5_login_script == 0:
            logger.info("LOGIN MT5 é 0. EA tentará conectar-se a uma instância MT5 já logada. Certifique-se que o MT5 está aberto e logado.")
            valid_mt5_config = True
        elif mt5_login_script > 0 and CONFIG["MT5_SETTINGS"].get("PASSWORD") and CONFIG["MT5_SETTINGS"].get("SERVER"):
            logger.info(f"LOGIN MT5: {mt5_login_script}, SERVER: {CONFIG['MT5_SETTINGS'].get('SERVER')}. EA tentará logar com estas credenciais.")
            valid_mt5_config = True
        else:
            logger.critical("LOGIN MT5 > 0 especificado, mas PASSWORD ou SERVER estão vazios no CONFIG. Preencha-os no arquivo de configuração.")
    else:
        logger.critical("CONFIG LOGIN MT5 deve ser um número inteiro não negativo. Verifique o arquivo de configuração.")

    if not valid_mt5_config:
        logger.critical("Configuração MT5 inválida. Encerrando EA.")
    else:
        seed = CONFIG["GENERAL_SETTINGS"]["RANDOM_SEED"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if CONFIG["GENERAL_SETTINGS"]["DEVICE"] == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"EA {CONFIG['MT5_SETTINGS']['SYMBOL']} v5.8.1 Iniciando com DEVICE: {CONFIG['GENERAL_SETTINGS']['DEVICE']}...") 
        run_adaptive_sae_renko_rl_ea()
