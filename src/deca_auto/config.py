"""
設定管理モジュール
USER_CONFIGの定義、検証、TOMLファイルの読み書き
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import tomlkit
import traceback
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class CapacitorConfig:
    """コンデンサ設定"""
    name: str
    path: Optional[str] = None  # SPICEモデルパス
    C: Optional[float] = None  # 容量値 [F]
    ESR: float = 15e-3  # 等価直列抵抗 [Ω]
    ESL: float = 0.5e-9  # 等価直列インダクタンス [H]
    L_mnt: Optional[float] = None  # マウントインダクタンス [H]


@dataclass
class UserConfig:
    """ユーザー設定のデータクラス"""
    
    # 周波数グリッド設定
    f_start: float = 1e2  # 開始周波数 [Hz]
    f_stop: float = 5e8  # 終了周波数 [Hz]
    num_points_per_decade: int = 768  # 10倍周波数ごとの点数
    
    # 評価帯域
    f_L: float = 1e3  # 下限周波数 [Hz]
    f_H: float = 1e8  # 上限周波数 [Hz]
    
    # 目標マスク
    z_target: float = 10e-3  # フラット目標インピーダンス [Ω]
    z_custom_mask: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [
            (1e3, 10e-3),
            (5e3, 10e-3),
            (2e4, 8e-3),
            (6e5, 8e-3),
            (1e8, 1.3e0),
        ]
    )  # カスタムマスク [(freq, impedance), ...]
    
    # PDN寄生成分
    R_vrm: float = 10e-3  # VRM抵抗 [Ω]
    L_vrm: float = 10e-9  # VRMインダクタンス [H]
    R_sN: float = 0.5e-3  # spreading抵抗（デカップリングコンデンサ用） [Ω]
    L_sN: float = 0.5e-9  # spreadingインダクタンス（デカップリングコンデンサ用） [H]
    L_mntN: float = 0.5e-9  # デフォルトマウントインダクタンス [H]
    R_s: float = 0.5e-3  # spreading抵抗 [Ω]
    L_s: float = 1e-9  # spreadingインダクタンス [H]
    R_v: float = 0.5e-3  # via抵抗 [Ω]
    L_v: float = 1e-9  # viaインダクタンス [H]
    R_p: float = 10e-3  # プレーナ抵抗 [Ω]
    C_p: float = 10e-12  # プレーナ容量 [F]
    tan_delta_p: float = 0.02  # 誘電正接
    
    # SPICEシミュレーション
    dc_bias: float = 5.0  # DCバイアス電圧 [V]
    model_path: str = "model"  # SPICEモデルディレクトリ
    
    # コンデンサリスト
    capacitors: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "C_0603_0.1u", "C": 0.1e-6, "ESR": 15e-3, "ESL": 0.5e-9},
        {"name": "C_0603_1u", "C": 1e-6, "ESR": 15e-3, "ESL": 0.5e-9},
        {"name": "C_0603_4.7u", "C": 4.7e-6, "ESR": 15e-3, "ESL": 0.5e-9},
        {"name": "C_1608_10u", "C": 10e-6, "ESR": 15e-3, "ESL": 0.8e-9},
        {"name": "C_2012_22u", "C": 22e-6, "ESR": 15e-3, "ESL": 1.0e-9},
        {"name": "C_Poly_100u", "C": 100e-6, "ESR": 100e-3, "ESL": 1.5e-9},
    ])
    
    # 探索設定
    max_total_parts: int = 12  # コンデンサ総数上限
    min_total_parts_ratio: float = 0.5  # 最小総数比率
    top_k: int = 10  # 上位候補数
    shuffle_evaluation: bool = True  # 評価順のシャッフル
    buffer_limit: float = 10e6  # バッファサイズ上限
    
    # スコア重み
    weight_max: float = 0.2
    weight_area: float = 1.0
    weight_mean: float = 0.35
    weight_anti: float = 0.2
    weight_flat: float = 0.2
    weight_under: float = 0.0
    weight_parts: float = 0.1
    weight_mc_worst: float = 1.0
    
    # Monte Carlo設定
    mc_enable: bool = True
    mc_samples: int = 64
    tol_C: float = 0.2  # 容量公差
    tol_ESR: float = 0.2  # ESR公差
    tol_ESL: float = 0.2  # ESL公差
    mlcc_derating: float = 0.15  # MLCCディレーティング
    
    # システム設定
    seed: int = 1234  # 乱数シード
    max_vram_ratio_limit: float = 0.6  # VRAM使用率上限
    cuda: int = 0  # GPU番号
    dtype_c: str = "complex64"  # 複素数精度
    dtype_r: str = "float32"  # 実数精度
    force_numpy: bool = False  # NumPy強制使用
    
    # GUI設定
    use_gui: bool = True
    server_port: int = 8501
    dark_theme: bool = True
    language: str = "jp"  # "jp" or "en"
    
    # Excel出力
    excel_path: str = "out"
    excel_name: str = "dcap_result"


# デフォルト設定のグローバルインスタンス
USER_CONFIG = UserConfig()


def parse_scientific_notation(value: Any) -> float:
    """科学的記数法を解析（10e3, 1.6e-19形式に対応）"""
    if value is None:
        raise ValueError("値がNoneです")
    
    # すでに数値の場合
    if isinstance(value, (int, float)):
        return float(value)
    
    # 文字列に変換
    value_str = str(value).strip()
    if not value_str:
        raise ValueError("空の文字列です")
    
    try:
        # 通常の浮動小数点数として解析を試みる
        return float(value_str)
    except ValueError:
        # 10e3形式の場合の処理
        value_str = value_str.replace(" ", "")
        if "e" in value_str.lower():
            parts = value_str.lower().split("e")
            if len(parts) == 2:
                try:
                    base = float(parts[0]) if parts[0] else 1.0
                    exp = float(parts[1])
                    return base * (10 ** exp)
                except:
                    pass
        raise ValueError(f"無効な数値形式: {value_str}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> UserConfig:
    """
    TOMLファイルから設定を読み込む
    
    Args:
        config_path: 設定ファイルパス（Noneの場合はデフォルト設定を使用）
    
    Returns:
        UserConfig: 読み込んだ設定
    """
    config = UserConfig()
    
    if config_path is None:
        return config
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {config_path}")
        return config
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            toml_data = tomlkit.load(f)
        
        # TOMLデータで設定を上書き
        for key, value in toml_data.items():
            if hasattr(config, key):
                # 特殊処理が必要なフィールド
                if key == "capacitors":
                    # コンデンサリストの処理
                    cap_list = []
                    for cap_data in value:
                        cap_dict = {}
                        for k, v in cap_data.items():
                            if k in ["C", "ESR", "ESL", "L_mnt"] and isinstance(v, str):
                                cap_dict[k] = parse_scientific_notation(v)
                            else:
                                cap_dict[k] = v
                        cap_list.append(cap_dict)
                    setattr(config, key, cap_list)
                elif key == "z_custom_mask" and value is not None:
                    # カスタムマスクの処理
                    mask_list = []
                    for point in value:
                        if len(point) == 2:
                            f_val = parse_scientific_notation(str(point[0])) if isinstance(point[0], str) else float(point[0])
                            z_val = parse_scientific_notation(str(point[1])) if isinstance(point[1], str) else float(point[1])
                            mask_list.append((f_val, z_val))
                    setattr(config, key, mask_list if mask_list else None)
                elif isinstance(value, str) and key.startswith(("f_", "R_", "L_", "C_", "z_", "tol_", "tan_", "dc_", "mlcc_", "max_vram_", "min_total_", "weight_")):
                    # 数値パラメータの処理
                    try:
                        parsed_value = parse_scientific_notation(value)
                        setattr(config, key, parsed_value)
                    except:
                        # 解析に失敗した場合は元の値を保持
                        print(f"警告: パラメータ {key} の解析に失敗: {value}")
                        # デフォルト値を保持（setattr しない）
                else:
                    setattr(config, key, value)
        
        print(f"設定ファイルを読み込みました: {config_path}")
        
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        traceback.print_exc()
    
    return config


def save_config(config: UserConfig, config_path: Union[str, Path]) -> bool:
    """
    設定をTOMLファイルに保存
    
    Args:
        config: 保存する設定
        config_path: 保存先パス
    
    Returns:
        bool: 保存成功フラグ
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データクラスを辞書に変換
        config_dict = asdict(config)
        
        # 特殊な型の処理
        for key, value in config_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                config_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
        
        # TOMLドキュメント作成
        doc = tomlkit.document()
        
        # セクション分けして保存
        sections = {
            "frequency": ["f_start", "f_stop", "num_points_per_decade", "f_L", "f_H"],
            "target": ["z_target", "z_custom_mask"],
            "pdn_parasitic": ["R_vrm", "L_vrm", "R_sN", "L_sN", "L_mntN", "R_s", "L_s", "R_v", "L_v", "R_p", "C_p", "tan_delta_p"],
            "spice": ["dc_bias", "model_path"],
            "search": ["max_total_parts", "min_total_parts_ratio", "top_k", "shuffle_evaluation", "buffer_limit"],
            "weights": ["weight_max", "weight_area", "weight_mean", "weight_anti", "weight_flat", "weight_under", "weight_parts", "weight_mc_worst"],
            "monte_carlo": ["mc_enable", "mc_samples", "tol_C", "tol_ESR", "tol_ESL", "mlcc_derating"],
            "system": ["seed", "max_vram_ratio_limit", "cuda", "dtype_c", "dtype_r", "force_numpy"],
            "gui": ["use_gui", "server_port", "dark_theme", "language"],
            "output": ["excel_path", "excel_name"]
        }
        
        # セクションごとに追加
        for section_name, keys in sections.items():
            section = tomlkit.table()
            for key in keys:
                if key in config_dict:
                    section[key] = config_dict[key]
            if section:
                doc[section_name] = section
        
        # コンデンサリストを追加
        doc["capacitors"] = config_dict["capacitors"]
        
        # ファイルに書き込み
        with open(config_path, "w", encoding="utf-8") as f:
            tomlkit.dump(doc, f)
        
        print(f"設定を保存しました: {config_path}")
        return True
        
    except Exception as e:
        print(f"設定ファイルの保存エラー: {e}")
        traceback.print_exc()
        return False


def validate_config(config: UserConfig) -> bool:
    """
    設定の妥当性を検証
    
    Args:
        config: 検証する設定
    
    Returns:
        bool: 検証成功フラグ
    """
    try:
        # 周波数範囲の検証
        assert config.f_start > 0, "開始周波数は正の値である必要があります"
        assert config.f_stop > config.f_start, "終了周波数は開始周波数より大きい必要があります"
        assert config.num_points_per_decade > 0, "周波数点数は正の値である必要があります"
        
        # 評価帯域の検証
        assert config.f_L >= config.f_start, "評価帯域下限は開始周波数以上である必要があります"
        assert config.f_H <= config.f_stop, "評価帯域上限は終了周波数以下である必要があります"
        assert config.f_L < config.f_H, "評価帯域が正しく設定されていません"
        
        # カスタムマスクの検証
        if config.z_custom_mask:
            for i, (f, z) in enumerate(config.z_custom_mask):
                assert f > 0, f"カスタムマスク[{i}]の周波数は正の値である必要があります"
                assert z > 0, f"カスタムマスク[{i}]のインピーダンスは正の値である必要があります"
            
            # 周波数の昇順チェック
            freqs = [f for f, _ in config.z_custom_mask]
            assert freqs == sorted(freqs), "カスタムマスクの周波数は昇順である必要があります"
        
        # PDN寄生成分の検証
        assert all(getattr(config, f) >= 0 for f in ["R_vrm", "L_vrm", "R_sN", "L_sN", "L_mntN", "R_s", "L_s", "R_v", "L_v", "R_p", "C_p", "tan_delta_p"]), \
            "PDN寄生成分は非負の値である必要があります"
        
        # コンデンサリストの検証
        assert len(config.capacitors) > 0, "コンデンサリストが空です"
        for i, cap in enumerate(config.capacitors):
            assert "name" in cap, f"コンデンサ[{i}]に名前がありません"
            if "C" in cap:
                assert cap["C"] > 0, f"コンデンサ[{i}]の容量は正の値である必要があります"
        
        # 探索設定の検証
        assert config.max_total_parts > 0, "最大総数は正の値である必要があります"
        assert 0 < config.min_total_parts_ratio <= 1, "最小総数比率は0-1の範囲である必要があります"
        assert config.top_k > 0, "上位候補数は正の値である必要があります"
        assert config.buffer_limit > 0, "バッファサイズは正の値である必要があります"
        
        # Monte Carlo設定の検証
        if config.mc_enable:
            assert config.mc_samples > 0, "MCサンプル数は正の値である必要があります"
            assert 0 <= config.tol_C <= 1, "容量公差は0-1の範囲である必要があります"
            assert 0 <= config.tol_ESR <= 1, "ESR公差は0-1の範囲である必要があります"
            assert 0 <= config.tol_ESL <= 1, "ESL公差は0-1の範囲である必要があります"
            assert 0 <= config.mlcc_derating <= 1, "MLCCディレーティングは0-1の範囲である必要があります"
        
        # システム設定の検証
        assert 0 < config.max_vram_ratio_limit <= 1, "VRAM使用率上限は0-1の範囲である必要があります"
        assert config.cuda >= 0, "GPU番号は非負の整数である必要があります"
        assert config.dtype_c in ["complex64", "complex128"], "複素数精度が無効です"
        assert config.dtype_r in ["float32", "float64"], "実数精度が無効です"
        
        # GUI設定の検証
        assert config.server_port > 0, "ポート番号は正の値である必要があります"
        assert config.language in ["jp", "en"], "言語設定が無効です"
        
        return True
        
    except AssertionError as e:
        print(f"設定検証エラー: {e}")
        return False
    except Exception as e:
        print(f"設定検証で予期しないエラー: {e}")
        traceback.print_exc()
        return False


def get_localized_text(key: str, config: UserConfig) -> str:
    """
    言語設定に基づいてローカライズされたテキストを取得
    
    Args:
        key: テキストキー
        config: 設定
    
    Returns:
        str: ローカライズされたテキスト
    """
    texts = {
        "jp": {
            "title": "Deca Auto【PDN自動最適化ツール】",
            "start_search": "探索開始",
            "stop_search": "停止",
            "settings": "設定",
            "results": "結果",
            "capacitor_list": "コンデンサリスト",
            "update_caplist":"コンデンサリストを更新",
            "apply_change":"変更を適用",
            "target_mask": "目標マスク",
            "update_mask":"目標マスクを更新",
            "frequency_grid": "周波数グリッド",
            "evaluation_band": "評価帯域",
            "search_settings": "探索設定",
            "monte_carlo": "Monte Carlo設定",
            "gpu_settings": "GPU設定",
            "weights": "評価重み",
            "calculate_zc_only": "Z_cのみ計算",
            "use_custom_mask": "カスタムマスクを使用",
            "load_file":"📁 ファイル",
            "save": "保存",
            "save_as": "名前を付けて保存",
            "load_config": "設定ファイルを読み込む",
            "drop_config": "設定ファイルをここにドロップ",
            "system":"システム",
            "language": "言語",
            "theme": "テーマ",
        },
        "en": {
            "title": "Deca Auto【PDN Impedance Optimization Tool】",
            "start_search": "Start Search",
            "stop_search": "Stop",
            "settings": "Settings",
            "results": "Results",
            "capacitor_list": "Capacitor List",
            "update_caplist":"Updated the capacitor list",
            "apply_change":"Apply Changes",
            "target_mask": "Target Mask",
            "update_mask":"Updated the custom mask",
            "frequency_grid": "Frequency Grid",
            "evaluation_band": "Evaluation Band",
            "search_settings": "Search Settings",
            "monte_carlo": "Monte Carlo Settings",
            "gpu_settings": "GPU Settings",
            "weights": "Evaluation Weights",
            "calculate_zc_only": "Calculate Z_c Only",
            "use_custom_mask": "Use Custom Mask",
            "load_file":"📁 File Utility",
            "save": "Save",
            "save_as": "Save As",
            "load_config": "Load Config File",
            "drop_config": "Drop config file here",
            "system":"system",
            "language": "Language",
            "theme": "theme",
        }
    }
    
    lang = config.language if config.language in texts else "jp"
    return texts[lang].get(key, key)