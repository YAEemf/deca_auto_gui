from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import tomlkit
import traceback
from dataclasses import dataclass, field, asdict
import numpy as np

from deca_auto.utils import unwrap_toml_value, to_float, to_int, parse_scientific_notation, logger


def _derive_name_from_path(path_value: Any) -> Optional[str]:
    """パスからファイル名（拡張子なし）を抽出"""
    if path_value is None:
        return None
    try:
        path_str = str(path_value).strip()
    except Exception:
        return None
    if not path_str:
        return None
    try:
        stem = Path(path_str).stem
    except Exception:
        return None
    return stem or None


def _ensure_unique_name(name_candidate: str, existing: set[str]) -> str:
    """既存集合と重複しない一意な名前を生成"""
    base = name_candidate
    suffix = 1
    while name_candidate in existing:
        name_candidate = f"{base}_{suffix}"
        suffix += 1
    existing.add(name_candidate)
    return name_candidate


@dataclass
class CapacitorConfig:
    """コンデンサ設定"""
    name: str
    path: Optional[str] = ""    # SPICEモデルパス
    C: Optional[float] = 0.0      # 容量値 [F]
    ESR: float = 15e-3          # 等価直列抵抗 [Ω]
    ESL: float = 4e-10         # 等価直列インダクタンス [H]
    L_mnt: Optional[float] = 0.5e-9  # マウントインダクタンス [H]
    MIN: Optional[int] = None   # 最小使用数
    MAX: Optional[int] = None   # 最大使用数


@dataclass
class UserConfig:
    """ユーザー設定のデータクラス"""
    
    # 周波数グリッド設定
    f_start: float = 1e2  # 開始周波数 [Hz]
    f_stop: float = 1e9  # 終了周波数 [Hz]
    num_points_per_decade: int = 128  # DECADEごとの点数
    
    # 評価帯域
    f_L: float = 1e3  # 下限周波数 [Hz]
    f_H: float = 1e8  # 上限周波数 [Hz]
    
    # 目標マスク
    target_impedance_mode: str = "auto"  # モード: "flat", "auto", "custom"
    z_target: float = 10e-3  # 目標インピーダンス(フラット) [Ω]　 ΔV/ΔI=10e-3Ω
    z_custom_mask: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [
            (1e3, 8e-3),
            (1e6, 8e-3),
            (1e7, 2e-2),
            (1e8, 0.20),
        ]
    )  # カスタムマスク [(freq, impedance), ...]

    # 自動計算モード用パラメーター(FPGAのCore電圧を想定)
    v_supply: float = 1.2  # 電源電圧 [V]
    ripple_ratio: Optional[float] = 5.0  # 許容リップル率 [%]
    ripple_voltage: Optional[float] = None  # 許容リップル電圧 [V] (ripple_ratioと排他)
    i_max: float = 10.0  # 最大供給電流 [A]
    switching_activity: Optional[float] = 0.5  # 電流変動率 (0-1)
    i_transient: Optional[float] = None  # 過渡電流 [A] (switching_activityと排他)
    design_margin: float = 15.0  # デザインマージン [%]
    
    # PDN寄生成分
    R_vrm: float = 15e-3    # VRM ESR [Ω]
    L_vrm: float = 5e-9     # VRM ESL [H]
    R_sN: float = 0.5e-3    # spreading抵抗（デカップリングコンデンサ用）[Ω]
    L_sN: float = 0.4e-9    # spreadingインダクタンス（デカップリングコンデンサ用）[H]
    L_mntN: float = 0.2e-10 # マウントインダクタンス [H]
    R_s: float = 0.2e-3     # spreading抵抗（VCC直前）[Ω]
    L_s: float = 0.25e-9    # spreadingインダクタンス（VCC直前）[H]
    R_v: float = 1e-3       # via抵抗 [Ω]
    L_v: float = 0.5e-9     # viaインダクタンス [H]
    R_p: float = 20e-3      # プレーナ抵抗 [Ω]
    C_p: float = 2.5e-11     # プレーナ容量 [F]
    tan_delta_p: float = 0.02  # 誘電正接
    
    # SPICEシミュレーション
    dc_bias: float = v_supply  # DCバイアス電圧 [V]
    model_path: str = "model"  # SPICEモデルディレクトリ
    
    # コンデンサリスト
    capacitors: List[Dict[str, Any]] = field(default_factory=lambda: [
        # {"name": "C_1005_10n", "C": 10e-9, "ESR": 20e-3, "ESL": 2e-10},
        # {"name": "C_1005_22n", "C": 22e-9, "ESR": 20e-3, "ESL": 2e-10},
        # {"name": "C_1005_33n", "C": 33e-9, "ESR": 20e-3, "ESL": 2e-10},
        {"name": "C_1005_47n", "C": 47e-9, "ESR": 20e-3, "ESL": 2e-10},
        {"name": "C_1005_68n", "C": 68e-9, "ESR": 20e-3, "ESL": 2e-10},
        {"name": "C_1608_0.1u", "C": 0.1e-6, "ESR": 15e-3, "ESL": 3e-10},
        {"name": "C_1608_0.22u", "C": 0.22e-6, "ESR": 15e-3, "ESL": 3e-10},
        {"name": "C_1608_0.33u", "C": 0.33e-6, "ESR": 15e-3, "ESL": 3e-10},
        {"name": "C_1608_0.47u", "C": 0.47e-6, "ESR": 15e-3, "ESL": 3e-10},
        {"name": "C_1608_0.68u", "C": 0.68e-6, "ESR": 15e-3, "ESL": 3e-10},
        {"name": "C_1608_1u", "C": 1e-6, "ESR": 12e-3, "ESL": 3e-10},
        {"name": "C_1608_2.2u", "C": 2.2e-6, "ESR": 12e-3, "ESL": 3e-10},
        {"name": "C_1608_3.3u", "C": 3.3e-6, "ESR": 12e-3, "ESL": 3e-10},
        {"name": "C_1608_4.7u", "C": 4.7e-6, "ESR": 12e-3, "ESL": 3e-10},
        {"name": "C_1608_6.8u", "C": 6.8e-6, "ESR": 12e-3, "ESL": 3e-10},
        {"name": "C_2012_10u", "C": 10e-6, "ESR": 10e-3, "ESL": 4e-10},
        {"name": "C_3216_22u", "C": 22e-6, "ESR": 10e-3, "ESL": 6e-10},
        {"name": "C_3216_33u", "C": 33e-6, "ESR": 10e-3, "ESL": 6e-10},
        {"name": "C_3216_47u", "C": 47e-6, "ESR": 10e-3, "ESL": 6e-10},
        {"name": "C_Poly_100u", "C": 100e-6, "ESR": 100e-3, "ESL": 1.5e-9},
    ])
    
    # 探索設定
    max_total_parts: int = 8  # コンデンサ総数上限
    min_total_parts_ratio: float = 0.6  # 最小総数比率
    top_k: int = 15  # 上位候補数
    shuffle_evaluation: bool = True  # 評価順のシャッフル
    buffer_limit: float = 100e6  # バッファサイズ上限
    
    # スコア重み
    weight_max: float = 0.6
    weight_area: float = 2.0
    weight_mean: float = 0.3
    weight_anti: float = 0.1
    weight_flat: float = 0.1
    weight_under: float = -0.1
    weight_parts: float = 0.1
    weight_num_types: float = 2.0
    weight_resonance: float = 0.1
    weight_improvement: float = 1.0
    weight_low_improvement: float = 2.0
    low_improvement_threshold: float = 0.15
    weight_mc_worst: float = 1.0
    ignore_safe_anti_resonance: bool = False
    
    # Monte Carlo設定
    mc_enable: bool = True
    mc_samples: int = 64
    tol_C: float = 0.15  # 容量公差
    tol_ESR: float = 0.15  # ESR公差
    tol_ESL: float = 0.15  # ESL公差
    mlcc_derating: float = 0.15  # MLCCディレーティング
    
    # システム設定
    seed: int = 1234  # 乱数シード
    max_vram_ratio_limit: float = 0.5  # VRAM使用率上限
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


def _load_config_field(config: UserConfig, key: str, value: Any) -> None:
    """
    設定フィールドを読み込む（特殊処理を含む）

    Args:
        config: 設定オブジェクト
        key: フィールド名
        value: 値
    """
    # 特殊処理が必要なフィールド
    if key == "capacitors" and value is not None:
        cap_list: List[Dict[str, Any]] = []
        existing_names: set[str] = set()

        for cap_data in value:
            if not isinstance(cap_data, dict):
                continue

            cap_dict: Dict[str, Any] = {}
            for k, v in cap_data.items():
                v_unwrapped = unwrap_toml_value(v)

                if k in ["C", "ESR", "ESL", "L_mnt"]:
                    if isinstance(v_unwrapped, str):
                        cap_dict[k] = parse_scientific_notation(v_unwrapped)
                    elif v_unwrapped is not None:
                        cap_dict[k] = to_float(v_unwrapped, 0.0)

                elif k in ["MIN", "MAX"]:
                    if v_unwrapped is not None:
                        cap_dict[k] = to_int(v_unwrapped, None)

                elif k == "path":
                    if v_unwrapped is not None:
                        cap_dict[k] = str(v_unwrapped).strip()

                elif k == "name":
                    if v_unwrapped is not None:
                        cap_dict[k] = str(v_unwrapped).strip()

                else:
                    if v_unwrapped is not None:
                        cap_dict[k] = v_unwrapped

            path_val = cap_dict.get("path", "")
            raw_name = cap_dict.get("name", "")
            if isinstance(raw_name, str):
                raw_name = raw_name.strip()
            else:
                raw_name = str(raw_name).strip() if raw_name is not None else ""

            if not raw_name:
                inferred = _derive_name_from_path(path_val)
                raw_name = inferred.strip() if inferred else ""

            if not raw_name:
                # 名前を特定できなければスキップ
                continue

            final_name = _ensure_unique_name(raw_name, existing_names)
            cap_dict["name"] = final_name
            cap_list.append(cap_dict)

        setattr(config, key, cap_list)

    elif key == "z_custom_mask" and value is not None:
        mask_list = []
        for point in value:
            if len(point) == 2:
                f_raw, z_raw = point
                f_val = parse_scientific_notation(f_raw) if isinstance(f_raw, str) else to_float(f_raw, None)
                z_val = parse_scientific_notation(z_raw) if isinstance(z_raw, str) else to_float(z_raw, None)
                if f_val is not None and z_val is not None:
                    mask_list.append((f_val, z_val))
        setattr(config, key, mask_list if mask_list else None)

    elif isinstance(value, str) and key.startswith(("f_", "R_", "L_", "C_", "z_", "tol_", "tan_", "dc_", "mlcc_", "max_vram_", "min_total_", "weight_", "v_", "i_", "ripple_", "switching_", "design_")):
        try:
            parsed_value = parse_scientific_notation(value)
            setattr(config, key, parsed_value)
        except Exception:
            print(f"警告: パラメータ {key} の解析に失敗: {value}")
    else:
        setattr(config, key, value)


def load_config(config_path: Optional[Union[str, Path]] = None, verbose: bool = True) -> UserConfig:
    """
    TOMLファイルから設定を読み込む

    Args:
        config_path: 設定ファイルパス（Noneの場合はデフォルト設定を使用）
        verbose: デバッグログの出力フラグ

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

        # デバッグ: 読み込んだTOMLデータの構造を確認
        if verbose:
            print(f"=== TOMLデータ読み込み開始: {config_path} ===")
            print(f"トップレベルキー: {list(toml_data.keys())}")

        # TOMLデータで設定を上書き（セクション構造に対応）
        for section_or_key, raw_value in toml_data.items():
            value = unwrap_toml_value(raw_value)

            # セクション（辞書）の場合は中身を展開
            if isinstance(value, dict):
                if verbose:
                    print(f"[{section_or_key}] セクションを処理中...")
                for key, sub_raw_value in value.items():
                    sub_value = unwrap_toml_value(sub_raw_value)
                    if hasattr(config, key):
                        _load_config_field(config, key, sub_value)
                        if verbose:
                            print(f"  {key} = {sub_value}")
                    else:
                        if verbose:
                            print(f"  警告: 未知のパラメータ '{key}' をスキップ")
            # トップレベルのキーの場合（セクションではない）
            elif hasattr(config, section_or_key):
                _load_config_field(config, section_or_key, value)
                if verbose:
                    print(f"{section_or_key} = {value}")
            else:
                if verbose:
                    print(f"警告: 未知のセクションまたはパラメータ '{section_or_key}' をスキップ")

        if verbose:
            print(f"=== 設定ファイルを読み込みました: {config_path} ===")
        
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

        # capacitors の None/空文字列をフィルタリング
        if "capacitors" in config_dict and config_dict["capacitors"] is not None:
            filtered_capacitors = []
            for cap in config_dict["capacitors"]:
                if not isinstance(cap, dict):
                    continue
                filtered_cap = {}
                for k, v in cap.items():
                    # None をスキップ
                    if v is None:
                        continue
                    # 空文字列をスキップ（name は除外）
                    if k != "name" and isinstance(v, str) and v.strip() == "":
                        continue
                    filtered_cap[k] = v
                # name が必須
                if "name" in filtered_cap and filtered_cap["name"]:
                    filtered_capacitors.append(filtered_cap)
            config_dict["capacitors"] = filtered_capacitors

        def _sanitize(value: Any) -> Any:
            """TOML変換前にNoneを除去"""
            if value is None:
                return None
            if isinstance(value, dict):
                cleaned_dict = {}
                for k, v in value.items():
                    sanitized = _sanitize(v)
                    if sanitized is not None:
                        cleaned_dict[k] = sanitized
                return cleaned_dict
            if isinstance(value, list):
                cleaned_list = []
                for item in value:
                    sanitized = _sanitize(item)
                    if sanitized is not None:
                        cleaned_list.append(sanitized)
                return cleaned_list
            if isinstance(value, tuple):
                cleaned_tuple = tuple(
                    sanitized for sanitized in (_sanitize(item) for item in value)
                    if sanitized is not None
                )
                return cleaned_tuple if cleaned_tuple else None
            return value

        # Noneを除去したクリーンな辞書を生成
        sanitized_config = _sanitize(config_dict)
        if not isinstance(sanitized_config, dict):
            sanitized_config = {}
        config_dict = {key: value for key, value in sanitized_config.items() if value is not None}

        # TOMLドキュメント作成
        doc = tomlkit.document()

        # セクション分けして保存
        sections = {
            "frequency": ["f_start", "f_stop", "num_points_per_decade", "f_L", "f_H"],
            "target": [
                "target_impedance_mode",
                "z_target",
                "z_custom_mask",
                "v_supply",
                "ripple_ratio",
                "ripple_voltage",
                "i_max",
                "switching_activity",
                "i_transient",
                "design_margin"
            ],
            "pdn_parasitic": ["R_vrm", "L_vrm", "R_sN", "L_sN", "L_mntN", "R_s", "L_s", "R_v", "L_v", "R_p", "C_p", "tan_delta_p"],
            "spice": ["dc_bias", "model_path"],
            "search": ["max_total_parts", "min_total_parts_ratio", "top_k", "shuffle_evaluation", "buffer_limit"],
            "weights": [
                "weight_max",
                "weight_area",
                "weight_mean",
                "weight_anti",
                "weight_flat",
                "weight_under",
                "weight_parts",
                "weight_num_types",
                "weight_resonance",
                "weight_improvement",
                "weight_low_improvement",
                "low_improvement_threshold",
                "weight_mc_worst",
                "ignore_safe_anti_resonance"
            ],
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
                    value = config_dict[key]
                    if value is None:
                        continue
                    section[key] = value
            if section:
                doc[section_name] = section

        # コンデンサリストを追加
        if "capacitors" in config_dict and config_dict["capacitors"]:
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
        
        # 目標インピーダンスモードの検証
        assert config.target_impedance_mode in ["flat", "auto", "custom"], \
            "目標インピーダンスモードは 'flat', 'auto', 'custom' のいずれかである必要があります"

        # 自動計算モードパラメーターの検証
        if config.target_impedance_mode == "auto":
            assert config.v_supply > 0, "電源電圧は正の値である必要があります"
            assert config.i_max > 0, "最大消費電流は正の値である必要があります"

            # ripple_ratio と ripple_voltage は排他的
            if config.ripple_ratio is not None and config.ripple_voltage is not None:
                print("警告: ripple_ratio と ripple_voltage が両方設定されています。ripple_ratio を優先します")

            if config.ripple_ratio is not None:
                assert 0 < config.ripple_ratio <= 100, "許容リップル率は0-100%の範囲である必要があります"
            elif config.ripple_voltage is not None:
                assert config.ripple_voltage > 0, "許容リップル電圧は正の値である必要があります"
            else:
                raise AssertionError("ripple_ratio または ripple_voltage のいずれかを設定する必要があります")

            # switching_activity と i_transient は排他的
            if config.switching_activity is not None and config.i_transient is not None:
                print("警告: switching_activity と i_transient が両方設定されています。i_transient を優先します")

            if config.switching_activity is not None:
                assert 0 < config.switching_activity <= 1, "電流変動率は0-1の範囲である必要があります"

            if config.i_transient is not None:
                assert config.i_transient > 0, "過渡電流は正の値である必要があります"

            assert 0 <= config.design_margin <= 100, "デザインマージンは0-100%の範囲である必要があります"

        # カスタムマスクの検証
        if config.target_impedance_mode == "custom" or config.z_custom_mask:
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
            if "path" not in cap:
                if "C" in cap:
                    assert cap["C"] > 0, f"コンデンサ[{i}]の容量は正の値である必要があります"

            min_val = cap.get("MIN") if isinstance(cap, dict) else None
            max_val = cap.get("MAX") if isinstance(cap, dict) else None

            if min_val is not None:
                assert float(min_val) >= 0, f"コンデンサ[{i}]の最小使用数は0以上である必要があります"
            if max_val is not None:
                assert float(max_val) >= 0, f"コンデンサ[{i}]の最大使用数は0以上である必要があります"
            if min_val is not None and max_val is not None:
                assert float(max_val) >= float(min_val), f"コンデンサ[{i}]の最大使用数は最小使用数以上である必要があります"

        # 探索設定の検証
        assert config.max_total_parts > 0, "最大総数は正の値である必要があります"
        assert config.min_total_parts_ratio <= 1, "最小総数比率は0-1の範囲である必要があります"
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
        assert config.dtype_c in ["complex64", "complex128", "complex256"], "複素数精度が無効です"
        assert config.dtype_r in ["float16", "float32", "float64", "float128"], "実数精度が無効です"
        
        # GUI設定の検証
        assert config.server_port > 0, "ポート番号は正の値である必要があります"
        assert config.language in ["jp", "en"], "言語設定が無効です"
        
        return True
        
    except AssertionError as e:
        print(f"設定検証エラー: {e}")
        return False


def resolve_capacitor_usage_bounds(config: UserConfig,
                                   ordered_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    ソート済みコンデンサ名に対するMIN/MAX制約を配列で返す

    Args:
        config: ユーザー設定
        ordered_names: 容量小→大で整列済みのコンデンサ名

    Returns:
        (min_counts, max_counts): 各コンデンサの最小/最大数
    """
    max_total_parts = max(0, int(config.max_total_parts))
    min_counts: List[int] = []
    max_counts: List[int] = []

    for name in ordered_names:
        cap_cfg = next((c for c in config.capacitors if c.get('name') == name), {})
        raw_min = cap_cfg.get('MIN') if cap_cfg else None
        raw_max = cap_cfg.get('MAX') if cap_cfg else None

        min_count = to_int(raw_min, 0) if raw_min is not None else 0
        max_default = max_total_parts if max_total_parts > 0 else 0
        max_count = to_int(raw_max, max_default) if raw_max is not None else max_default

        if min_count is None:
            min_count = 0
        if max_count is None:
            max_count = max_default

        if min_count < 0:
            logger.info(f"コンデンサ {name} の最小使用数 {min_count} を0に切り上げます")
            min_count = 0

        if max_count < 0:
            logger.info(f"コンデンサ {name} の最大使用数 {max_count} を max_total_parts ({max_default}) に合わせます")
            max_count = max_default

        if max_total_parts > 0 and max_count > max_total_parts:
            logger.info(f"コンデンサ {name} の最大使用数 {max_count} を max_total_parts ({max_total_parts}) に合わせます")
            max_count = max_total_parts

        if max_count < min_count:
            logger.info(f"コンデンサ {name} の最大使用数 {max_count} が最小使用数 {min_count} より小さいため調整します")
            max_count = min_count

        min_counts.append(int(min_count))
        max_counts.append(int(max_count))

    return np.asarray(min_counts, dtype=np.int32), np.asarray(max_counts, dtype=np.int32)


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
            "title": "Deca Auto",
            "start_search": "探索開始",
            "stop_search": "停止",
            "settings": "設定",
            "results": "結果",
            "capacitor_list": "コンデンサリスト",
            "update_caplist":"コンデンサリストを更新",
            "apply_change":"変更を適用",
            "target_mask": "目標インピーダンス",
            "update_mask":"目標インピーダンスを更新",
            "frequency_grid": "周波数グリッド",
            "evaluation_band": "評価帯域",
            "search_settings": "探索設定",
            "monte_carlo": "Monte Carlo設定",
            "gpu_settings": "GPU設定",
            "weights": "評価重み",
            "weight_num_types": "種類数の重み",
            "weight_resonance": "共振ペナルティの重み",
            "ignore_safe_anti": "目標以下の反共振を無視",
            "reset_weights": "重みをリセット",
            "calculate_zc_only": "|Z_c|計算",
            "use_custom_mask": "カスタムマスクを使用",
            "load_file":"📁 ファイル",
            "save": "保存",
            "save_as": "名前を付けて保存",
            "load_config": "設定ファイルを読み込む",
            "drop_config": "設定ファイルをここにドロップ",
            "system":"システム",
            "language": "言語",
            "theme": "テーマ",
            "stray_parameters": "寄生成分",
            "usage_range": "使用数範囲",
            "show_column": "",
            "show_column_help": "グラフの表示を切り替え",
            "label_R_vrm": "R_vrm [Ω]",
            "label_L_vrm": "L_vrm [H]",
            "label_R_sN": "R_sN [Ω]",
            "label_L_sN": "L_sN [H]",
            "label_L_mntN": "L_mntN [H]",
            "label_R_s": "R_s [Ω]",
            "label_L_s": "L_s [H]",
            "label_R_v": "R_v [Ω]",
            "label_L_v": "L_v [H]",
            "label_R_p": "R_p [Ω]",
            "label_C_p": "C_p [F]",
            "label_tan_delta_p": "tanδ"
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
            "target_mask": "Target Impedance",
            "update_mask":"Updated the custom mask",
            "frequency_grid": "Frequency Grid",
            "evaluation_band": "Evaluation Band",
            "search_settings": "Search Settings",
            "monte_carlo": "Monte Carlo Settings",
            "gpu_settings": "GPU Settings",
            "weights": "Evaluation Weights",
            "weight_num_types": "Num types weight",
            "weight_resonance": "Resonance penalty weight",
            "ignore_safe_anti": "Ignore anti-resonances under target",
            "reset_weights": "Reset Weights",
            "calculate_zc_only": "Calculate |Z_c| Only",
            "use_custom_mask": "Use Custom Mask",
            "load_file":"📁 File Utility",
            "save": "Save",
            "save_as": "Save As",
            "load_config": "Load Config File",
            "drop_config": "Drop config file here",
            "system":"system",
            "language": "Language",
            "theme": "theme",
            "stray_parameters": "Stray Parameters",
            "usage_range": "Usage Range",
            "show_column": "",
            "show_column_help": "Toggle visibility in the chart",
            "label_R_vrm": "R_vrm [Ω]",
            "label_L_vrm": "L_vrm [H]",
            "label_R_sN": "R_sN [Ω]",
            "label_L_sN": "L_sN [H]",
            "label_L_mntN": "L_mntN [H]",
            "label_R_s": "R_s [Ω]",
            "label_L_s": "L_s [H]",
            "label_R_v": "R_v [Ω]",
            "label_L_v": "L_v [H]",
            "label_R_p": "R_p [Ω]",
            "label_C_p": "C_p [F]",
            "label_tan_delta_p": "tanδ"
        }
    }
    
    lang = config.language if config.language in texts else "jp"
    return texts[lang].get(key, key)
