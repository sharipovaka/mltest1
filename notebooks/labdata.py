# -*- coding: utf-8 -*-
"""
Эталонная предобработка индивидуальной выборки (результат лабораторной 1).

Работы 2–9 начинаются с готовых матриц «объекты–признаки», поэтому ошибки,
допущенные в первой работе, не тянутся через весь семестр. Модуль повторяет
ровно те шаги, которые студент выполняет вручную в лабораторной 1:

    приведение типов -> нормализация категорий -> удаление дубликатов и
    вырожденных признаков -> исправление опечаток масштаба -> удаление `id`
    -> разбиение train/test -> заполнение пропусков и кодирование
    (параметры оцениваются только по обучающей части).

Использование::

    from variants import get_variant
    from labdata import load_personal

    data = load_personal(get_variant("Иванов Иван", lab=2))
    data["X_train"].shape, data["task"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from variants import make_table

__all__ = ["clean_table", "load_personal"]


def _to_numeric_safe(s: pd.Series, threshold: float = 0.2) -> pd.Series:
    """Строковый столбец -> числовой, если он на самом деле числовой."""
    if s.dtype != object:
        return s
    cleaned = (s.astype(str)
                 .str.replace(" ", "", regex=False)
                 .str.replace(" ", "", regex=False)
                 .str.replace(",", ".", regex=False)
                 .str.strip())
    converted = pd.to_numeric(cleaned, errors="coerce")
    lost = (converted.isna() & s.notna()).mean()
    return converted if lost <= threshold else s


def clean_table(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Очистка сырой таблицы (шаги частей 3–6 лабораторной 1)."""
    df = df.copy()
    target = meta["target"]

    for col in df.columns:
        df[col] = _to_numeric_safe(df[col])

    for col in meta["categorical"]:
        df[col] = df[col].astype(object).map(
            lambda v: v.strip().lower() if isinstance(v, str) else v)

    df = df.drop_duplicates().reset_index(drop=True)

    drop = [c for c in df.columns.drop(target)
            if df[c].nunique(dropna=False) <= 1
            or df[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.99]
    df = df.drop(columns=drop)

    # опечатки масштаба: значение больше 30 * q_{0.99} -- потерянный разделитель
    for col in df.select_dtypes(include="number").columns.drop(target):
        mask = df[col] > 30 * df[col].quantile(0.99)
        df.loc[mask, col] = df.loc[mask, col] / 100.0

    return df.drop(columns=[c for c in ("id",) if c in df.columns])


def load_personal(variant: dict, test_size: float = 0.25, random_state: int = 42,
                  return_frame: bool = False) -> dict:
    """Личная выборка студента, приведённая к матрицам «объекты–признаки».

    Returns
    -------
    dict с ключами ``X_train``, ``X_test``, ``y_train``, ``y_test``,
    ``feature_names``, ``preprocessor``, ``task``, ``target``, ``domain``
    (и ``frame`` при ``return_frame=True``).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    raw, meta = make_table(variant)
    df = clean_table(raw, meta)
    target = meta["target"]

    X, y = df.drop(columns=[target]), df[target]
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    stratify = y if meta["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), cat_cols),
    ])

    out = {
        "X_train": preprocessor.fit_transform(X_train),
        "X_test": preprocessor.transform(X_test),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
        "feature_names": list(preprocessor.get_feature_names_out()),
        "preprocessor": preprocessor,
        "task": meta["task"],
        "target": target,
        "domain": meta["domain"],
    }
    if return_frame:
        out["frame"] = df
    return out


if __name__ == "__main__":
    from variants import get_variant

    d = load_personal(get_variant("Образцов Пётр Никитич", lab=2))
    print(d["domain"], d["task"], d["X_train"].shape, d["X_test"].shape)
    print(d["feature_names"][:5])
