# -*- coding: utf-8 -*-
"""
Генератор индивидуальных вариантов для лабораторного практикума
по курсу «Машинное обучение» (каф. ФН1, 4 курс).

Идея (по образцу курсового задания МФТИ): студент вводит свою почту или ФИО,
из строки детерминированно вычисляется seed, а по seed выдаются

  * индивидуальный «грязный» табличный датасет (лаб. 1, сквозной для курса);
  * индивидуальный набор методов/параметров, которые нужно проанализировать.

Два студента с разными ФИО почти наверняка получат разные варианты, а один и
тот же студент при повторном запуске -- ровно тот же вариант (воспроизводимость).

Использование::

    from variants import get_variant, make_table

    v = get_variant("Иванов Иван Иванович", lab=1)
    print(v)
    df = make_table(v)

Файл не требует ничего, кроме numpy и pandas.
"""

from __future__ import annotations

import hashlib
import textwrap

import numpy as np
import pandas as pd

__all__ = [
    "student_seed",
    "get_variant",
    "make_table",
    "describe_variant",
    "DOMAINS",
]

# Соль курса. Меняется раз в год -- тогда варианты прошлого года становятся
# бесполезными, и отчёты старших курсов не переиспользуются.
SALT = "ml-fn1-2026"


# ---------------------------------------------------------------------------
#                          seed и выбор варианта
# ---------------------------------------------------------------------------
def student_seed(identity: str, lab: int = 0) -> int:
    """Детерминированный seed по строке идентификации студента.

    Parameters
    ----------
    identity : str
        ФИО или почта. Регистр, лишние пробелы и `ё`/`е` не влияют.
    lab : int
        Номер лабораторной. Разные лабораторные получают разные seed'ы,
        но выведенные из одного и того же `identity`.
    """
    norm = " ".join(str(identity).strip().lower().replace("ё", "е").split())
    if not norm:
        raise ValueError("Пустая строка идентификации: укажите ФИО или почту.")
    payload = f"{SALT}|{norm}|lab{lab}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _pick(rng: np.random.Generator, options):
    """Равновероятный выбор одного элемента (обёртка над rng для читаемости)."""
    return options[int(rng.integers(len(options)))]


def _pick_k(rng: np.random.Generator, options, k: int):
    """k различных элементов без повторов, порядок фиксирован."""
    idx = rng.choice(len(options), size=k, replace=False)
    return [options[int(i)] for i in sorted(idx)]


# Таблицы вариантов по лабораторным работам. Каждая функция получает rng,
# порождённый из seed студента, и возвращает словарь параметров задания.
def _variant_lab1(rng):
    return {
        "domain": _pick(rng, list(DOMAINS)),
        "n_objects": int(rng.integers(900, 1600)),
        "missing_rate": round(float(rng.uniform(0.05, 0.18)), 3),
        "outlier_rate": round(float(rng.uniform(0.01, 0.04)), 3),
        "duplicate_rate": round(float(rng.uniform(0.01, 0.05)), 3),
        "scaler": _pick(rng, ["StandardScaler", "MinMaxScaler", "RobustScaler"]),
        "encoder": _pick(rng, ["OneHotEncoder", "OrdinalEncoder + OneHot для номинальных"]),
    }


def _variant_lab2(rng):
    return {
        "losses": _pick_k(rng, ["квадратичная", "абсолютная", "Хьюбера", "квантильная (q=0.75)"], 2),
        "solver_pair": _pick(rng, [
            ("нормальные уравнения (np.linalg.solve)", "SVD (np.linalg.lstsq)"),
            ("QR-разложение (scipy.linalg.qr)", "SVD (np.linalg.svd вручную)"),
            ("нормальные уравнения через обращение (np.linalg.inv)", "SVD (np.linalg.svd вручную)"),
        ]),
        "poly_degrees": sorted(_pick_k(rng, [1, 2, 3, 5, 7, 9, 11, 15], 4)),
        "noise_sigma": round(float(rng.uniform(0.3, 1.5)), 2),
    }


def _variant_lab3(rng):
    return {
        "own_regularizer": _pick(rng, ["Ridge (L2)", "LASSO (L1, координатный спуск)", "ElasticNet"]),
        "gd_flavor": _pick(rng, ["постоянный шаг", "затухающий шаг h_t = h0/sqrt(t)", "шаг по Армихо"]),
        "batch_sizes": sorted(_pick_k(rng, [1, 4, 8, 16, 32, 64, 128], 3)),
        "collinearity": round(float(rng.uniform(0.90, 0.999)), 3),
        "lambda_grid": _pick(rng, ["np.logspace(-4, 3, 40)", "np.logspace(-3, 4, 50)", "np.logspace(-5, 2, 60)"]),
    }


def _variant_lab4(rng):
    return {
        "dataset": _pick(rng, ["breast_cancer", "wine (2 класса: 0 против остальных)", "digits (3 против 8)"]),
        "kernels": _pick_k(rng, ["linear", "poly (d=2)", "poly (d=3)", "rbf", "sigmoid"], 3),
        "main_metric": _pick(rng, ["ROC-AUC", "F1", "PR-AUC (average precision)", "сбалансированная точность"]),
        "class_imbalance": _pick(rng, ["без искажения", "оставить 20% объектов класса 1", "оставить 10% объектов класса 1"]),
        "own_solver": _pick(rng, ["scipy.optimize.minimize (SLSQP)", "проекционный градиентный подъём по alpha", "SMO (упрощённый)"]),
    }


def _variant_lab5(rng):
    return {
        "complexity_axis": _pick(rng, [
            "степень полинома в линейной регрессии",
            "глубина решающего дерева",
            "число соседей k в kNN (обратная сложность)",
            "коэффициент регуляризации C в SVM",
        ]),
        "cv_scheme": _pick(rng, ["KFold(k=5)", "KFold(k=10)", "StratifiedKFold(k=5)", "ShuffleSplit(n=20, test=0.25)"]),
        "n_repeats": int(rng.integers(200, 600)),
        "leak_scenario": _pick(rng, [
            "масштабирование до разбиения",
            "отбор признаков по всей выборке",
            "заполнение пропусков средним по всей выборке",
        ]),
    }


def _variant_lab6(rng):
    return {
        "criterion": _pick(rng, ["энтропийный", "Джини", "дисперсионный (регрессия)"]),
        "own_tree_task": _pick(rng, ["классификация", "регрессия"]),
        "ensembles": _pick_k(rng, ["Bagging", "RandomForest", "AdaBoost", "GradientBoosting", "ExtraTrees"], 3),
        "pruning": _pick(rng, ["cost-complexity (ccp_alpha)", "ограничение min_samples_leaf", "ограничение max_depth"]),
        "importance": _pick(rng, ["impurity-based + permutation", "permutation + drop-column"]),
    }


def _variant_lab7(rng):
    return {
        "metric_method": _pick(rng, [
            "kNN с равными весами",
            "kNN с весами w(i) = q^i",
            "парзеновское окно фиксированной ширины",
            "парзеновское окно переменной ширины",
            "метод потенциальных функций",
        ]),
        "kernel": _pick(rng, ["прямоугольное", "треугольное", "Епанечникова", "квартическое", "гауссовское"]),
        "distance": _pick(rng, ["евклидово", "манхэттенское", "Минковского (p=3)", "косинусное"]),
        "bayes_method": _pick(rng, ["наивный байесовский (Gaussian)", "наивный байесовский (Multinomial, текст)", "LDA", "QDA"]),
        "use_stolp": bool(rng.integers(0, 2)),
    }


def _variant_lab8(rng):
    return {
        "linkage": _pick(rng, ["single", "complete", "average", "ward"]),
        "gmm_cov": _pick(rng, ["spherical", "diag", "full"]),
        "k_selection": _pick(rng, ["метод локтя + силуэт", "силуэт + BIC", "BIC + AIC"]),
        "hard_case": _pick(rng, [
            "кластеры разного размера",
            "анизотропные (вытянутые) кластеры",
            "кластеры разной плотности",
            "вложенные кольца",
        ]),
    }


def _variant_lab9(rng):
    return {
        "pca_task": _pick(rng, [
            "сжатие изображений (digits/faces)",
            "визуализация выборки в 2D",
            "PCA как предобработка перед классификатором",
        ]),
        "scaling": _pick(rng, ["без масштабирования vs StandardScaler", "StandardScaler vs RobustScaler"]),
        "n_components_rule": _pick(rng, ["95% объяснённой дисперсии", "90% объяснённой дисперсии", "правило локтя на scree-plot"]),
        "embedding": _pick(rng, ["классическое MDS", "метрическое MDS (SMACOF)", "MDS + сравнение с t-SNE"]),
    }


_VARIANT_BUILDERS = {
    1: _variant_lab1,
    2: _variant_lab2,
    3: _variant_lab3,
    4: _variant_lab4,
    5: _variant_lab5,
    6: _variant_lab6,
    7: _variant_lab7,
    8: _variant_lab8,
    9: _variant_lab9,
}


def get_variant(identity: str, lab: int = 1) -> dict:
    """Вернуть словарь параметров индивидуального варианта.

    Ключи `seed`, `identity`, `lab` есть всегда; остальные зависят от номера
    лабораторной работы.
    """
    if lab not in _VARIANT_BUILDERS:
        raise ValueError(f"Номер лабораторной должен быть от 1 до 9, получено: {lab}")
    seed = student_seed(identity, lab)
    rng = np.random.default_rng(seed)
    variant = {"identity": identity, "lab": lab, "seed": seed}
    variant.update(_VARIANT_BUILDERS[lab](rng))
    # Датасет лабораторной 1 сквозной: его параметры доступны из любой работы.
    if lab != 1:
        variant["personal_table"] = get_variant(identity, 1)
    return variant


def describe_variant(variant: dict) -> None:
    """Печать варианта в читаемом виде (для вставки в отчёт)."""
    print(f"Вариант для «{variant['identity']}», лабораторная работа {variant['lab']}")
    print(f"seed = {variant['seed']}")
    print("-" * 64)
    for key, value in variant.items():
        if key in ("identity", "lab", "seed", "personal_table"):
            continue
        text = ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        wrapped = textwrap.fill(text, width=46, subsequent_indent=" " * 26)
        print(f"  {key:<22} : {wrapped}")


# ---------------------------------------------------------------------------
#                    индивидуальный «грязный» датасет
# ---------------------------------------------------------------------------
# Каждая предметная область порождает свою таблицу признаков и свою целевую
# переменную. Данные синтетические, но воспроизводят типовые дефекты реальных
# таблиц: пропуски, выбросы, дубликаты, «грязные» категории, мусорные и
# «подглядывающие» (leaky) признаки.


def _apartments(rng, n):
    area = np.round(np.clip(rng.lognormal(3.85, 0.36, n), 18, 220), 1)
    rooms = np.clip(np.round(area / 24 + rng.normal(0, 0.55, n)), 1, 6).astype(int)
    floors_total = rng.integers(2, 26, n)
    floor = np.minimum(rng.integers(1, 26, n), floors_total)
    district = rng.choice(["Центр", "Север", "Юг", "Запад", "Восток"], n, p=[0.14, 0.2, 0.26, 0.2, 0.2])
    house = rng.choice(["панельный", "кирпичный", "монолит"], n, p=[0.45, 0.3, 0.25])
    repair = rng.choice(["без ремонта", "косметический", "евро"], n, p=[0.3, 0.45, 0.25])
    metro_min = np.round(np.clip(rng.gamma(2.2, 4.0, n), 1, 60), 1)
    balcony = rng.binomial(1, 0.6, n)
    year = rng.integers(1955, 2024, n)

    price = (
        820 * area
        + 9500 * rooms
        - 950 * metro_min
        + 320 * (year - 1955)
        + np.select([district == "Центр", district == "Запад"], [95000, 42000], 0)
        + np.select([house == "монолит", house == "кирпичный"], [38000, 22000], 0)
        + np.select([repair == "евро", repair == "косметический"], [55000, 18000], 0)
        + 12000 * balcony
        + 900 * (floor > 1) * (floor < floors_total)
        + rng.normal(0, 28000, n)
    )
    frame = pd.DataFrame({
        "площадь_кв_м": area,
        "комнат": rooms,
        "этаж": floor,
        "этажей_в_доме": floors_total,
        "район": district,
        "тип_дома": house,
        "ремонт": repair,
        "до_метро_мин": metro_min,
        "балкон": balcony,
        "год_постройки": year,
    })
    target = np.round(np.clip(price, 25000, None), -2)
    return frame, pd.Series(target, name="цена_аренды"), "regression", ["район", "тип_дома", "ремонт"]


def _bank(rng, n):
    age = np.clip(rng.normal(41, 12, n), 18, 85).round().astype(int)
    income = np.round(np.clip(rng.lognormal(11.0, 0.5, n), 15000, None), -2)
    tenure = np.clip(rng.gamma(2.0, 2.5, n), 0, 40).round(1)
    education = rng.choice(["среднее", "среднее спец.", "высшее", "учёная степень"], n, p=[0.2, 0.3, 0.44, 0.06])
    marital = rng.choice(["холост", "женат", "разведён"], n, p=[0.35, 0.5, 0.15])
    products = rng.integers(1, 6, n)
    has_loan = rng.binomial(1, 0.35, n)
    region = rng.choice(["Москва", "Санкт-Петербург", "Урал", "Сибирь", "Юг"], n)
    balance = np.round(np.clip(rng.normal(180000, 120000, n), 0, None), -2)

    logit = (
        -3.1
        + 0.026 * (age - 41)
        + 1.05e-5 * (income - 60000)
        + 0.09 * tenure
        + np.select([education == "высшее", education == "учёная степень"], [0.45, 0.8], 0.0)
        + 0.22 * products
        - 0.75 * has_loan
        + 1.6e-6 * balance
        + rng.normal(0, 0.35, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    frame = pd.DataFrame({
        "возраст": age,
        "доход": income,
        "стаж_лет": tenure,
        "образование": education,
        "семейное_положение": marital,
        "число_продуктов": products,
        "есть_кредит": has_loan,
        "регион": region,
        "баланс": balance,
    })
    target = rng.binomial(1, prob)
    return frame, pd.Series(target, name="отклик"), "classification", ["образование", "семейное_положение", "регион"]


def _cars(rng, n):
    year = rng.integers(1998, 2024, n)
    mileage = np.round(np.clip(rng.gamma(2.5, 45000, n) * (2024 - year) / 12 + rng.normal(0, 15000, n), 500, 600000), -2)
    engine = np.round(np.clip(rng.normal(1.9, 0.55, n), 0.8, 5.0), 1)
    power = np.round(np.clip(60 * engine + rng.normal(0, 22, n), 45, 500)).astype(int)
    gearbox = rng.choice(["механика", "автомат", "робот", "вариатор"], n, p=[0.4, 0.38, 0.1, 0.12])
    fuel = rng.choice(["бензин", "дизель", "гибрид"], n, p=[0.72, 0.24, 0.04])
    drive = rng.choice(["передний", "задний", "полный"], n, p=[0.6, 0.15, 0.25])
    condition = rng.choice(["требует ремонта", "хорошее", "отличное"], n, p=[0.15, 0.55, 0.3])
    owners = rng.integers(1, 6, n)

    price = (
        45000 * (year - 1998)
        - 1.6 * mileage
        + 2600 * power
        + np.select([gearbox == "автомат", gearbox == "вариатор"], [70000, 35000], 0)
        + np.select([fuel == "дизель", fuel == "гибрид"], [60000, 120000], 0)
        + np.select([drive == "полный"], [95000], 0)
        + np.select([condition == "отличное", condition == "хорошее"], [110000, 40000], 0)
        - 25000 * owners
        + rng.normal(0, 70000, n)
    )
    frame = pd.DataFrame({
        "год_выпуска": year,
        "пробег_км": mileage,
        "объём_двигателя": engine,
        "мощность_лс": power,
        "коробка": gearbox,
        "топливо": fuel,
        "привод": drive,
        "состояние": condition,
        "владельцев": owners,
    })
    target = np.round(np.clip(price, 60000, None), -3)
    return frame, pd.Series(target, name="цена"), "regression", ["коробка", "топливо", "привод", "состояние"]


def _medical(rng, n):
    age = np.clip(rng.normal(52, 14, n), 20, 90).round().astype(int)
    sex = rng.choice(["м", "ж"], n)
    bmi = np.round(np.clip(rng.normal(26.5, 4.5, n), 15, 55), 1)
    pressure = np.clip(rng.normal(128, 17, n) + 0.25 * (age - 52), 85, 220).round().astype(int)
    chol = np.round(np.clip(rng.normal(5.4, 1.1, n) + 0.012 * (age - 52), 2.5, 12.0), 2)
    glucose = np.round(np.clip(rng.normal(5.5, 1.2, n) + 0.05 * (bmi - 26.5), 3.0, 20.0), 2)
    smoking = rng.binomial(1, 0.28, n)
    activity = rng.choice(["низкая", "средняя", "высокая"], n, p=[0.42, 0.4, 0.18])

    logit = (
        -6.0
        + 0.055 * age
        + 0.09 * (bmi - 25)
        + 0.021 * (pressure - 120)
        + 0.38 * (chol - 5.0)
        + 0.30 * (glucose - 5.5)
        + 0.75 * smoking
        + np.select([activity == "низкая", activity == "высокая"], [0.45, -0.5], 0.0)
        + 0.55 * (sex == "м")
        + rng.normal(0, 0.4, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    frame = pd.DataFrame({
        "возраст": age,
        "пол": sex,
        "имт": bmi,
        "давление_систолическое": pressure,
        "холестерин": chol,
        "глюкоза": glucose,
        "курение": smoking,
        "физнагрузка": activity,
    })
    target = rng.binomial(1, prob)
    return frame, pd.Series(target, name="диагноз"), "classification", ["пол", "физнагрузка"]


def _telecom(rng, n):
    tenure = rng.integers(1, 73, n)
    monthly = np.round(np.clip(rng.normal(1150, 380, n), 250, 3500), 2)
    plan = rng.choice(["базовый", "стандарт", "премиум"], n, p=[0.4, 0.42, 0.18])
    contract = rch = rng.choice(["помесячно", "год", "два года"], n, p=[0.55, 0.28, 0.17])
    support = rng.binomial(1, 0.45, n)
    tickets = rng.poisson(1.3, n)
    internet = rng.choice(["DSL", "оптика", "нет"], n, p=[0.34, 0.52, 0.14])
    autopay = rng.binomial(1, 0.5, n)

    logit = (
        -0.4
        - 0.035 * tenure
        + 0.0009 * (monthly - 1150)
        + np.select([rch == "помесячно"], [1.15], 0.0)
        + np.select([contract == "два года"], [-0.9], 0.0)
        - 0.6 * support
        + 0.28 * tickets
        + np.select([internet == "оптика"], [0.35], 0.0)
        - 0.55 * autopay
        + rng.normal(0, 0.4, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    frame = pd.DataFrame({
        "срок_обслуживания_мес": tenure,
        "ежемесячный_платёж": monthly,
        "тариф": plan,
        "тип_договора": contract,
        "техподдержка": support,
        "обращений_в_поддержку": tickets,
        "интернет": internet,
        "автоплатёж": autopay,
    })
    target = rng.binomial(1, prob)
    return frame, pd.Series(target, name="отток"), "classification", ["тариф", "тип_договора", "интернет"]


def _students(rng, n):
    prep_hours = np.round(np.clip(rng.gamma(3.0, 12.0, n), 0, 200), 1)
    attendance = np.round(np.clip(rng.beta(6, 2, n) * 100, 5, 100), 1)
    ege = np.clip(rng.normal(74, 12, n), 40, 100).round().astype(int)
    faculty = rng.choice(["ФН", "СМ", "МТ", "ИУ", "РЛ"], n)
    form = rng.choice(["очная", "вечерняя"], n, p=[0.8, 0.2])
    course = rng.integers(1, 5, n)
    scholarship = rng.binomial(1, 0.4, n)
    works = rng.binomial(1, 0.3, n)

    score = (
        18
        + 0.19 * prep_hours
        + 0.22 * attendance
        + 0.28 * (ege - 74)
        + np.select([faculty == "ФН", faculty == "ИУ"], [4.0, 2.0], 0.0)
        + 3.5 * scholarship
        - 5.0 * works
        + np.select([form == "вечерняя"], [-3.0], 0.0)
        + rng.normal(0, 6.5, n)
    )
    frame = pd.DataFrame({
        "часов_подготовки": prep_hours,
        "посещаемость_проц": attendance,
        "балл_егэ": ege,
        "факультет": faculty,
        "форма_обучения": form,
        "курс": course,
        "стипендия": scholarship,
        "работает": works,
    })
    target = np.round(np.clip(score, 0, 100), 1)
    return frame, pd.Series(target, name="итоговый_балл"), "regression", ["факультет", "форма_обучения"]


DOMAINS = {
    "квартиры": _apartments,
    "банк": _bank,
    "автомобили": _cars,
    "медицина": _medical,
    "телеком": _telecom,
    "студенты": _students,
}


def _spoil(frame, target, rng, variant, categorical):
    """Внести в чистую таблицу типовые дефекты реальных данных."""
    n = len(frame)
    numeric = [c for c in frame.columns if c not in categorical]
    # Непрерывные признаки: в них осмысленно портить масштаб и формат записи.
    # Бинарные и малозначные счётчики трогаем только пропусками.
    continuous = [c for c in numeric if frame[c].nunique() > 15] or numeric

    # (1) Пропуски. Один столбец -- MCAR (полностью случайно),
    #     другой -- MAR (вероятность пропуска зависит от другого признака).
    mcar_col = numeric[int(rng.integers(len(numeric)))]
    mask = rng.random(n) < variant["missing_rate"]
    frame.loc[mask, mcar_col] = np.nan

    mar_col = [c for c in numeric if c != mcar_col][0]
    driver = frame[numeric[0]].to_numpy(dtype=float)
    driver_rank = pd.Series(driver).rank(pct=True).to_numpy()
    mask = rng.random(n) < variant["missing_rate"] * 1.8 * driver_rank
    frame.loc[mask, mar_col] = np.nan

    cat_col = categorical[0]
    mask = rng.random(n) < variant["missing_rate"] * 0.6
    frame.loc[mask, cat_col] = np.nan

    # (2) Выбросы: экстремальные значения и «опечатки» (лишний порядок).
    #     Целочисленные столбцы предварительно приводим к float.
    for col in continuous:
        frame[col] = frame[col].astype(float)
    out_col = continuous[-1]
    idx = rng.choice(n, size=max(3, int(variant["outlier_rate"] * n)), replace=False)
    frame.loc[frame.index[idx], out_col] = frame[out_col].max() * rng.uniform(4, 12, len(idx))

    typo_col = continuous[0]
    idx = rng.choice(n, size=max(2, int(0.004 * n)), replace=False)
    frame.loc[frame.index[idx], typo_col] = frame[typo_col].iloc[idx].to_numpy() * 100

    # (3) «Грязные» категории: разный регистр и хвостовые пробелы.
    dirty = frame[cat_col].astype("object").copy()
    idx = rng.choice(n, size=int(0.25 * n), replace=False)
    dirty.iloc[idx] = dirty.iloc[idx].map(lambda s: s.upper() if isinstance(s, str) else s)
    idx = rng.choice(n, size=int(0.15 * n), replace=False)
    dirty.iloc[idx] = dirty.iloc[idx].map(lambda s: f" {s} " if isinstance(s, str) else s)
    frame[cat_col] = dirty

    # (4) Числовой столбец, записанный строкой: разделитель разрядов и запятая
    #     как десятичный разделитель. При наивном read_csv станет object.
    str_col = continuous[1] if len(continuous) > 1 else continuous[0]
    frame[str_col] = frame[str_col].map(
        lambda v: v if pd.isna(v) else f"{v:,.2f}".replace(",", " ").replace(".", ",")
    )

    # (5) Мусорные признаки: константа и почти константа.
    frame["служебный_код"] = "AX-17"
    flag = np.zeros(n, dtype=int)
    flag[rng.choice(n, size=3, replace=False)] = 1
    frame["флаг_проверки"] = flag

    # (6) Ловушка на утечку: идентификатор выдан почти в порядке возрастания
    #     цели (данные когда-то были отсортированы). Признак `id` даёт
    #     завышенное качество и не имеет содержательного смысла.
    #     Шум подобран так, чтобы связь была сильной, но не идеальной:
    #     студент должен увидеть «слишком хорошее» качество, а не тождество.
    #     Для бинарной цели ранги расслоены, поэтому шум нужен крупнее.
    sigma = 0.35 if target.nunique() <= 2 else 0.15
    noisy_rank = target.rank(pct=True).to_numpy() + rng.normal(0, sigma, n)
    order = np.argsort(noisy_rank, kind="stable")
    ident = np.empty(n, dtype=int)
    ident[order] = np.arange(1, n + 1) * 7 + 1000
    frame.insert(0, "id", ident)

    # (7) Дубликаты строк.
    k = int(variant["duplicate_rate"] * n)
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        frame = pd.concat([frame, frame.iloc[idx]], ignore_index=False)
        target = pd.concat([target, target.iloc[idx]], ignore_index=False)

    # (8) Перемешивание строк, чтобы порядок ничего не подсказывал.
    perm = rng.permutation(len(frame))
    frame = frame.iloc[perm].reset_index(drop=True)
    target = target.iloc[perm].reset_index(drop=True)
    return frame, target


def make_table(variant: dict, clean: bool = False):
    """Построить индивидуальную таблицу по варианту.

    Parameters
    ----------
    variant : dict
        Результат ``get_variant(identity, lab=1)`` (либо любой вариант --
        тогда используется вложенный ключ ``personal_table``).
    clean : bool
        Если True, вернуть таблицу без внесённых дефектов (для преподавателя
        и для сверки).

    Returns
    -------
    df : pandas.DataFrame
        Таблица признаков вместе со столбцом целевой переменной (последний).
    meta : dict
        ``task`` ('regression' / 'classification'), ``target``, ``categorical``.
    """
    if variant.get("lab") != 1 and "personal_table" in variant:
        variant = variant["personal_table"]
    rng = np.random.default_rng(variant["seed"])
    frame, target, task, categorical = DOMAINS[variant["domain"]](rng, variant["n_objects"])
    if not clean:
        frame, target = _spoil(frame, target, rng, variant, categorical)
    df = frame.copy()
    df[target.name] = target.to_numpy()
    meta = {
        "task": task,
        "target": target.name,
        "categorical": categorical,
        "domain": variant["domain"],
        "seed": variant["seed"],
    }
    return df, meta


if __name__ == "__main__":  # быстрая самопроверка: python variants.py
    import sys

    who = sys.argv[1] if len(sys.argv) > 1 else "Иванов Иван Иванович"
    for lab in range(1, 10):
        describe_variant(get_variant(who, lab))
        print()
    table, info = make_table(get_variant(who, 1))
    print(info)
    print(table.head())
    print(table.dtypes)
