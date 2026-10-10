"""Partition, as JMP's: a decision tree of one response on many factors.

A continuous Y gives a regression tree (each leaf predicts a mean), a nominal
Y a classification tree (each leaf predicts a level and its probabilities).
The tree grows best split first, one split at a time - scikit-learn's
``max_leaf_nodes`` growth is exactly that order - so "Split" and "Prune" are
one more or one fewer leaf, and the history of R² against the number of
splits comes from growing the same tree a split at a time.

Nominal factors are split by groups of levels, as JMP splits them: the levels
are ordered by the response (their mean, or their share of the first level)
and a threshold on that order is a split into two groups - the optimal binary
grouping for a regression tree, a close one for a classification tree.

No Qt here: the table operation dialog runs this on a worker thread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CONTINUOUS = "continuous"
NOMINAL = "nominal"

TRAINING = "Training"
VALIDATION = "Validation"


@dataclass(slots=True)
class PartitionSpec:
    """What to partition and how."""

    response: str
    factors: list[str]
    #: Column name -> CONTINUOUS or NOMINAL; the response's decides the tree.
    kinds: dict[str, str]
    #: Splits wanted (leaves - 1); ignored when ``auto``.
    splits: int = 3
    #: The fewest rows a leaf may hold.
    min_leaf: int = 5
    #: A column saying which rows validate (0/1, or Training/Validation), if any.
    validation_column: str | None = None
    #: Otherwise this share of the rows, drawn at random, validates.
    validation_fraction: float = 0.0
    seed: int = 1
    #: Grow while validation improves, up to ``max_splits``.
    auto: bool = False
    max_splits: int = 30
    #: TREE, FOREST (JMP's Bootstrap Forest) or BOOSTED (Boosted Tree).
    method: str = "tree"
    #: Trees in a forest or a boosted model; for a boosted one, ``splits`` is per tree.
    trees: int = 100
    #: A boosted model's learning rate: how much each tree corrects the ones before.
    learning_rate: float = 0.1


TREE, FOREST, BOOSTED = "tree", "forest", "boosted"
METHODS: tuple[tuple[str, str], ...] = ((TREE, "Decision Tree"), (FOREST, "Bootstrap Forest"), (BOOSTED, "Boosted Tree"))


@dataclass(slots=True)
class PartitionResult:
    """The tree and its report."""

    response: str
    kind: str
    factors: list[str]
    splits: int
    #: Leaf, Rule, Count, then Mean (continuous) or Prediction and one Prob(level) each.
    leaves: pd.DataFrame
    #: Splits, then R² or Misclassification for training (and validation).
    history: pd.DataFrame
    #: Column, Splits, contribution (SS or G²), Portion.
    contributions: pd.DataFrame
    summary: dict[str, float]
    #: Per row of the frame given (its index): leaf number, prediction, Training/Validation.
    leaf: pd.Series
    predicted: pd.Series
    role: pd.Series
    #: Per level, the predicted probability, for a nominal response.
    probabilities: pd.DataFrame | None = None
    #: Actual (rows) by predicted (columns), the training rows', for a nominal response.
    confusion: pd.DataFrame | None = None
    levels: list[str] = field(default_factory=list)
    note: str = ""
    #: The spec's method: TREE has leaves and a split history; an ensemble neither.
    method: str = "tree"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def _validation_mask(frame: pd.DataFrame, spec: PartitionSpec) -> np.ndarray:
    """True for the rows that validate."""
    if spec.validation_column:
        values = frame[spec.validation_column]
        as_text = values.astype(str).str.strip().str.lower()
        return as_text.isin({"1", "1.0", "true", "validation", "valid", "v"}).to_numpy()
    if spec.validation_fraction > 0:
        rng = np.random.default_rng(spec.seed)
        return rng.random(len(frame)) < float(spec.validation_fraction)
    return np.zeros(len(frame), dtype=bool)


@dataclass(slots=True)
class _Encoded:
    """The factors as numbers, and how to read a threshold back."""

    matrix: np.ndarray
    #: Nominal column -> its levels in code order (code i is levels[i]).
    orders: dict[str, list[str]]


def _encode(frame: pd.DataFrame, spec: PartitionSpec, target: np.ndarray, train: np.ndarray) -> _Encoded:
    """Continuous factors as they are; nominal ones as the rank of each level by the response."""
    columns: list[np.ndarray] = []
    orders: dict[str, list[str]] = {}
    for name in spec.factors:
        if spec.kinds.get(name) == NOMINAL:
            text = frame[name].astype(str)
            # Ordered by the training rows' response: mean of Y, or share of
            # the first level - so a threshold is a grouping of levels.
            score = pd.Series(target[train], index=text[train].to_numpy()).groupby(level=0).mean()
            levels = sorted(set(text), key=lambda level: (score.get(level, np.inf), level))
            codes = {level: index for index, level in enumerate(levels)}
            columns.append(text.map(codes).to_numpy(dtype=float))
            orders[name] = levels
        else:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
    return _Encoded(np.column_stack(columns) if columns else np.empty((len(frame), 0)), orders)


# ----------------------------------------------------------------------
# The tree
# ----------------------------------------------------------------------

def _model(spec: PartitionSpec, leaves: int):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    common = {
        "max_leaf_nodes": max(2, int(leaves)),
        "min_samples_leaf": max(1, int(spec.min_leaf)),
        "random_state": 0,
    }
    if spec.kinds.get(spec.response) == NOMINAL:
        return DecisionTreeClassifier(criterion="entropy", **common)
    return DecisionTreeRegressor(**common)


def _score(model: Any, x: np.ndarray, y: np.ndarray, nominal: bool) -> float:
    """R² (continuous) or misclassification rate (nominal); NaN on no rows."""
    if len(y) == 0:
        return math.nan
    predicted = model.predict(x)
    if nominal:
        return float(np.mean(predicted != y))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else math.nan


def _leaf_count(model: Any) -> int:
    return int(model.tree_.n_leaves)


def _rules(model: Any, factors: list[str], orders: dict[str, list[str]], kinds: dict[str, str]) -> dict[int, str]:
    """Node id -> the rule that leads to that leaf, JMP-like: "x ≥ 3 & g ∈ {a, b}"."""
    tree = model.tree_
    rules: dict[int, str] = {}

    def describe(bounds: dict[int, list[float]]) -> str:
        parts = []
        for feature, (low, high) in sorted(bounds.items()):
            name = factors[feature]
            if kinds.get(name) == NOMINAL:
                levels = orders[name]
                chosen = [level for code, level in enumerate(levels) if low < code <= high]
                parts.append(f"{name} ∈ {{{', '.join(sorted(chosen))}}}")
            elif math.isinf(low):
                parts.append(f"{name} < {high:.6g}")
            elif math.isinf(high):
                parts.append(f"{name} ≥ {low:.6g}")
            else:
                parts.append(f"{low:.6g} ≤ {name} < {high:.6g}")
        return " & ".join(parts) or "All rows"

    def walk(node: int, bounds: dict[int, list[float]]) -> None:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:  # a leaf
            rules[node] = describe(bounds)
            return
        feature, threshold = int(tree.feature[node]), float(tree.threshold[node])
        low, high = bounds.get(feature, [-math.inf, math.inf])
        walk(left, {**bounds, feature: [low, min(high, threshold)]})
        walk(right, {**bounds, feature: [max(low, threshold), high]})

    walk(0, {})
    # sklearn sends x <= t left; shown as "< t" on the left and "≥ t" on the
    # right with the cut written as the threshold, as JMP writes its cuts.
    return rules


def _contributions(model: Any, factors: list[str], nominal: bool) -> pd.DataFrame:
    """Each factor's share of what the splits explain: SS, or G² for a nominal response."""
    tree = model.tree_
    gain = np.zeros(len(factors))
    count = np.zeros(len(factors), dtype=int)
    for node in range(tree.node_count):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            continue
        weighted = (
            tree.weighted_n_node_samples[node] * tree.impurity[node]
            - tree.weighted_n_node_samples[left] * tree.impurity[left]
            - tree.weighted_n_node_samples[right] * tree.impurity[right]
        )
        gain[tree.feature[node]] += weighted
        count[tree.feature[node]] += 1
    if nominal:
        gain = 2.0 * math.log(2.0) * gain  # entropy in bits -> G² (likelihood ratio chi-square)
    total = float(gain.sum())
    frame = pd.DataFrame({
        "Column": factors,
        "Splits": count,
        "G^2" if nominal else "SS": gain,
        "Portion": gain / total if total > 0 else np.zeros(len(factors)),
    })
    return frame.sort_values("Portion", ascending=False, kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------
# Partition
# ----------------------------------------------------------------------

def partition(frame: pd.DataFrame, spec: PartitionSpec) -> PartitionResult:
    """Grow the tree on *frame* as *spec* says and report it."""
    if not spec.factors:
        raise ValueError("Choose at least one X factor.")
    nominal = spec.kinds.get(spec.response) == NOMINAL
    used = [spec.response, *spec.factors] + ([spec.validation_column] if spec.validation_column else [])
    data = frame[used].copy()
    for name in spec.factors:
        if spec.kinds.get(name) != NOMINAL:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    if not nominal:
        data[spec.response] = pd.to_numeric(data[spec.response], errors="coerce")
    complete = data.dropna(subset=[spec.response, *spec.factors])
    dropped = len(data) - len(complete)
    if len(complete) < 2 * max(1, spec.min_leaf):
        raise ValueError(f"Too few complete rows ({len(complete)}) for leaves of {spec.min_leaf}.")

    valid = _validation_mask(complete, spec)
    train = ~valid
    if not train.any():
        raise ValueError("No training rows: every row is marked for validation.")

    if nominal:
        y_text = complete[spec.response].astype(str).to_numpy()
        levels = sorted(set(y_text))
        y = y_text
        target_score = (y_text == levels[0]).astype(float)
    else:
        levels = []
        y = complete[spec.response].to_numpy(dtype=float)
        target_score = y
    encoded = _encode(complete, spec, target_score, train)
    x = encoded.matrix
    if spec.method in (FOREST, BOOSTED):
        return _ensemble(spec, complete, x, y, train, valid, nominal, levels, dropped)

    # The history: the same tree grown one split at a time.
    largest = max(1, min(spec.max_splits if spec.auto else spec.splits, len(complete) // max(1, spec.min_leaf)))
    history_rows = []
    best_splits, best_score = 1, math.inf
    for splits in range(1, largest + 1):
        model = _model(spec, splits + 1).fit(x[train], y[train])
        if splits > 1 and _leaf_count(model) - 1 < splits:
            break  # min_leaf allows no further split
        train_score = _score(model, x[train], y[train], nominal)
        valid_score = _score(model, x[valid], y[valid], nominal) if valid.any() else math.nan
        history_rows.append((splits, train_score, valid_score))
        # Lower misclassification, higher R², on validation (training without it).
        judged = valid_score if valid.any() else train_score
        loss = judged if nominal else -judged
        if spec.auto and not math.isnan(loss) and loss < best_score - 1e-12:
            best_splits, best_score = splits, loss
    measure = "Misclassification" if nominal else "R²"
    history = pd.DataFrame(history_rows, columns=["Splits", f"{measure} (training)", f"{measure} (validation)"])
    if not valid.any():
        history = history.drop(columns=[f"{measure} (validation)"])

    splits = best_splits if spec.auto else (int(history["Splits"].iloc[-1]) if len(history) else 1)
    note = ""
    if spec.auto and not valid.any():
        note = "Automatic splitting needs validation rows; the tree has the most splits the leaves allow."
    model = _model(spec, splits + 1).fit(x[train], y[train])

    # Leaves numbered left to right, with their rules.
    rules = _rules(model, spec.factors, encoded.orders, spec.kinds)
    leaf_nodes = list(rules)
    number = {node: index + 1 for index, node in enumerate(leaf_nodes)}
    node_of_row = model.apply(x)
    leaf = pd.Series([number[node] for node in node_of_row], index=complete.index, name="Leaf")
    role = pd.Series(np.where(valid, VALIDATION, TRAINING), index=complete.index, name="Role")

    leaf_rows = []
    probabilities = None
    confusion = None
    if nominal:
        proba = model.predict_proba(x)
        classes = [str(c) for c in model.classes_]
        predicted = pd.Series(model.predict(x).astype(str), index=complete.index)
        probabilities = pd.DataFrame(proba, index=complete.index, columns=[f"Prob({c})" for c in classes])
        for node in leaf_nodes:
            counts = model.tree_.value[node][0]
            shares = counts / counts.sum() if counts.sum() else counts
            in_leaf = (node_of_row == node) & train
            leaf_rows.append({
                "Leaf": number[node], "Rule": rules[node], "Count": int(in_leaf.sum()),
                "Prediction": classes[int(np.argmax(shares))],
                **{f"Prob({c})": float(p) for c, p in zip(classes, shares)},
            })
        confusion = pd.crosstab(
            pd.Series(y[train], name="Actual"), pd.Series(predicted.to_numpy()[train], name="Predicted"),
        )
    else:
        predicted = pd.Series(model.predict(x), index=complete.index)
        for node in leaf_nodes:
            in_leaf = (node_of_row == node) & train
            values = y[in_leaf]
            leaf_rows.append({
                "Leaf": number[node], "Rule": rules[node], "Count": int(in_leaf.sum()),
                "Mean": float(values.mean()) if len(values) else math.nan,
                "Std Dev": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
            })

    summary: dict[str, float] = {
        "Rows (training)": float(train.sum()),
        "Rows (validation)": float(valid.sum()),
        "Splits": float(splits),
        "Leaves": float(len(leaf_nodes)),
    }
    if nominal:
        summary["Misclassification (training)"] = _score(model, x[train], y[train], True)
        if valid.any():
            summary["Misclassification (validation)"] = _score(model, x[valid], y[valid], True)
        # Entropy R²: how much of the null model's -log-likelihood the tree explains.
        prob_train = model.predict_proba(x[train])
        index = {c: i for i, c in enumerate(model.classes_)}
        picked = np.clip(prob_train[np.arange(train.sum()), [index[v] for v in y[train]]], 1e-12, 1)
        shares = pd.Series(y[train]).value_counts(normalize=True)
        null = -float(np.sum(np.log(shares.reindex(y[train]).to_numpy())))
        summary["Entropy R² (training)"] = 1.0 - (-float(np.sum(np.log(picked)))) / null if null > 0 else math.nan
    else:
        summary["R² (training)"] = _score(model, x[train], y[train], False)
        residual = y[train] - model.predict(x[train])
        summary["RMSE (training)"] = float(np.sqrt(np.mean(residual**2)))
        if valid.any():
            summary["R² (validation)"] = _score(model, x[valid], y[valid], False)
    if dropped:
        note = (note + " " if note else "") + f"{dropped} rows with a missing value were left out."

    return PartitionResult(
        response=spec.response,
        kind=NOMINAL if nominal else CONTINUOUS,
        factors=list(spec.factors),
        splits=splits,
        leaves=pd.DataFrame(leaf_rows),
        history=history,
        contributions=_contributions(model, spec.factors, nominal),
        summary=summary,
        leaf=leaf,
        predicted=predicted,
        role=role,
        probabilities=probabilities,
        confusion=confusion,
        levels=levels,
        note=note,
    )


# ----------------------------------------------------------------------
# Bootstrap Forest and Boosted Tree
# ----------------------------------------------------------------------

def _ensemble_model(spec: PartitionSpec, nominal: bool, trees: int):
    common = {"n_estimators": max(1, int(trees)), "min_samples_leaf": max(1, int(spec.min_leaf)),
              "random_state": int(spec.seed)}
    if spec.method == FOREST:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return (RandomForestClassifier(criterion="entropy", **common) if nominal
                else RandomForestRegressor(**common))
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    # JMP's Boosted Tree: many small trees (``splits`` each), each fitted to
    # what the ones before left unexplained.
    boosted = {**common, "learning_rate": float(spec.learning_rate), "max_leaf_nodes": max(2, int(spec.splits) + 1)}
    return GradientBoostingClassifier(**boosted) if nominal else GradientBoostingRegressor(**boosted)


def _staged(model: Any, spec: PartitionSpec, x: np.ndarray, nominal: bool):
    """(trees, prediction) after each tree: the forest's running mean, the boosted model's stages."""
    if spec.method == BOOSTED:
        yield from enumerate(model.staged_predict(x), start=1)
        return
    total = None
    for count, tree in enumerate(model.estimators_, start=1):
        part = tree.predict_proba(x) if nominal else tree.predict(x)
        total = part if total is None else total + part
        yield count, (model.classes_[np.argmax(total, axis=1)] if nominal else total / count)


def _staged_score(predicted: np.ndarray, y: np.ndarray, nominal: bool) -> float:
    if len(y) == 0:
        return math.nan
    if nominal:
        return float(np.mean(predicted != y))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else math.nan


def _ensemble(spec: PartitionSpec, complete: pd.DataFrame, x: np.ndarray, y: np.ndarray, train: np.ndarray,
              valid: np.ndarray, nominal: bool, levels: list[str], dropped: int) -> PartitionResult:
    """A Bootstrap Forest or a Boosted Tree: no single tree to read, so no leaves -
    the report is the fit, the columns' contributions and the score by number of trees."""
    model = _ensemble_model(spec, nominal, spec.trees).fit(x[train], y[train])
    measure = "Misclassification" if nominal else "R²"
    checkpoints = set(np.unique(np.linspace(1, spec.trees, min(spec.trees, 25)).round().astype(int)))
    rows = []
    for count, predicted_all in _staged(model, spec, x, nominal):
        if count in checkpoints:
            rows.append((count, _staged_score(predicted_all[train], y[train], nominal),
                         _staged_score(predicted_all[valid], y[valid], nominal) if valid.any() else math.nan))
    history = pd.DataFrame(rows, columns=["Trees", f"{measure} (training)", f"{measure} (validation)"])
    if not valid.any():
        history = history.drop(columns=[f"{measure} (validation)"])

    trees, note = spec.trees, ""
    if spec.auto and spec.method == BOOSTED:
        if valid.any():
            # Stop where validation was best, as JMP's early stopping.
            # On a tie, the most trees: their probabilities are the better ones.
            judged = history[f"{measure} (validation)"][::-1]
            trees = int(history["Trees"][judged.idxmin() if nominal else judged.idxmax()])
            model = _ensemble_model(spec, nominal, trees).fit(x[train], y[train])
        else:
            note = "Automatic stopping needs validation rows; every tree was kept."

    # Each column's share of the impurity the trees removed, and how often it split.
    estimators = model.estimators_.ravel() if spec.method == BOOSTED else model.estimators_
    used = np.zeros(len(spec.factors), dtype=int)
    for tree in estimators:
        features = tree.tree_.feature
        np.add.at(used, features[features >= 0], 1)
    importance = model.feature_importances_
    contributions = pd.DataFrame({
        "Column": spec.factors, "Splits": used, "G^2" if nominal else "SS": importance,
        "Portion": importance / importance.sum() if importance.sum() > 0 else importance,
    }).sort_values("Portion", ascending=False, kind="stable").reset_index(drop=True)

    role = pd.Series(np.where(valid, VALIDATION, TRAINING), index=complete.index, name="Role")
    summary: dict[str, float] = {"Rows (training)": float(train.sum()), "Rows (validation)": float(valid.sum()),
                                 "Trees": float(trees)}
    probabilities = confusion = None
    if nominal:
        predicted = pd.Series(model.predict(x).astype(str), index=complete.index)
        proba = model.predict_proba(x)
        classes = [str(c) for c in model.classes_]
        probabilities = pd.DataFrame(proba, index=complete.index, columns=[f"Prob({c})" for c in classes])
        summary["Misclassification (training)"] = _staged_score(predicted.to_numpy()[train], y[train], True)
        if valid.any():
            summary["Misclassification (validation)"] = _staged_score(predicted.to_numpy()[valid], y[valid], True)
        index = {c: i for i, c in enumerate(model.classes_)}
        picked = np.clip(proba[train][np.arange(train.sum()), [index[v] for v in y[train]]], 1e-12, 1)
        shares = pd.Series(y[train]).value_counts(normalize=True)
        null = -float(np.sum(np.log(shares.reindex(y[train]).to_numpy())))
        summary["Entropy R² (training)"] = 1.0 - (-float(np.sum(np.log(picked)))) / null if null > 0 else math.nan
        confusion = pd.crosstab(pd.Series(y[train], name="Actual"),
                                pd.Series(predicted.to_numpy()[train], name="Predicted"))
    else:
        predicted = pd.Series(model.predict(x), index=complete.index)
        summary["R² (training)"] = _staged_score(predicted.to_numpy()[train], y[train], False)
        residual = y[train] - predicted.to_numpy()[train]
        summary["RMSE (training)"] = float(np.sqrt(np.mean(residual**2)))
        if valid.any():
            summary["R² (validation)"] = _staged_score(predicted.to_numpy()[valid], y[valid], False)
    if dropped:
        note = (note + " " if note else "") + f"{dropped} rows with a missing value were left out."

    return PartitionResult(
        response=spec.response, kind=NOMINAL if nominal else CONTINUOUS, factors=list(spec.factors),
        splits=int(spec.splits), leaves=pd.DataFrame(), history=history, contributions=contributions,
        summary=summary, leaf=pd.Series(np.nan, index=complete.index, name="Leaf"), predicted=predicted,
        role=role, probabilities=probabilities, confusion=confusion, levels=levels, note=note,
        method=spec.method,
    )
