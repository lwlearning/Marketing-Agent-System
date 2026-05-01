from typing import Dict, Any

import numpy as np
import pandas as pd

EPS = 1e-8


def normalize_policy_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in {"bandit", "contextual", "contextual_bandit"}:
        return "contextual_bandit"
    if key in {"uplift"}:
        return "uplift"
    if key in {"fusion", "hybrid", "uplift_bandit"}:
        return "fusion"
    return "contextual_bandit"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (x - mean) / std


def _row_standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.clip(std, EPS, None)


class LinearRewardModel:
    def __init__(self, n_actions: int, dim: int, default_reward: float):
        self.n_actions = n_actions
        self.dim = dim
        self.default_reward = float(default_reward)
        self.thetas = np.zeros((n_actions, dim), dtype=float)
        self.a_invs = [np.eye(dim, dtype=float) for _ in range(n_actions)]
        self.action_means = np.full(n_actions, self.default_reward, dtype=float)
        self.action_counts = np.zeros(n_actions, dtype=int)

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        pscores: np.ndarray,
        reg: float = 1.0,
        min_samples: int = 20,
        use_ipw: bool = True
    ) -> "LinearRewardModel":
        for action in range(self.n_actions):
            mask = actions == action
            self.action_counts[action] = int(mask.sum())
            self.action_means[action] = float(rewards[mask].mean()) if np.any(mask) else self.default_reward

            if self.action_counts[action] < min_samples:
                self.thetas[action] = 0.0
                self.a_invs[action] = np.eye(self.dim, dtype=float)
                continue

            xa = contexts[mask]
            ya = rewards[mask]

            if use_ipw:
                wa = 1.0 / np.clip(pscores[mask], 1e-6, None)
                wa = np.clip(wa, 1.0, 50.0)
            else:
                wa = np.ones_like(ya)

            sw = np.sqrt(wa).reshape(-1, 1)
            xw = xa * sw
            yw = ya * np.sqrt(wa)

            a_mat = xw.T @ xw + reg * np.eye(self.dim, dtype=float)
            b_vec = xw.T @ yw

            try:
                theta = np.linalg.solve(a_mat, b_vec)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(a_mat) @ b_vec

            self.thetas[action] = theta

            try:
                self.a_invs[action] = np.linalg.inv(a_mat)
            except np.linalg.LinAlgError:
                self.a_invs[action] = np.linalg.pinv(a_mat)

        return self

    def predict_values(self, contexts: np.ndarray) -> np.ndarray:
        if contexts.ndim == 1:
            contexts = contexts.reshape(1, -1)
        values = contexts @ self.thetas.T
        for action in range(self.n_actions):
            if self.action_counts[action] < 20:
                values[:, action] = self.action_means[action]
        return np.clip(values, 0.0, 1.0)


def build_logged_bandit_feedback(
    rfm: pd.DataFrame,
    n_actions: int = 5,
    random_state: int = 42,
    min_pscore: float = 0.03
) -> Dict[str, Any]:
    if len(rfm) == 0:
        raise ValueError("rfm 为空，无法构建 bandit 反馈数据")

    df = rfm.copy()

    required_numeric = [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "avg_items_per_order",
        "customer_lifetime",
    ]
    for col in required_numeric:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(float(median) if pd.notna(median) else 0.0)

    if "user_segment" not in df.columns:
        df["user_segment"] = "一般价值用户"
    if "customer_unique_id" not in df.columns:
        df["customer_unique_id"] = [f"user_{i}" for i in range(len(df))]

    seg_codes, seg_uniques = pd.factorize(df["user_segment"].astype(str))
    context_raw = np.column_stack(
        [
            df["recency"].to_numpy(float),
            df["frequency"].to_numpy(float),
            df["monetary"].to_numpy(float),
            df["avg_order_value"].to_numpy(float),
            df["avg_items_per_order"].to_numpy(float),
            df["customer_lifetime"].to_numpy(float),
            seg_codes.astype(float),
        ]
    )
    context = _zscore(context_raw)

    rng = np.random.default_rng(random_state)
    n_rounds, dim = context.shape

    behavior_w = rng.normal(0.0, 0.35, size=(dim, n_actions))
    behavior_b = rng.normal(0.0, 0.12, size=(n_actions,))
    behavior_logits = context @ behavior_w + behavior_b
    behavior_prob = _softmax(behavior_logits)

    if 0.0 < min_pscore < 1.0 / n_actions:
        behavior_prob = behavior_prob * (1.0 - n_actions * min_pscore) + min_pscore
        behavior_prob = behavior_prob / np.clip(behavior_prob.sum(axis=1, keepdims=True), EPS, None)

    actions = np.array([rng.choice(n_actions, p=behavior_prob[i]) for i in range(n_rounds)], dtype=int)
    pscores = behavior_prob[np.arange(n_rounds), actions]

    reward_w = rng.normal(0.0, 0.45, size=(dim, n_actions))
    action_bias = np.linspace(-0.08, 0.14, n_actions)
    reward_logits = context @ reward_w + action_bias
    reward_logits += 0.2 * np.tanh(context[:, [0]]) * np.linspace(-1.0, 1.0, n_actions)
    reward_prob = _sigmoid(reward_logits)
    chosen_prob = reward_prob[np.arange(n_rounds), actions]
    rewards = rng.binomial(1, np.clip(chosen_prob, 0.02, 0.98)).astype(float)

    return {
        "n_rounds": int(n_rounds),
        "context": context.astype(float),
        "action": actions.astype(int),
        "reward": rewards.astype(float),
        "pscore": np.clip(pscores.astype(float), 1e-6, None),
        "customer_unique_id": df["customer_unique_id"].astype(str).tolist(),
        "segment_mapping": {int(i): str(v) for i, v in enumerate(seg_uniques.tolist())},
    }


def contextual_score_matrix(contexts: np.ndarray, model: LinearRewardModel, alpha: float = 0.6) -> np.ndarray:
    means = model.predict_values(contexts)
    uncertainty = np.zeros_like(means)
    for action in range(model.n_actions):
        a_inv = model.a_invs[action]
        uncertainty[:, action] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", contexts, a_inv, contexts), 0.0))
        if model.action_counts[action] < 10:
            uncertainty[:, action] += 0.2
    return means + alpha * uncertainty


def uplift_score_matrix(contexts: np.ndarray, model: LinearRewardModel, baseline_action: int) -> np.ndarray:
    means = model.predict_values(contexts)
    uplift = means - means[:, [baseline_action]]
    uplift[:, baseline_action] = 0.0
    return uplift


def fusion_score_matrix(
    contextual_scores: np.ndarray,
    uplift_scores: np.ndarray,
    reward_means: np.ndarray,
    uplift_weight: float = 0.45,
    bandit_weight: float = 0.55
) -> np.ndarray:
    contextual_norm = _row_standardize(contextual_scores)
    uplift_norm = _row_standardize(uplift_scores)
    return bandit_weight * contextual_norm + uplift_weight * uplift_norm + 0.1 * reward_means


def estimate_ope(
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    pscores: np.ndarray,
    target_actions: np.ndarray,
    q_hat: np.ndarray,
    policy_name: str
) -> Dict[str, Any]:
    n = len(rewards)
    idx = np.arange(n)
    indicator = (logged_actions == target_actions).astype(float)
    w = indicator / np.clip(pscores, 1e-6, None)

    ips = np.mean(w * rewards)
    weight_sum = float(np.sum(w))
    snips = float(np.sum(w * rewards) / weight_sum) if weight_sum > 0 else 0.0

    q_pi = q_hat[idx, target_actions]
    q_logged = q_hat[idx, logged_actions]
    dr = np.mean(q_pi + w * (rewards - q_logged))

    return {
        "方法": "IPS/SNIPS/DR",
        "策略名称": policy_name,
        "行为策略平均回报": round(float(np.mean(rewards)), 6),
        "动作重合率": round(float(np.mean(indicator)), 6),
        "IPS策略价值估计": round(float(ips), 6),
        "SNIPS策略价值估计": round(float(snips), 6),
        "DR策略价值估计": round(float(dr), 6),
        "有效样本权重和": round(weight_sum, 6),
    }


def prepare_offline_bandit_suite(
    rfm: pd.DataFrame,
    n_actions: int = 5,
    random_state: int = 42,
    min_pscore: float = 0.03,
    alpha: float = 0.6,
    uplift_weight: float = 0.45,
    bandit_weight: float = 0.55
) -> Dict[str, Any]:
    feedback = build_logged_bandit_feedback(
        rfm=rfm,
        n_actions=n_actions,
        random_state=random_state,
        min_pscore=min_pscore
    )

    contexts = np.asarray(feedback["context"], dtype=float)
    actions = np.asarray(feedback["action"], dtype=int)
    rewards = np.asarray(feedback["reward"], dtype=float)
    pscores = np.asarray(feedback["pscore"], dtype=float)

    model = LinearRewardModel(
        n_actions=n_actions,
        dim=contexts.shape[1],
        default_reward=float(rewards.mean())
    ).fit(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        pscores=pscores,
        reg=1.0,
        min_samples=20,
        use_ipw=True
    )

    reward_means = model.predict_values(contexts)
    baseline_action = int(np.argmax(np.bincount(actions, minlength=n_actions)))

    contextual_scores = contextual_score_matrix(contexts, model, alpha=alpha)
    uplift_scores = uplift_score_matrix(contexts, model, baseline_action=baseline_action)
    fusion_scores = fusion_score_matrix(
        contextual_scores=contextual_scores,
        uplift_scores=uplift_scores,
        reward_means=reward_means,
        uplift_weight=uplift_weight,
        bandit_weight=bandit_weight
    )

    policy_actions = {
        "contextual_bandit": np.argmax(contextual_scores, axis=1).astype(int),
        "uplift": np.argmax(uplift_scores, axis=1).astype(int),
        "fusion": np.argmax(fusion_scores, axis=1).astype(int),
    }

    score_matrices = {
        "contextual_bandit": contextual_scores,
        "uplift": uplift_scores,
        "fusion": fusion_scores,
    }

    ope_reports = {}
    for policy_name, target_actions in policy_actions.items():
        ope_reports[policy_name] = estimate_ope(
            logged_actions=actions,
            rewards=rewards,
            pscores=pscores,
            target_actions=target_actions,
            q_hat=reward_means,
            policy_name=policy_name
        )

    return {
        "feedback": feedback,
        "model": model,
        "baseline_action": baseline_action,
        "policy_actions": policy_actions,
        "score_matrices": score_matrices,
        "ope_reports": ope_reports,
    }


def get_policy_decision(suite: Dict[str, Any], policy_name: str, round_id: int) -> Dict[str, Any]:
    policy = normalize_policy_name(policy_name)
    n_rounds = int(suite["feedback"]["n_rounds"])
    idx = int(round_id) % n_rounds
    selected_action = int(suite["policy_actions"][policy][idx])
    scores = suite["score_matrices"][policy][idx].astype(float).tolist()
    return {
        "policy": policy,
        "round_id": idx,
        "selected_action": selected_action,
        "scores": scores,
    }


def get_ope_report(suite: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    policy = normalize_policy_name(policy_name)
    return dict(suite["ope_reports"][policy])
