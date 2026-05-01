from typing import Dict, Any, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config
from src.utils.bandit_suite import (
    normalize_policy_name,
    get_policy_decision,
    get_ope_report as get_base_ope_report,
    estimate_ope,
)

EPS = 1e-8
POLICY_ORDER = ["uplift", "contextual_bandit", "fusion", "rl_policy", "rule_engine"]


class RouterAgent:
    def __init__(
        self,
        bandit_suite: Dict[str, Any],
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        pscores: np.ndarray,
    ):
        self.bandit_suite = bandit_suite
        self.contexts = np.asarray(contexts, dtype=float)
        self.actions = np.asarray(actions, dtype=int)
        self.rewards = np.asarray(rewards, dtype=float)
        self.pscores = np.asarray(pscores, dtype=float)
        self.n_rounds = int(self.contexts.shape[0])
        self.n_actions = int(np.max(self.actions)) + 1 if len(self.actions) > 0 else 5

        self.low_pscore_threshold = float(getattr(config, "ROUTER_LOW_PSCORE", 0.08))
        self.high_complexity_threshold = float(getattr(config, "ROUTER_HIGH_COMPLEXITY", 0.95))
        self.churn_recency_threshold = float(getattr(config, "ROUTER_CHURN_RECENCY", 90.0))
        self.force_rule_on_missing_context = bool(
            getattr(config, "ROUTER_FORCE_RULE_ENGINE_ON_MISSING_CONTEXT", True)
        )

        self.router_model_min_confidence = float(getattr(config, "ROUTER_MODEL_MIN_CONFIDENCE", 0.45))
        self.router_force_rule_pscore = float(getattr(config, "ROUTER_FORCE_RULE_ENGINE_PSCORE", 0.02))
        self.router_model_random_seed = int(getattr(config, "ROUTER_MODEL_RANDOM_SEED", 42))
        self.router_model_min_samples = int(getattr(config, "ROUTER_MODEL_MIN_SAMPLES", 200))

        self.ope_feedback_enabled = bool(getattr(config, "ROUTER_OPE_FEEDBACK_ENABLED", True))
        self.ope_feedback_lr = float(getattr(config, "ROUTER_OPE_FEEDBACK_LR", 0.15))
        self.ope_feedback_decay = float(getattr(config, "ROUTER_OPE_FEEDBACK_DECAY", 0.02))
        self.ope_feedback_clip = float(getattr(config, "ROUTER_OPE_FEEDBACK_CLIP", 0.25))
        metric = str(getattr(config, "ROUTER_OPE_FEEDBACK_METRIC", "DR")).strip().upper()
        self.ope_feedback_metric = metric if metric in {"IPS", "SNIPS", "DR"} else "DR"
        self.policy_temperature = max(0.1, float(getattr(config, "ROUTER_POLICY_TEMPERATURE", 1.0)))

        self.policy_bias: Dict[str, float] = {p: 0.0 for p in POLICY_ORDER}
        self.policy_feedback_stats: Dict[str, Dict[str, Any]] = {
            p: {"count": 0, "ema_metric": 0.0, "ema_advantage": 0.0} for p in POLICY_ORDER
        }

        self._extra_ope_reports: Dict[str, Dict[str, Any]] = {}
        self._dataset_policy_actions: Dict[str, np.ndarray] = {}

        self.router_model = None
        self.router_model_ready = False
        self.router_train_report: Dict[str, Any] = {}

        self._prepare_extra_ope_reports()
        self._train_router_model()

    @staticmethod
    def _normalize_router_policy(name: str) -> str:
        key = (name or "").strip().lower()
        if key in {"auto"}:
            return "auto"
        if key in {"rule", "rules", "rule_engine"}:
            return "rule_engine"
        if key in {"rl", "rl_policy"}:
            return "rl_policy"
        if key in {"uplift", "fusion", "hybrid", "uplift_bandit", "bandit", "contextual", "contextual_bandit"}:
            return normalize_policy_name(key)
        return "rule_engine"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)

    @staticmethod
    def _normalize_vector(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max - x_min < EPS:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def _to_prob_distribution(self, scores: Dict[str, float]) -> Dict[str, float]:
        vals = np.asarray([max(0.0, float(scores.get(p, 0.0))) for p in POLICY_ORDER], dtype=float)
        total = float(vals.sum())
        if total <= EPS:
            vals = np.full(len(POLICY_ORDER), 1.0 / len(POLICY_ORDER), dtype=float)
        else:
            vals = vals / total
        return {p: float(vals[i]) for i, p in enumerate(POLICY_ORDER)}

    def _apply_policy_bias(self, scores: Dict[str, float]) -> Dict[str, float]:
        probs = self._to_prob_distribution(scores)
        base = np.asarray([probs[p] for p in POLICY_ORDER], dtype=float)
        bias = np.asarray([float(self.policy_bias.get(p, 0.0)) for p in POLICY_ORDER], dtype=float)

        logits = np.log(np.clip(base, EPS, None)) + bias
        logits = logits / self.policy_temperature
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        soft = exp / np.clip(exp.sum(), EPS, None)
        return {p: float(soft[i]) for i, p in enumerate(POLICY_ORDER)}

    def get_policy_bias(self) -> Dict[str, float]:
        return {p: round(float(v), 6) for p, v in self.policy_bias.items()}

    def _get_round_index(self, round_id: int) -> int:
        if self.n_rounds <= 0:
            return 0
        return int(round_id) % self.n_rounds

    def _get_context(self, round_id: int, user_profile: Dict[str, Any]) -> np.ndarray:
        profile_context = user_profile.get("obd_context", [])
        if isinstance(profile_context, list) and len(profile_context) > 0:
            try:
                x = np.asarray(profile_context, dtype=float)
                if x.ndim == 1 and x.size == self.contexts.shape[1]:
                    return x
            except Exception:
                pass
        idx = self._get_round_index(round_id)
        return self.contexts[idx]

    def _compute_churn_risk(self, user_profile: Dict[str, Any], context: np.ndarray) -> float:
        recency = self._safe_float(user_profile.get("recency"), default=30.0)
        if recency > 0:
            return float(np.clip((recency - 30.0) / max(1.0, self.churn_recency_threshold), 0.0, 1.0))
        if context.size > 0:
            return float(np.clip((context[0] + 1.5) / 3.0, 0.0, 1.0))
        return 0.2

    def _rule_engine_scores(self, user_profile: Dict[str, Any], context: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.n_actions, dtype=float)
        segment = str(user_profile.get("user_segment", ""))
        segment_rules = {
            "高价值忠诚用户": 1,
            "潜力增长用户": 0,
            "新用户": 2,
            "一般价值用户": 3,
            "流失风险用户": 4,
        }

        if segment in segment_rules:
            action = segment_rules[segment] % self.n_actions
        else:
            recency_signal = float(context[0]) if context.size > 0 else 0.0
            if recency_signal > 0.8:
                action = min(self.n_actions - 1, 4)
            elif recency_signal > 0.2:
                action = min(self.n_actions - 1, 3)
            elif recency_signal < -0.8:
                action = min(self.n_actions - 1, 1)
            else:
                action = 0

        scores[action] = 1.0
        return scores

    def _rl_scores(self, round_id: int, user_profile: Dict[str, Any]) -> np.ndarray:
        idx = self._get_round_index(round_id)
        contextual = np.asarray(self.bandit_suite["score_matrices"]["contextual_bandit"][idx], dtype=float)
        uplift = np.asarray(self.bandit_suite["score_matrices"]["uplift"][idx], dtype=float)
        context = self._get_context(round_id, user_profile)
        churn_risk = self._compute_churn_risk(user_profile, context)

        short_term = 0.6 * self._normalize_vector(contextual) + 0.4 * self._normalize_vector(uplift)
        retention_pref = self._normalize_vector(np.linspace(0.0, 1.0, self.n_actions))

        recency_bonus = np.zeros(self.n_actions, dtype=float)
        if context.size > 0:
            recency_factor = float(np.clip((context[0] + 1.0) / 2.0, 0.0, 1.0))
            recency_bonus = recency_factor * retention_pref

        long_term = 0.7 * retention_pref + 0.3 * recency_bonus
        return (1.0 - churn_risk) * short_term + churn_risk * long_term

    def _build_heuristic_router_scores(
        self,
        has_context: bool,
        complexity: float,
        logged_pscore: float,
        churn_risk: float,
    ) -> Dict[str, float]:
        low_pscore_signal = float(
            np.clip(
                (self.low_pscore_threshold - logged_pscore) / max(self.low_pscore_threshold, EPS),
                0.0,
                1.0,
            )
        )
        complexity_signal = float(np.clip(complexity / max(self.high_complexity_threshold, EPS), 0.0, 1.5))
        uplift_friendly = float(np.clip(1.0 - complexity_signal, 0.0, 1.0))

        return {
            "contextual_bandit": 0.6 * complexity_signal + 0.2 * (1.0 - churn_risk) + 0.2 * max(logged_pscore, 0.0),
            "uplift": 0.55 * uplift_friendly + 0.25 * (1.0 - low_pscore_signal) + 0.2 * (1.0 - churn_risk),
            "fusion": 0.6 * low_pscore_signal + 0.25 * complexity_signal + 0.15 * churn_risk,
            "rl_policy": 0.7 * churn_risk + 0.2 * complexity_signal + 0.1 * low_pscore_signal,
            "rule_engine": 0.2 if has_context else 0.9,
        }

    def _build_dataset_policy_actions(self) -> Dict[str, np.ndarray]:
        if self._dataset_policy_actions:
            return self._dataset_policy_actions

        policy_actions = {
            "contextual_bandit": np.asarray(self.bandit_suite["policy_actions"]["contextual_bandit"], dtype=int),
            "uplift": np.asarray(self.bandit_suite["policy_actions"]["uplift"], dtype=int),
            "fusion": np.asarray(self.bandit_suite["policy_actions"]["fusion"], dtype=int),
        }

        rl_actions = []
        rule_actions = []
        for idx in range(self.n_rounds):
            profile = {"obd_context": self.contexts[idx].astype(float).tolist()}
            rl_actions.append(int(np.argmax(self._rl_scores(idx, profile))))
            rule_actions.append(int(np.argmax(self._rule_engine_scores(profile, self.contexts[idx]))))

        policy_actions["rl_policy"] = np.asarray(rl_actions, dtype=int)
        policy_actions["rule_engine"] = np.asarray(rule_actions, dtype=int)

        self._dataset_policy_actions = policy_actions
        return self._dataset_policy_actions

    def _train_router_model(self):
        if self.n_rounds < max(50, self.router_model_min_samples):
            self.router_train_report = {
                "enabled": False,
                "reason": f"insufficient_samples({self.n_rounds})",
            }
            return

        q_hat = self.bandit_suite["model"].predict_values(self.contexts)
        actions_by_policy = self._build_dataset_policy_actions()
        n = self.n_rounds
        k = len(POLICY_ORDER)
        idx = np.arange(n)

        utility = np.zeros((n, k), dtype=float)
        for j, policy in enumerate(POLICY_ORDER):
            policy_actions = actions_by_policy[policy]
            utility[:, j] = q_hat[idx, policy_actions]

        complexity = np.mean(np.abs(self.contexts), axis=1)
        complexity_signal = np.clip(complexity / max(self.high_complexity_threshold, EPS), 0.0, 1.5)
        low_pscore_signal = np.clip(
            (self.low_pscore_threshold - self.pscores) / max(self.low_pscore_threshold, EPS),
            0.0,
            1.0,
        )
        churn_proxy = np.clip((self.contexts[:, 0] + 1.5) / 3.0, 0.0, 1.0)

        utility[:, POLICY_ORDER.index("fusion")] += 0.03 * low_pscore_signal
        utility[:, POLICY_ORDER.index("contextual_bandit")] += 0.02 * complexity_signal
        utility[:, POLICY_ORDER.index("rl_policy")] += 0.03 * churn_proxy
        utility[:, POLICY_ORDER.index("rule_engine")] -= 0.01

        y = np.argmax(utility, axis=1).astype(int)
        X = self.contexts.astype(float)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            self.router_train_report = {
                "enabled": False,
                "reason": "single_class_labels",
                "label_distribution": {POLICY_ORDER[int(c)]: int(nc) for c, nc in zip(classes, counts)},
            }
            return

        stratify = y if int(np.min(counts)) >= 2 else None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.router_model_random_seed,
            stratify=stratify,
        )

        model = RandomForestClassifier(
            n_estimators=240,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=self.router_model_random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, pred))

        self.router_model = model
        self.router_model_ready = True
        self.router_train_report = {
            "enabled": True,
            "n_samples": int(n),
            "val_accuracy": round(acc, 6),
            "label_distribution": {POLICY_ORDER[int(c)]: int(nc) for c, nc in zip(classes, counts)},
        }

    def _predict_policy_probs(self, context: np.ndarray) -> Dict[str, float]:
        scores = {p: 0.0 for p in POLICY_ORDER}
        if not self.router_model_ready or self.router_model is None:
            return scores

        x = np.asarray(context, dtype=float).reshape(1, -1)
        prob = self.router_model.predict_proba(x)[0]
        classes = self.router_model.classes_

        for cls_id, p in zip(classes, prob):
            idx = int(cls_id)
            if 0 <= idx < len(POLICY_ORDER):
                scores[POLICY_ORDER[idx]] = float(p)

        return self._to_prob_distribution(scores)

    def apply_ope_feedback(self, policy_name: str, ope_report: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ope_feedback_enabled:
            return {"enabled": False, "reason": "disabled"}

        policy = self._normalize_router_policy(policy_name)
        if policy not in POLICY_ORDER:
            return {"enabled": False, "reason": "invalid_policy"}

        metric_key_map = {
            "IPS": "IPS策略价值估计",
            "SNIPS": "SNIPS策略价值估计",
            "DR": "DR策略价值估计",
        }
        metric_key = metric_key_map[self.ope_feedback_metric]
        metric_value = self._safe_float(ope_report.get(metric_key), 0.0)
        baseline_value = self._safe_float(ope_report.get("行为策略平均回报"), 0.0)

        advantage = metric_value - baseline_value
        clipped_advantage = float(np.clip(advantage, -self.ope_feedback_clip, self.ope_feedback_clip))

        decay = float(np.clip(self.ope_feedback_decay, 0.0, 0.99))
        for p in POLICY_ORDER:
            self.policy_bias[p] = float(self.policy_bias[p] * (1.0 - decay))

        old_bias = float(self.policy_bias[policy])
        self.policy_bias[policy] = float(old_bias + self.ope_feedback_lr * clipped_advantage)

        bias_values = np.asarray([self.policy_bias[p] for p in POLICY_ORDER], dtype=float)
        bias_center = float(np.mean(bias_values))
        for p in POLICY_ORDER:
            self.policy_bias[p] = float(np.clip(self.policy_bias[p] - bias_center, -1.5, 1.5))

        stat = self.policy_feedback_stats[policy]
        stat["count"] = int(stat["count"]) + 1
        ema_beta = 0.2
        stat["ema_metric"] = float((1 - ema_beta) * float(stat["ema_metric"]) + ema_beta * metric_value)
        stat["ema_advantage"] = float((1 - ema_beta) * float(stat["ema_advantage"]) + ema_beta * advantage)

        return {
            "enabled": True,
            "policy": policy,
            "metric": self.ope_feedback_metric,
            "metric_value": round(metric_value, 6),
            "baseline": round(baseline_value, 6),
            "advantage": round(advantage, 6),
            "clipped_advantage": round(clipped_advantage, 6),
            "old_bias": round(old_bias, 6),
            "new_bias": round(float(self.policy_bias[policy]), 6),
            "bias_vector": self.get_policy_bias(),
            "feedback_count": int(stat["count"]),
        }

    def decide(self, routing_mode: str, user_profile: Dict[str, Any], round_id: int) -> Dict[str, Any]:
        mode = self._normalize_router_policy(routing_mode)
        context = self._get_context(round_id, user_profile)
        has_context = bool(context.size > 0)

        complexity = float(np.mean(np.abs(context))) if has_context else 0.0
        logged_pscore = self._safe_float(user_profile.get("obd_pscore"), default=1.0)
        churn_risk = self._compute_churn_risk(user_profile, context)

        reason_codes: List[str] = []

        if mode != "auto":
            selected_policy = mode
            confidence = 1.0
            candidate_scores = {p: (1.0 if p == selected_policy else 0.0) for p in POLICY_ORDER}
            reason_codes.append("manual_override")
        else:
            if (not has_context) and self.force_rule_on_missing_context:
                selected_policy = "rule_engine"
                confidence = 1.0
                candidate_scores = {p: (1.0 if p == "rule_engine" else 0.0) for p in POLICY_ORDER}
                reason_codes.append("missing_context_force_rule_engine")
            else:
                if self.router_model_ready:
                    raw_scores = self._predict_policy_probs(context)
                    reason_codes.append("auto_learned_router")
                else:
                    raw_scores = self._build_heuristic_router_scores(
                        has_context=has_context,
                        complexity=complexity,
                        logged_pscore=logged_pscore,
                        churn_risk=churn_risk,
                    )
                    reason_codes.append("auto_heuristic_router")

                candidate_scores = self._apply_policy_bias(raw_scores)
                selected_policy = max(candidate_scores, key=candidate_scores.get)
                confidence = float(candidate_scores[selected_policy])

                if logged_pscore < self.router_force_rule_pscore:
                    selected_policy = "rule_engine"
                    reason_codes.append("force_rule_by_very_low_pscore")
                elif self.router_model_ready and confidence < self.router_model_min_confidence:
                    selected_policy = "rule_engine"
                    reason_codes.append("low_router_confidence_fallback_rule")

        if selected_policy not in POLICY_ORDER:
            selected_policy = "rule_engine"
            reason_codes.append("invalid_policy_fallback_rule")

        if logged_pscore < self.low_pscore_threshold:
            reason_codes.append("low_pscore")
        if complexity >= self.high_complexity_threshold:
            reason_codes.append("high_complexity")
        if churn_risk >= 0.7:
            reason_codes.append("high_churn_risk")

        confidence = float(np.clip(confidence, 0.0, 1.0))
        return {
            "policy": selected_policy,
            "mode": mode,
            "router_type": "learned" if self.router_model_ready else "heuristic",
            "model_ready": bool(self.router_model_ready),
            "model_report": self.router_train_report,
            "confidence": round(confidence, 6),
            "complexity": round(complexity, 6),
            "logged_pscore": round(logged_pscore, 6),
            "churn_risk": round(churn_risk, 6),
            "temperature": round(self.policy_temperature, 6),
            "feedback_metric": self.ope_feedback_metric,
            "policy_bias": self.get_policy_bias(),
            "reason_codes": reason_codes,
            "candidate_scores": {k: round(float(v), 6) for k, v in candidate_scores.items()},
        }

    def act(self, policy_name: str, round_id: int, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        policy = self._normalize_router_policy(policy_name)
        idx = self._get_round_index(round_id)

        if policy in {"uplift", "contextual_bandit", "fusion"}:
            decision = get_policy_decision(self.bandit_suite, policy, idx)
            scores = np.asarray(decision["scores"], dtype=float)
            selected_action = int(decision["selected_action"])
            method = decision["policy"]
        elif policy == "rl_policy":
            scores = self._rl_scores(idx, user_profile)
            selected_action = int(np.argmax(scores))
            method = "rl_policy"
        else:
            context = self._get_context(idx, user_profile)
            scores = self._rule_engine_scores(user_profile, context)
            selected_action = int(np.argmax(scores))
            method = "rule_engine"

        top_actions = np.argsort(scores)[::-1][:3].astype(int).tolist()

        return {
            "policy": method,
            "round_id": idx,
            "selected_action": selected_action,
            "scores": scores.astype(float).tolist(),
            "top_actions": top_actions,
            "policy_score": round(float(scores[selected_action]), 6),
        }

    def _prepare_extra_ope_reports(self):
        if self.n_rounds == 0:
            return

        q_hat = self.bandit_suite["model"].predict_values(self.contexts)
        policy_actions = self._build_dataset_policy_actions()

        self._extra_ope_reports["rl_policy"] = estimate_ope(
            logged_actions=self.actions,
            rewards=self.rewards,
            pscores=self.pscores,
            target_actions=policy_actions["rl_policy"],
            q_hat=q_hat,
            policy_name="rl_policy",
        )
        self._extra_ope_reports["rule_engine"] = estimate_ope(
            logged_actions=self.actions,
            rewards=self.rewards,
            pscores=self.pscores,
            target_actions=policy_actions["rule_engine"],
            q_hat=q_hat,
            policy_name="rule_engine",
        )

    def get_ope_report(self, policy_name: str) -> Dict[str, Any]:
        policy = self._normalize_router_policy(policy_name)
        if policy in {"uplift", "contextual_bandit", "fusion"}:
            return get_base_ope_report(self.bandit_suite, policy)
        if policy in self._extra_ope_reports:
            return dict(self._extra_ope_reports[policy])
        return get_base_ope_report(self.bandit_suite, "contextual_bandit")