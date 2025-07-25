# Falsification-Driven Exploration: A Proactive Approach to Reinforcement Learning

## 1. Core Idea & Motivation

The central challenge in reinforcement learning, particularly in sparse-reward or deceptive environments, is efficient exploration. Many contemporary methods incentivize exploration by rewarding an agent for "curiosity," typically defined as the error in its predictive model of the world (its "world model"). While effective, this approach is **reactive**; the agent is rewarded for mistakes it has already made.

This research project investigates a **proactive** exploration strategy inspired by Karl Popper's philosophy of science. Popper argued that science progresses not by verifying theories but by attempting to **falsify** them. A robust scientific theory is one that is highly falsifiable yet has withstood rigorous attempts at refutation.

We translate this principle into a novel intrinsic reward mechanism:
> Instead of rewarding an agent for passively encountering states where its world model is wrong, we reward it for actively designing and executing "experiments" (sequences of actions) that are intended to maximally falsify its own world model.

This reframes exploration from a random walk towards novelty into a goal-directed, adversarial search for the agent's own epistemic weaknesses.

## 2. Implemented Technical Mechanism

The system is built on a PPO backbone and consists of several interacting components that follow a "scientific method" loop. The implementation details below reflect the current state of the codebase.

### 2.1 System Components

1.  **Shared Feature Extractor ($\phi_{E}$):** A `MiniGridCNN` (from `falsify.models`) that processes raw image observations $s_t$ into a compact feature vector $\mathbf{z}_t = \phi_{E}(s_t)$. This extractor is shared across all other components, creating a strong inductive bias that the learned features must be useful for both prediction and control.

2.  **Policy & Value Network ($\pi_{\theta}$):** A standard actor-critic network (`PolicyValueNet` from `falsify.models`) that takes state features $\mathbf{z}_t$ as input and outputs an action distribution and a value estimate. It is trained via the PPO objective.

3.  **The "Theory" Model ($M_{\psi}$):** A predictive forward dynamics model, implemented as `TheoryModel` and managed by `TheoryModule` (from `falsify.components`).
    *   **Input:** The current state feature $\mathbf{z}_t$ and an action $a_t$.
    *   **Output:** A prediction of the next state's feature vector, $\hat{\mathbf{z}}_{t+1}$.
    *   **Objective:** To minimize the Mean Squared Error between its prediction and the actual next state feature vector produced by the frozen feature extractor. This is the agent's "scientific theory."
    *   **Loss Function:** $L_{Theory} = || \phi_{E}(s_{t+1}) - M_{\psi}(\mathbf{z}_t, a_t) ||^2$

4.  **The "Falsifier" Model ($F_{\omega}$):** An adversarial, recurrent model whose goal is to find flaws in the Theory Model. It is implemented as `FalsifierModel` and managed by `FalsifierModule` (from `falsify.components`).
    *   **Input:** A sequence of state features and actions over a future horizon $H$, collected from the agent's past experience: $(\langle \mathbf{z}_t, a_t \rangle, \langle \mathbf{z}_{t+1}, a_{t+1} \rangle, ..., \langle \mathbf{z}_{t+H-1}, a_{t+H-1} \rangle)$. A Gated Recurrent Unit (GRU) processes this sequence.
    *   **Output:** A single scalar value, the "falsification score," which estimates the *cumulative* future error of the Theory Model over that trajectory.
    *   **Objective:** To accurately predict the true, summed error of the Theory Model over the horizon.
    *   **Loss Function:** $L_{Falsifier} = || F_{\omega}(\text{sequence}) - \sum_{i=0}^{H-1} \text{target\_error}_{t+i} ||^2$. Crucially, the target values are **normalized** (z-scored) within each batch to prevent model collapse and force the Falsifier to learn *relative*, not absolute, error.

### 2.2 The Training Loop: A Two-Optimizer Approach

The `Trainer` and `FalsificationAgent` execute the following steps, which differ from the initial proposal by using two separate optimizers for stability and modularity.

1.  **Data Collection (Rollout):** The agent uses its current policy $\pi_{\theta}$ to interact with the environment for `num_steps`, collecting a buffer of experiences $(s_t, a_t, r_t^{ext}, d_t)$ into `RolloutStorage`.
2.  **Intrinsic Reward Calculation:** The `FalsifierModule` analyzes sequences from the rollout buffer. The output "falsification score" from $F_{\omega}$ for each starting step becomes the intrinsic reward $r_t^{int}$.
3.  **Reward Combination:** The total reward is a weighted sum: $r_t = r_t^{ext} + \beta \cdot r_t^{int}$. The intrinsic reward is normalized and clipped before being added to prevent it from destabilizing the policy.
4.  **Advantage & Returns Calculation:** Generalized Advantage Estimation (GAE) is computed using the combined total rewards.
5.  **Policy and Model Updates (Two Separate Steps):**
    *   **A) PPO Update:** The `policy_optimizer` updates the parameters of the **Feature Extractor ($\phi_E$)** and the **Policy/Value Network ($\pi_{\theta}$)**. Its objective is to minimize the PPO loss ($L_{PPO}$), which includes policy gradient, value function, and entropy terms calculated from the combined rewards.
    *   **B) Auxiliary Update:** The `aux_optimizer` updates the parameters of the **Theory Model ($M_{\psi}$)** and the **Falsifier Model ($F_{\omega}$)**. Its objective is to minimize a combined auxiliary loss: $L_{Aux} = c_1 \cdot L_{Theory} + c_2 \cdot L_{Falsifier}$. This separation ensures that the learning of the intrinsic motivation modules does not directly interfere with the policy's optimization step. The Feature Extractor's weights are used for calculating these losses but are only updated by the `policy_optimizer`.

## 3. Key Research Questions

This project aims to answer the following questions:

1.  **Sample Efficiency:** Does goal-directed falsification lead to more sample-efficient exploration and faster convergence than undirected curiosity (ICM) and standard PPO, especially in environments with sparse rewards and hierarchical task structures?
2.  **Emergence of Complex Experiments:** Does the falsification objective lead to the emergence of complex, multi-step behaviors that are purely information-seeking? For instance, will an agent learn to use a key on a door not for an extrinsic reward, but purely to test its theory of what lies behind it?
3.  **Interpretable Uncertainty:** Can we analyze the Falsifier's proposals (i.e., the trajectories it assigns high scores to) to gain a more structured and interpretable understanding of the agent's epistemic uncertainty compared to raw prediction error?

## 4. Implemented Challenges & Mitigations

*   **Challenge:** The Falsifier becomes "lazy" and always predicts a low error as the Theory model improves.
    *   **Mitigation (Implemented):** The Falsifier's training targets (the actual summed errors) are normalized per-batch (`_prepare_falsifier_targets` in `falsification_agent.py`). This forces it to predict *relative* error.

*   **Challenge:** The agent learns to "game" the system by keeping its Theory Model predictably wrong.
    *   **Mitigation (Implemented):** The policy network uses the features from the shared feature extractor $\phi_E$. If these features become nonsensical to make the theory model easy to falsify, the policy itself will fail to solve the extrinsic task, creating a powerful counter-incentive.

*   **Challenge:** Unstable adversarial dynamics between the Theory and Falsifier models.
    *   **Mitigation (Implemented):** The Falsifier's targets are only prepared and cached periodically, controlled by `falsify_update_freq`. This gives the Theory Model time to learn between Falsifier updates. Additionally, intrinsic rewards are clipped in the `Trainer` to bound their magnitude.

## 5. Future Work & Extensions

Beyond this initial implementation, several exciting research avenues exist. The following steps outline a path from the current on-policy agent to a more sophisticated, off-policy planning agent, drawing inspiration from recent advances in the field.

### 5.1 From On-Policy Analysis to Off-Policy Planning

The current implementation is on-policy: the Falsifier retroactively scores sequences that the policy has already executed. The most significant next step is to make this process proactive by moving to an off-policy, planning-based framework. This transforms the agent from a "historian" analyzing the past into a "scientist" actively designing future experiments.

This would involve a major architectural overhaul:

*   **Adopt an Off-Policy Algorithm:** Replace the PPO backbone with a more sample-efficient off-policy algorithm like **Soft Actor-Critic (SAC)**. This requires implementing actor, critic, and target networks.
*   **Introduce a Replay Buffer:** Store all `(s, a, r, s', d)` transitions in a large replay buffer, allowing the agent to learn from a diverse history of experiences.
*   **Implement a Trajectory Planner:** This is the core of the "scientist" agent. A planner like the **Cross-Entropy Method (CEM)** would be used to:
    1.  Sample `N` candidate action sequences from the current state.
    2.  Use the `TheoryModel` to perform "imaginary rollouts" for each sequence.
    3.  Score each imagined trajectory using the `FalsifierModel`.
    4.  Iteratively refine the action distribution towards the sequences that yield the highest falsification scores.
*   **Goal-Conditioned Execution:** The best plan generated by the CEM becomes a temporary goal. The agent receives a large intrinsic reward for successfully executing this self-generated experiment, which is then used to train the SAC agent.

### 5.2 Hybrid Value Functions for Exploration Stability

A key challenge with any intrinsic motivation method is that large or noisy intrinsic rewards can destabilize the policy, causing it to ignore the extrinsic task. Recent work on "Strangeness-Driven Exploration" (Kim et al., 2024) proposes an elegant solution that could be directly adapted to our Falsification agent.

The idea is to maintain two separate action-value functions:

*   **Goal Action-Value Function `Q_goal(s, a)`:** This network is trained **only on extrinsic rewards** (`r_ext`). It learns a clean, unbiased value function for solving the actual task.
*   **Exploration Action-Value Function `Q_exp(s, a)`:** This network is used to generate behavior during training. It is trained on a **mixed reward** (`r_ext + β * r_int`), allowing it to be driven by the Falsifier's score.

By decoupling the goal-oriented value function from the exploratory one, the agent can explore aggressively without corrupting its knowledge of how to solve the underlying problem. This would likely allow for a higher `intrinsic_coef` and more robust training.
