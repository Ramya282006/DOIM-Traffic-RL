# DOIM-Traffic-RL: Formal Algorithm Pseudocode

**Target Publication:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Exact Tensor Dimensions, Operator Mapping, and Algorithmic Flow

---

## Algorithm 1: DOIM-Traffic-RL Online Control Loop with Active Safety Shield

```
Input:
    Network graph G = (V, E) with adjacency A ∈ {0, 1}^(|V|×|V|)
    Controllable traffic lights TLS = {A1, B0, B1, B2, C1} (|TLS| = 5)
    Trained policy parameters θ, value parameters ϕ, SCOIL parameters Ω
    Safety shield parameters: g_min = 10s, g_max = 60s, y_clear = 3s
    Rolling history buffer length L = 10
    Authoritative objective normalizer bounds [L_k, U_k] for k ∈ {0, ..., 4}

Initialize:
    TraCI simulation connection to Eclipse SUMO
    Spatial history deque H_seq = deque(maxlen=L)
    Previous preference weight anchor w_prev = [0.2, 0.2, 0.2, 0.2, 0.2]^T
    Reset safety shield states: time_in_phase[j] = 0, current_phase[j] = 0 for j ∈ TLS

For each control step t = 1, 2, ..., T do:
    1. Extract microscopic state:
       For each intersection i ∈ V:
           x_{i,t} = [q_{i,t}, w_{i,t}, v_{i,t}, ρ_{i,t}, p_{i,t}]^T ∈ ℝ^5
       X_t = [x_{1,t}, ..., x_{|V|,t}]^T ∈ ℝ^(|V|×5)

    2. Spatial Graph Convolution:
       H_t^(1) = ReLU( D̃^(-1/2) Ã D̃^(-1/2) X_t W_gcn^(1) )
       H_t^(2) = ReLU( D̃^(-1/2) Ã D̃^(-1/2) H_t^(1) W_gcn^(2) )
       h_t^spat = MeanPool(H_t^(2)) ∈ ℝ^64
       H_seq.append(h_t^spat)

    3. Temporal Recurrent Encoding:
       h_t^temp = GRU(H_seq) ∈ ℝ^64

    4. Multi-Objective Extraction & Normalization:
       Extract raw physical vector: z_{raw,t} = [z_wait, z_queue, z_co2, z_tp, z_fair]^T
       Compute urgency vector z_t ∈ [0, 1]^5 via AuthoritativeObjectiveConfig:
           z_{k,t} = clip((z_{raw,k,t} - L_k) / (U_k - L_k), 0, 1)  for k ∈ {0, 1, 2, 4}
           z_{3,t} = 1.0 - clip((z_{raw,3,t} - L_3) / (U_3 - L_3), 0, 1)  (throughput inversion)

    5. SCOIL Asymmetric Interaction Learning:
       Context: c_t = [h_t^temp ∥ z_t] ∈ ℝ^(64+5)
       Q_t = reshape(W_q c_t, [5, d_k]),  K_t = reshape(W_k c_t, [5, d_k])
       M_t^raw = (Q_t K_t^T / √d_k) + reshape(W_asym c_t, [5, 5])
       M_t = tanh(M_t^raw) ∈ [-1, 1]^(5×5), anchored with M_{t,kk} = 1.0
       Predictive dispersion: U_t = Softplus(W_u c_t),  U_{t,kk} = 0.0
       Shrinkage: w_shrink = 1 / (1 + λ_shrink U_t)
       M̃_t = w_shrink ⊙ M_t + (1 - w_shrink) ⊙ M_prior

    6. DOIM Latent Manifold & Adaptive Preference Generation:
       u_t = [h_t^temp ∥ MLP_z(z_t) ∥ MLP_m(vec(M̃_t))]
       m_t = tanh(LayerNorm(W_m u_t + b_m)) ∈ ℝ^64
       Adaptive weights: w_t = Softmax((W_p m_t + b_p) / τ) ∈ Δ^4

    7. Multi-Discrete Action Sampling:
       For each controllable junction j ∈ {1, ..., 5}:
           logits_j = W_a^(j) ReLU(W_π [m_t ∥ s_raw,t])
           Sample proposed action a_{j,t}^prop ~ Categorical(Softmax(logits_j))
       a_t^prop = [a_{1,t}^prop, ..., a_{5,t}^prop]^T ∈ {0, 1}^5

    8. TrafficSafetyShield Action Filtering:
       a_t^safe, target_phases, interventions = Shield.filter_action(a_t^prop, current_phases)
       If Shield detects unresolvable conflict / dead-end state:
           Raise SafetyFaultError; mark is_valid_run = False; abort episode
       For each j ∈ TLS:
           If target_phases[j] is not None:
               traci.trafficlight.setPhase(j, target_phases[j])

    9. Advance Simulation Step:
       traci.simulationStep()
       Compute step scalar reward: R_t = - w_t^T z_t ∈ [-1.0, 0.0]
       Record step telemetry: (a_t^prop, a_t^safe, interventions, z_{raw,t}, z_t, w_t, R_t)

End For
```

---

## Algorithm 2: IA-MOPPO Joint Policy & Transition Learning Update

```
Input:
    Rollout buffer B containing N_rollout transitions:
        {H_seq, z_t, M̃_t, s_raw,t, a_t^prop, a_t^exec, log_prob_old, R_t, z_{t+1}, w_{t-1}, same_ep_mask}
    Hyperparameters: epochs K_epoch, batch size B_size, clipping ϵ = 0.2, GAE (γ, λ)
    Loss coefficients: c_v (value), λ_aux (SCOIL/transition), λ_smooth (temporal preference),
                       c_kl (urgency alignment), c_ent (preference entropy), c_π (policy entropy)

1. Compute GAE Advantages and Discounted Returns:
   For step t = N_rollout - 1 down to 0:
       δ_t = R_t + γ V_ϕ(s_{t+1}) (1 - done_t) - V_ϕ(s_t)
       Â_t = δ_t + γ λ (1 - done_t) Â_{t+1}
   Normalize advantages: Â_t = (Â_t - mean(Â)) / (std(Â) + 1e-8)

2. Multi-Task Minibatch Optimization:
   For epoch = 1, ..., K_epoch do:
       Shuffle buffer indices and partition into minibatches of size B_size
       For each minibatch b do:
           a. Recompute forward pass with autograd:
              Spatial GCN + Temporal GRU: h_b^temp
              SCOIL forward: M̃_b, U_b = SCOIL(h_b^temp, z_b)
              DOIM manifold: m_b = DOIM(h_b^temp, z_b, M̃_b)
              Preference weights: w_b = Softmax(W_p m_b / τ)
              Actor-Critic: logits_b, V_ϕ(s_b)

           b. Clipped Surrogate Policy Loss:
              r_b(θ) = exp(log_prob(a_b^prop | logits_b) - log_prob_old)
              L_policy(θ) = - mean( min( r_b(θ) Â_b, clip(r_b(θ), 1-ϵ, 1+ϵ) Â_b ) )

           c. Value Function Loss:
              L_value(ϕ) = mean( (V_ϕ(s_b) - Returns_b)^2 )

           d. Heteroscedastic Predictive Dynamics Loss (using EXECUTED action a_b^exec):
              ẑ_{t+1} = clip( z_b + M̃_b (W_a a_b^exec), 0, 1 )
              σ_k^2 = mean_j(U_{b,kj}) + 1e-4
              L_scoil(Ω) = 0.5 * mean_k [ (ẑ_{t+1,k} - z_{t+1,k})^2 / σ_k^2 + log(σ_k^2) ]
              L_aux = MSE( Network_pred(m_b, a_b^exec), z_{t+1} )

           e. Temporal Preference Smoothness Regularization:
              L_smooth(w) = sum( same_ep_mask * ||w_b - stop_gradient(w_{b,prev})||_2^2 ) / (sum(same_ep_mask) + 1e-8)

           f. Urgency Alignment and Entropy Regularization:
              u_b = (z_b + δ) / sum(z_b + δ)
              L_kl = D_KL(w_b || u_b)
              H(w_b) = - sum( w_b * log(w_b + 1e-8) )
              H(π_θ) = sum_j Categorical_Entropy(logits_b^(j))

           g. Total Multi-Task Loss:
              L_total = L_policy + c_v L_value + λ_aux (L_aux + L_scoil)
                        + λ_smooth L_smooth + c_kl L_kl - c_ent H(w_b) - c_π H(π_θ)

           h. Backward & Optimization Step:
              Optimizer.zero_grad()
              L_total.backward()
              Clip gradients: ||g||_2 ≤ 0.5
              Optimizer.step()
       End For
   End For

Output: Updated parameters θ*, ϕ*, Ω*
```

