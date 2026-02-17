import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1

def run_u1_pure_gauge(L=16, beta=2.0, n_sweeps=2000, n_therm=500):
    """
    1+1D U(1) Pure Gauge Monte Carlo.
    Returns: dictionary of Wilson Loop expectations <W(R,T)>.
    """
    # Lattice: (L, L, 2). Last dim is direction mu (0=x, 1=t).
    # Cold start (all angles 0)
    links = np.zeros((L, L, 2))
    
    # Store measurements
    # We will measure loops R x T for R, T in [1..4]
    loop_dims = [(r, t) for r in range(1, 5) for t in range(1, 5)]
    history = {k: [] for k in loop_dims}
    
    print(f"Running MC: L={L}, Beta={beta}...")
    
    for sweep in range(n_therm + n_sweeps):
        # --- Metropolis Update ---
        for x in range(L):
            for t in range(L):
                for mu in range(2):
                    old_theta = links[x, t, mu]
                    
                    # Calculate local action (staple sum)
                    # In 2D, each link touches 2 plaquettes.
                    # We compute sum of staples 'A' such that Action ~ -beta*cos(theta + A)
                    # This is valid for standard Wilson action.
                    
                    # Simplified logic: Calculate total action of affected plaquettes before/after
                    # This is slower but less error-prone for a benchmark.
                    
                    def get_local_S(val):
                        # P1: Plaquette where link is U_mu(x) (Bottom/Left edge)
                        # P2: Plaquette where link is U_mu(x-mu) (Top/Right edge) - wait, simplified:
                        # Just update the link and sum the cos(P) of the 2 plaquettes it touches.
                        
                        s_local = 0
                        if mu == 0: # x-link
                            # Plaq (x,t)
                            p1 = val + links[(x+1)%L, t, 1] - links[x, (t+1)%L, 0] - links[x, t, 1]
                            # Plaq (x, t-1)
                            p2 = links[x, (t-1)%L, 0] + links[(x+1)%L, (t-1)%L, 1] - val - links[x, (t-1)%L, 1]
                            s_local = -beta * (np.cos(p1) + np.cos(p2))
                        else: # t-link
                            # Plaq (x,t)
                            p1 = links[x, t, 0] + links[(x+1)%L, t, 1] - links[x, (t+1)%L, 0] - val
                            # Plaq (x-1, t)
                            p2 = links[(x-1)%L, t, 0] + val - links[(x-1)%L, (t+1)%L, 0] - links[(x-1)%L, t, 1]
                            s_local = -beta * (np.cos(p1) + np.cos(p2))
                        return s_local

                    S_old = get_local_S(old_theta)
                    new_theta = old_theta + np.random.uniform(-0.5, 0.5)
                    S_new = get_local_S(new_theta)
                    
                    # Metropolis
                    if S_new < S_old or np.random.rand() < np.exp(-(S_new - S_old)):
                        links[x, t, mu] = new_theta

        # --- Measurement ---
        if sweep >= n_therm:
            for (R, T) in loop_dims:
                # Calculate W(R,T) averaged over lattice volume
                w_sum = 0
                for x in range(L):
                    for t in range(L):
                        # Wilson Loop Sum
                        # Bottom (x to x+R)
                        loop = np.sum([links[(x+i)%L, t, 0] for i in range(R)])
                        # Right (t to t+T)
                        loop += np.sum([links[(x+R)%L, (t+j)%L, 1] for j in range(T)])
                        # Top (x+R to x) - dagger
                        loop -= np.sum([links[(x+i)%L, (t+T)%L, 0] for i in range(R)])
                        # Left (t+T to t) - dagger
                        loop -= np.sum([links[x, (t+j)%L, 1] for j in range(T)])
                        w_sum += np.cos(loop)
                history[(R,T)].append(w_sum / (L*L))
                
    return {k: np.mean(v) for k, v in history.items()}

# --- Execution & Plotting ---
BETA = 3.0
results = run_u1_pure_gauge(L=16, beta=BETA)

areas = []
log_w = []

print(f"\n--- Results (Beta={BETA}) ---")
for (R, T), val in results.items():
    areas.append(R*T)
    log_w.append(-np.log(val))

# Analytical
sigma_exact = -np.log(i1(BETA)/i0(BETA))

# Plot
plt.figure(figsize=(6, 5))
plt.plot(areas, log_w, 'ro', label='MC Data')
x_line = np.linspace(0, max(areas), 10)
plt.plot(x_line, sigma_exact * x_line, 'b--', label=f'Exact $\sigma={sigma_exact:.4f}$')
plt.xlabel('Loop Area $A$')
plt.ylabel(r'$-\ln \langle W \rangle$')
plt.title(f'Figure 1: Confinement Area Law ($\\beta={BETA}$)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()