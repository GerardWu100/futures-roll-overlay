# Realized Variance Definition

Let $r_t$ be the daily log return at date $t$.

Daily realized variance:

$$
rv_t = r_t^2
$$

Forward $H$-day realized variance:

$$
RV_{t,t+H} = \sum_{i=1}^{H} r_{t+i}^2
$$

Annualized forward target with annualization factor $A$:

$$
RV^{ann}_{t,t+H} = \frac{A}{H} RV_{t,t+H}
$$

The persistence benchmark uses only variance observable after the close at
date $t$:

$$
K_t = \frac{A}{H}\sum_{i=0}^{H-1}r_{t-i}^2
$$

The implementation builds the forward target from explicit leads $1$ through
$H$. A shifted rolling window is invalid here because a standard rolling sum
looks backward and would include the return dated $t$.

In this project:

- default `A = 252`
- `H` is configured in `config.toml` under `[research].target_horizon_days`
- model target column is `target_forward_rv_annualized`
- feasible persistence input is `known_trailing_rv_annualized`
