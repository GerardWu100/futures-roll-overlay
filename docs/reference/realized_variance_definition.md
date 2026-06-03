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

In this project:

- default `A = 252`
- `H` is configured in `config.toml` under `[research].target_horizon_days`
- model target column is `target_forward_rv_annualized`
