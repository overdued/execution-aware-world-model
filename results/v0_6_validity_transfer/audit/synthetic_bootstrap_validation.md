# Synthetic bootstrap 验证

```json
{
 "n_rep": 50,
 "true_sd_of_mean": 0.22360679774997896,
 "naive_window_boot": {
  "coverage@95": 0.08,
  "mean_width": 0.05839114196703582
 },
 "episode_boot": {
  "coverage@95": 0.94,
  "mean_width": 0.8511201523035585
 }
}
```
解读：20 个 episode、每 episode 随机效应 sd=1 时，窗级(naive) bootstrap 把 4000 个相关窗口当独立样本，CI 宽度 ~0.05 且覆盖率远低于 95%；episode 级 bootstrap 宽度 ~0.9 接近真值 sd(mean)=0.22 的 4σ 区间，覆盖率正常。→ 同 episode 重复窗口不能凭空提高独立证据数量。