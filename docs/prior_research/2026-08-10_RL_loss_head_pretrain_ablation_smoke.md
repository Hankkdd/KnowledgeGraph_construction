# RL loss ablation：head-pretrain anchor smoke

日期：2026-08-10

## 研究問題

目前 RL 的 hindsight allocation-head pretraining 可能把 policy 錨定在接近固定配置。降低 `head_pretrain_coef` 是否能讓 RL 學到更有變化、且更有效的配置？

## 凍結設計

- no-KG control：`embedding-source=random`
- 2024→2025；fixed-5；seed=41
- pretrain epochs=5、RL episodes=10、`action_blend=0.025`
- `relative_reward_coef=1.0`、encoder 解凍
- 比較 `head_pretrain_coef ∈ {1.0, 0.1, 0.0}`

## 結果

| head-pretrain coef | total return | Sharpe | MDD | test action-weight std（約） |
|---:|---:|---:|---:|---:|
| 1.0 | 0.3241 | 1.4061 | -0.2055 | 0.0058–0.0104 |
| 0.1 | 0.3236 | 1.3960 | -0.2081 | 0.0052–0.0068 |
| 0.0 | 0.3176 | 1.3966 | -0.2057 | 0.0028–0.0048 |

降低係數確實降低了配置變化幅度，尤其 `coef=0`；但沒有帶來 Sharpe、return 或 MDD 的改善。三組差異小於單 seed smoke 可支持的穩定結論。

## 解讀與決策

`head_pretrain_coef` 會影響 policy 的活動量，但不是目前 daily RL 沒有 KG 增益的主要瓶頸。這次只使用 no-KG、單 seed、低 episodes，不能宣稱三組正式排名；但沒有足夠訊號支持進入昂貴的 5-seed tuning。

**Decision：stop this loss-anchor branch.** 不把降低 pretrain loss 當成 KG 效果的補救方案，也不再調整係數追求測試 Sharpe。

既有 `relative_reward_coef=1.0` 的歷史 ablation 也顯示幾乎沒有績效改善；因此目前 reward/loss 修改的優先級低於重新設計 dynamic/event KG 或 hierarchical slow policy。

原始 run：`0601_atten_V2/runs/reward_loss_ablation_s10/`；使用腳本：`0601_atten_V2/run_attention_fixed5.py`。
