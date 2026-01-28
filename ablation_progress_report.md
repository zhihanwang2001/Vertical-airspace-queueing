# Ablation Experiment Progress Report

**Time**: $(date '+%Y-%m-%d %H:%M:%S')

## Current Status

### Experiment 1/12: HCA2C-Full seed=42

**Progress**: 
- Updates: 9700/15625 (62.1%)
- Timesteps: 310,400/500,000 (62.1%)
- Mean Reward: ~227,000 (stable, matching expected performance)
- Training Speed: ~560 FPS

**Performance**:
- Process: Running (PID 74054)
- CPU Usage: 109.3% (multi-core)
- Memory: 61,936 KB (0.4%)

**Estimated Completion**:
- Remaining timesteps: 189,600
- Time remaining: ~5-6 minutes
- Expected completion: ~10:29 AM

## Overall Progress

- **Completed**: 0/12 runs
- **In Progress**: Run 1/12 (HCA2C-Full seed=42) - 62% complete
- **Remaining**: 11 runs

## Key Observations

1. ✅ Training is stable with consistent reward ~227K
2. ✅ Performance matches expected HCA2C-Full baseline
3. ✅ Training speed is good (~560 FPS)
4. ✅ No crashes or errors detected

## Next Steps

1. **Immediate** (5-6 minutes): First run will complete
2. **Next 2.5 hours**: Runs 2-3 (HCA2C-Full seeds 43, 44)
3. **Following runs**: HCA2C-Flat, HCA2C-Wide, A2C-Enhanced variants

## Timeline

| Run | Variant | Seed | Status | ETA |
|-----|---------|------|--------|-----|
| 1/12 | HCA2C-Full | 42 | 62% complete | ~10:29 AM |
| 2/12 | HCA2C-Full | 43 | Pending | ~12:59 PM |
| 3/12 | HCA2C-Full | 44 | Pending | ~3:29 PM |
| 4/12 | HCA2C-Flat | 42 | Pending | ~5:59 PM |
| ... | ... | ... | ... | ... |
| 12/12 | A2C-Enhanced | 44 | Pending | Tomorrow ~4:12 PM |

---

**All systems operational. Experiments running smoothly.** 🚀
