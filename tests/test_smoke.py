"""
test_smoke.py — guards the wiring the reorg could break: paths resolve, the production
checkpoints load, and the feature contract (291 M1 feats incl. macro / 14 M2 feats) holds.

Run:  py -3.10 tests/test_smoke.py        (plain, no pytest needed)
      py -3.10 -m pytest tests/           (also works under pytest)
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import paths


def test_paths_resolve():
    """Every directory paths.py advertises must exist (and ROOT must not be hardcoded)."""
    for name in ('ROOT', 'SRC_DIR', 'DATA_DIR', 'MODELS_DIR', 'OUTPUTS_DIR', 'LOGS_DIR', 'CONFIG_DIR'):
        p = getattr(paths, name)
        assert os.path.isdir(p), f"{name} missing: {p}"
    assert os.path.basename(paths.SRC_DIR) == 'src'


def test_market_data_present():
    """The 4 timeframes x 3 assets the pooled model needs, plus the macro source."""
    import crypto_features as FE
    for asset, tfs in FE.ASSETS.items():
        for tf, fname in tfs.items():
            assert os.path.exists(os.path.join(paths.DATA_DIR, fname)), f"missing {asset}/{tf}: {fname}"
    for m in ('nq_1d_data.csv', 'es_1d_data.csv', 'gold_1d_data_2018_to_2025.csv'):
        assert os.path.exists(os.path.join(paths.DATA_DIR, m)), f"missing macro source {m}"


def test_production_m1_contract():
    """Production M1 loads and still has the 291-feature macro contract."""
    import torch
    import btc_backtest_cross_nf as BT
    ckpt = torch.load(BT.CKPT_PATH, map_location='cpu', weights_only=False)
    assert ckpt['model_cfg']['n_feat'] == 291, ckpt['model_cfg']['n_feat']
    assert ckpt['seq_len'] == 64
    assert 'mac_regime' in ckpt['feat_cols'], "macro features missing from production M1"
    assert not any(c.startswith('fund_') for c in ckpt['feat_cols']), "rejected funding feats leaked in"


def test_production_m2_contract():
    """Production M2 loads, is paired to M1, and carries no rejected conviction-dynamics cols."""
    import btc_meta_label as ML
    m2 = ML.load_m2()
    assert len(m2['feat_order']) == 14, m2['feat_order']
    assert not any(c.startswith('cd_') for c in m2['feat_order']), "rejected convdyn feats leaked in"
    assert 0.0 < m2['skip_thr'] < m2['full_thr'] <= 1.0


def test_default_flags_are_production():
    """Rejected experiments must stay default-OFF so production is the import-time default."""
    import crypto_features as FE
    import btc_meta_label as ML
    assert FE.USE_MACRO is True, "MACRO is production — must default ON"
    assert FE.USE_FUNDING is False, "FUNDING was rejected — must default OFF"
    assert ML.USE_CONVDYN is False, "CONVDYN was rejected — must default OFF"
    assert FE.TAG == 'emb_cross_nf', FE.TAG


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            fails += 1; print(f"  FAIL  {name}: {e}")
    print(f"\n{'all smoke tests passed' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)
