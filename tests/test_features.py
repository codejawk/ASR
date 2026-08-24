import torch

from edge_asr.features import LogMelFrontend, OnlineCMVN


def test_frontend_shapes():
    fe = LogMelFrontend(n_mels=80)
    wav = torch.randn(16000)  # 1 s
    feats = fe(wav)
    assert feats.shape[0] == 1
    assert feats.shape[2] == 80
    # ~100 frames/s
    assert 95 <= feats.shape[1] <= 105
    assert torch.isfinite(feats).all()


def test_cmvn_is_causal():
    fe = LogMelFrontend(n_mels=80)
    cmvn = OnlineCMVN(n_mels=80, warmup_frames=10)
    feats = fe(torch.randn(16000))
    out = cmvn(feats)
    # changing a late frame must NOT change an early normalized frame
    feats2 = feats.clone()
    feats2[0, -1] += 5.0
    out2 = cmvn(feats2)
    assert torch.allclose(out[0, 0], out2[0, 0], atol=1e-5)


if __name__ == "__main__":
    test_frontend_shapes()
    test_cmvn_is_causal()
    print("OK")
