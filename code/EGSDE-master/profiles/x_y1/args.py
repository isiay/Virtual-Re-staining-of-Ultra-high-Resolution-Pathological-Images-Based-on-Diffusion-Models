import argparse
argsall = argparse.Namespace(
    testdata_path = '',
    ckpt = '',
    dsepath = '',
    config_path = '',
    t = 50,#500,#150, #500,
    ls = 500,
    li = 2,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step= 1,
    batch_size = 20,
    diffusionmodel = 'ADM',
    down_N = 32,
    seed=1234)