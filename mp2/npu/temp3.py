

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    
    assert in_channels % 128 == 0

    
    assert nl.tile_size.gemm_moving_fmax >= out_width

    B, Cin, H, Win = X.shape
    Cout, Cin_, FH, FW = W.shape
    assert Cin == Cin_, "Cin mismatch"

    OH = H - FH + 1
    OW = Win - FW + 1

    TILE_K = nl.tile_size.pmax                  # 128 (partition dim)
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128 (M)
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512 (N)
    assert Cin % TILE_K == 0
    assert Cout % TILE_M == 0
    assert OW <= TILE_N

    nK = Cin // TILE_K
    nM = Cout // TILE_M
    
    ACC_TILE = 16
    Y = nl.ndarray((B, Cout, OH, OW), dtype=X.dtype, buffer=nl.hbm)

    for b in nl.affine_range(B):

        for m in nl.affine_range(nM):
            c0 = m * TILE_M

            W_blk_hbm = W[c0:c0 + TILE_M, :, :, :]                # [128, Cin, FH, FW]
            W_blk = nl.ndarray(W_blk_hbm.shape, dtype=W.dtype, buffer=nl.sbuf)
            W_blk[...] = nl.load(W_blk_hbm)

            # Bias tile (broadcast over width later)
            bias_tile = nl.ndarray((TILE_M, 1), dtype=bias.dtype, buffer=nl.sbuf)
            bias_tile[...] = nl.load(bias[c0:c0 + TILE_M])

            for h in nl.sequential_range(OH):
                # PSUM must be fp32
                acc = nl.zeros((TILE_M, OW), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(nK):
                    k0 = k * TILE_K

                    # HBM slice: [128, FH, OW+FW-1]
                    fused_hbm = X[b, k0:k0 + TILE_K, h:h + FH, 0:OW + FW - 1]

                    fused = nl.ndarray((nl.par_dim(TILE_K), FH, OW + FW - 1),
                                       dtype=X.dtype, buffer=nl.sbuf)
                    fused[...] = nl.load(fused_hbm)  # ONE nl.load per (h, k0)

                    for fi in nl.affine_range(FH):
                        row = fused[:, fi, :]                             # [128, OW+FW-1]
                        for fj in nl.affine_range(FW):
                            X_tile = row[:, fj: fj + OW]                   # [128, OW]
                            W_tile = W_blk[:, k0:k0 + TILE_K, fi, fj]      # [128, 128]
                            acc += nl.matmul(W_tile, X_tile)               

                acc_sb = nl.copy(acc, dtype=Y.dtype)                       
                out_sb = nl.add(acc_sb, bias_tile)                         
                nl.store(Y[b, c0:c0 + TILE_M, h, :], value=out_sb)
    return Y

