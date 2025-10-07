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
    
    INPUT_CHANNEL_TILE = nl.tile_size.pmax                  # 128 (partition dim)
    INPUT_CHANNEL_CHUNKS = in_channels // INPUT_CHANNEL_TILE
    INPUT_HEIGHT_TILE = 16
    INPUT_CHUNKS = (input_height + INPUT_HEIGHT_TILE - 1)//INPUT_HEIGHT_TILE
    INPUT_HEIGHT_CHUNK = (INPUT_HEIGHT_TILE + filter_height - 1)
    COUT_TILE = nl.tile_size.pmax # 128
    COUT_CHUNKS = out_channels // COUT_TILE
    Y = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    for b in nl.affine_range(batch_size):
        all_input_chunks_sbuf = nl.ndarray(
            (INPUT_CHUNKS, INPUT_CHANNEL_CHUNKS, nl.par_dim(INPUT_CHANNEL_TILE), INPUT_HEIGHT_CHUNK, input_width),
            dtype=X.dtype,
            buffer=nl.sbuf
        )
        for input_chunk in nl.affine_range(INPUT_CHUNKS):
            h_start = input_chunk * INPUT_HEIGHT_TILE
            for in_ch_tile in nl.affine_range(INPUT_CHANNEL_CHUNKS):
                c_start = in_ch_tile * INPUT_CHANNEL_TILE
                ch_idx, row_idx, col_idx = nl.mgrid[0:INPUT_CHANNEL_TILE, 0:INPUT_HEIGHT_CHUNK, 0:input_width]
                all_input_chunks_sbuf[input_chunk, in_ch_tile, :, :, :] = nl.load(X[b, c_start:c_start+INPUT_CHANNEL_TILE, 
                                                                                    h_start:h_start+INPUT_HEIGHT_CHUNK, 0:input_width], 
                                                                                  mask=(h_start + row_idx) < input_height)
            for cout_tile in nl.affine_range(COUT_CHUNKS):
                W_blk_hbm = W[cout_tile * COUT_TILE:(cout_tile + 1) * COUT_TILE, :, :, :]             
                W_blk = nl.ndarray(W_blk_hbm.shape, dtype=W.dtype, buffer=nl.sbuf)
                W_blk[...] = nl.load(W_blk_hbm)

                # Bias tile (broadcast over width later)
                bias_tile = nl.ndarray((COUT_TILE, 1), dtype=bias.dtype, buffer=nl.sbuf)
                bias_tile[...] = nl.load(bias[cout_tile * COUT_TILE:(cout_tile + 1) * COUT_TILE])
                result_sbuf = nl.zeros((COUT_TILE, INPUT_HEIGHT_TILE, out_width), dtype=X.dtype, buffer=nl.sbuf)
                for h_in_chunk in nl.affine_range(INPUT_HEIGHT_TILE):
                    acc = nl.zeros(shape=(nl.par_dim(COUT_TILE), out_width),
                                    dtype=nl.float32,
                                    buffer=nl.psum)
                    for in_ch_tile in nl.affine_range(INPUT_CHANNEL_CHUNKS):
                        for fi in nl.affine_range(filter_height):
                            for fj in nl.affine_range(filter_width):
                                input_slice = all_input_chunks_sbuf[input_chunk,in_ch_tile, :, h_in_chunk + fi, fj:fj + out_width]
                                weight_slice = W_blk[:, in_ch_tile * INPUT_CHANNEL_TILE:(in_ch_tile + 1) * INPUT_CHANNEL_TILE, fi, fj]
                                acc[:, :] += nl.matmul(weight_slice, input_slice)
                    result_sbuf[:, h_in_chunk, :] = nl.add(acc, bias_tile)
                ch_idx, row_idx, col_idx = nl.mgrid[0:COUT_TILE, 0:INPUT_HEIGHT_TILE, 0:out_width]
                nl.store(Y[b, cout_tile * COUT_TILE:cout_tile * COUT_TILE + COUT_TILE, h_start:h_start+INPUT_HEIGHT_TILE, 0:out_width], 
                        result_sbuf, mask=(h_start + row_idx) < out_height)
   
            
    return Y