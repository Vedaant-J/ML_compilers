import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
import neuronxcc.nki.compiler as ncc

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    INPUT_CHANNEL_TILE = nl.tile_size.pmax                  # 128
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
    bias_sbuf = nl.ndarray(
        shape = (COUT_TILE, COUT_CHUNKS),
        dtype = bias.dtype,
        buffer = nl.sbuf
    )   
    
    weights_transposed = nl.ndarray(
        (filter_height, filter_width, COUT_CHUNKS, INPUT_CHANNEL_CHUNKS, nl.par_dim(COUT_TILE), INPUT_CHANNEL_TILE),
        dtype=W.dtype,
        buffer=nl.sbuf
    )
    
    for cout_idx in nl.affine_range(COUT_CHUNKS):
        bias_sbuf[:,cout_idx] = nl.load(bias[cout_idx * COUT_TILE : (cout_idx+1) * COUT_TILE])
        for cin_idx in nl.affine_range(INPUT_CHANNEL_CHUNKS):
            temp = nl.load(W[cout_idx * COUT_TILE:(cout_idx + 1)*COUT_TILE, 
                                                        cin_idx * INPUT_CHANNEL_TILE:(cin_idx + 1)*INPUT_CHANNEL_TILE,:,:])
            for fi in nl.affine_range(filter_height):
                for fj in nl.affine_range(filter_width):
                    weights_transposed[fi, fj, cout_idx, cin_idx] = nl.transpose(temp[:, :, fi, fj])
    
    for b in nl.affine_range(batch_size):
        for input_chunk in nl.affine_range(INPUT_CHUNKS):
            h_start = input_chunk * INPUT_HEIGHT_TILE
            input_chunk_sbuf = nl.ndarray(
                (INPUT_CHANNEL_CHUNKS, nl.par_dim(INPUT_CHANNEL_TILE), INPUT_HEIGHT_CHUNK, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )
            
            for in_ch_tile in nl.affine_range(INPUT_CHANNEL_CHUNKS):
                c_start = in_ch_tile * INPUT_CHANNEL_TILE
                ch_idx, row_idx, col_idx = nl.mgrid[0:INPUT_CHANNEL_TILE, 0:INPUT_HEIGHT_CHUNK, 0:input_width]
                mask_row = (h_start + row_idx)
                input_chunk_sbuf[in_ch_tile, :, :, :] = nl.load(X[b, c_start + ch_idx, 
                                                                                    mask_row, 0+col_idx], 
                                                                                  mask=mask_row < input_height)
            
            for cout_tile in nl.affine_range(COUT_CHUNKS):
                result_sbuf = nl.zeros((nl.par_dim(COUT_TILE), INPUT_HEIGHT_TILE, out_width), dtype=X.dtype, buffer=nl.sbuf)
                for h_in_chunk in nl.affine_range(INPUT_HEIGHT_TILE):
                    acc = nl.zeros(shape=(COUT_TILE, out_width),
                                    dtype=nl.float32,
                                    buffer=nl.psum)
                    for in_ch_tile in nl.affine_range(INPUT_CHANNEL_CHUNKS):
                        for fi in nl.affine_range(filter_height):
                            for fj in nl.affine_range(filter_width):
                                acc += nl.matmul(weights_transposed[fi, fj, cout_tile, in_ch_tile, :, :], 
                                                 input_chunk_sbuf[in_ch_tile, :, h_in_chunk + fi, fj:fj + out_width], 
                                                 transpose_x =True)
                    result_sbuf[:, h_in_chunk, :] = nl.add(acc, bias_sbuf[:,cout_tile])
                ch_idx, row_idx, col_idx = nl.mgrid[0:COUT_TILE, 0:INPUT_HEIGHT_TILE, 0:out_width]
                mask_row = (h_start + row_idx)
                nl.store(Y[b, cout_tile * COUT_TILE + ch_idx, mask_row, 0+col_idx], 
                        result_sbuf, mask=mask_row < out_height)
   
            
    return Y
