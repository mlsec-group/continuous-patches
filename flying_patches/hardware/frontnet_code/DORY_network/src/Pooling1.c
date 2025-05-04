/*
 * pooling_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// first_layer                    0
// sdk                            gap_sdk
// number_of_clusters             1
// optional_type                  8bit
// func_name                      Pooling1
// flag_DW                        0
// optional                       MaxPool
// FLAG_BATCHNORM                 0
// has_bias                       0
// FLAG_RELU                      0
// type                           char
// conv_overlap1                  0
// conv_overlap2                  0
// padding_top                    0
// padding_bottom                 0
// padding_left                   0
// padding_right                  0
// stride                         2
// g                              1
// nif                            32
// out_shift                      0
// data_type_x                    uint
// data_type_y                    uint
// data_type_activations          int
// data_type_weights              int
// nof                            32
// factor                         1
// double_buffering               2
// x_h                            48
// x_w                            80
// x_data_size_byte               8
// x_tile_size_nif                11
// x_tile_size_h                  18
// x_tile_size_w                  74
// x_tile_size_byte               14652
// x_tile_size_nif_byte           11
// x_stride_w_byte                2560
// x_stride_c_byte                32
// y_h                            24
// y_w                            40
// y_data_size_byte               8
// act_dim_bit                    None
// y_tile_size_nof                11
// y_tile_size_h                  9
// y_tile_size_w                  37
// y_tile_size_byte               3663
// y_stride_w_byte                1280
// y_stride_c_byte                32
// y_tile_size_nof_byte           11
// tile_dim_h                     3
// tile_dim_w                     2
// tile_dim_nof                   3
// tile_dim_nif                   3
// tile_n_in_last                 10
// fs1                            2
// fs2                            2
// W_data_size_byte               None
// W_tile_size_nof                11
// b_size_byte                    0
// W_tile_size_nif                33
// W_tile_size_nif_last           30
// k_tile_size_byte               0
// lambda_tile_size_byte          0
// k_size_byte                    0
// lambda_size_byte               0
// l1_x_offset                    0
// l1_y_offset                    29312
// y_tile_size_nof_last           10
// y_tile_size_h_last             6
// y_tile_size_w_last             3
// y_length_nof_byte_last         10
// x_tile_size_nif_last           10
// x_tile_size_nif_byte_last      10
// x_tile_size_h_last             12
// x_tile_size_w_last             6


#include "Pooling1.h"
void Pooling1(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_shift_in = (unsigned int) real_arg[10];
  int p_r, p_l, p_t, p_b;
  int last_nof_exec;
  int last_nif_exec;
  int last_h_exec;
  int last_w_exec;
  unsigned short x_tile_size_nif;
  unsigned short x_tile_size_h;
  unsigned short x_tile_size_w;
  unsigned short x_tile_size_byte;
  unsigned short x_length_h_px;
  unsigned short x_length_nif_byte;
  int pad_offset_h, pad_offset_w;
  char *x;
  char *y;
  int x_tile_size_nif_exec;
  int x_tile_size_h_exec;
  int x_tile_size_w_exec;
  int y_tile_size_nof;
  int y_tile_size_h;
  int y_tile_size_w;
  int y_tile_size_byte;
  int y_length_h_px;
  int y_length_nof_byte;
  int db_x;
  int db_y;
  int exec_db_x;
  int exec_db_W;
 char *im2col;
  im2col = l1_buffer + 36670;
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_x, DMA_copy_y;
  // copy first tiles
  //l2_x has input activations

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = 2560;
  DMA_copy_x.stride_1d = 32;
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = dory_dma_channel;

  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = 1280;
  DMA_copy_y.stride_1d = 32;
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = dory_dma_channel;

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + 0) + 0;
  DMA_copy_x.number_of_2d_copies = 18;
  DMA_copy_x.number_of_1d_copies = 74;
  DMA_copy_x.length_1d_copy = 11;
  dory_dma_memcpy_async(DMA_copy_x);
  dory_dma_barrier(DMA_copy_x);
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

  // double buffering state
  int db_state_x=0;
  int db_state_y=1;
  int db_state_acc_out=1;
  int flag_first_ch_out;

  // last-tile flags
  int last_nof_load = (3 == 1) ? 1 : 0;
  int last_nif_load = (3 == 1) ? 1 : 0;
  int last_h_load = (3 == 1) ? 1 : 0;
  int last_w_load = (2 == 1) ? 1 : 0;
  int iter;
  // tile loop nest
  for(iter=0; iter<3*3*2; iter++) {
    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==2)
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==3)
      {
        _i_h_load = 0;
        _i_nif_load += 1;
        _i_nof_load += 1;
      }
    }
    if (_i_nof_exec==0)
      flag_first_ch_out = 1;
    else
      flag_first_ch_out = 0;
    // wait for x,W read
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_w_exec = last_w_load;
    last_nof_load = (_i_nof_load+1 == 3) ? 1 : 0;
    last_nif_load = (_i_nof_load+1 == 3) ? 1 : 0;
    last_h_load = (_i_h_load+1 == 3) ? 1 : 0;
    last_w_load = (_i_w_load+1 == 2) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 14652 : 0;
    db_y = !db_state_y ? 3663 : 0;
    exec_db_x = db_state_x ? 14652 : 0;
    db_state_x = ! db_state_x;

    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single filter_out
    db_state_y = ! db_state_y;
    if(iter<3*3*2-1)
    {
      x_tile_size_nif = (last_nif_load) ? 10 : 11;
      x_tile_size_h   = (last_h_load)   ? 12 : 18;
      x_tile_size_w   = (last_w_load)   ? 6 : 74;
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*8/8;
      x_length_nif_byte = (last_nif_load)   ? 10 : 11;
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = 0;
      if(_i_w_load > 0)
        pad_offset_w = 0;

      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, 18, 74, 11, 80, 32,  0, 0,0, pad_offset_h, pad_offset_w, 0, 8);
      DMA_copy_x.loc = (l1_buffer + 0) + db_x;
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      dory_dma_memcpy_async(DMA_copy_x);
      y_tile_size_h   = (last_h_load)   ? 6 : 9;
      y_tile_size_w   = (last_w_load)   ? 3 : 37;
    }
    x = (char *) (l1_buffer + 0 + exec_db_x);
    y = (char *) (l1_buffer + 29312 + db_y);

    x_tile_size_nif_exec = (last_nif_exec) ? 10 : 11;
    x_tile_size_h_exec   = (last_h_exec)   ? 12 : 18;
    x_tile_size_w_exec   = (last_w_exec)   ? 6 : 74;

    y_tile_size_nof = (last_nof_exec) ? 10 : 11;
    y_tile_size_h   = (last_h_exec)   ? 6 : 9;
    y_tile_size_w   = (last_w_exec)   ? 3 : 37;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*8/8;
    y_length_nof_byte = (last_nof_exec)   ? 10 : 11;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = 0;
    if (_i_w_exec == 0)
      p_l = 0;
    if (_i_h_exec == 3-1)
      p_b = 0;
    if (_i_w_exec == 2-1)
      p_r = 0;
    dory_cores_barrier();

// aggiungere padding su tutti i lati, acc_out, and filter asymettric
    pulp_nn_maxpool(
    x, y,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    y_tile_size_w,
    y_tile_size_h,
    2,
    2,
    p_t,
    p_b,
    p_l,
    p_r,
    2,
    2
    );
    dory_cores_barrier();
    dory_dma_barrier(DMA_copy_x);
    dory_dma_barrier(DMA_copy_y);
    // transfering of output to L2
    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 9, 37, 11, 40, 32, 0, 0, 0, 0, 0, 0, 8);
    DMA_copy_y.loc = (l1_buffer + 29312) + db_y;
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    dory_dma_memcpy_async(DMA_copy_y);
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    dory_cores_barrier();
  }
  // wait for final write
  dory_dma_barrier(DMA_copy_y);
  dory_dma_deallocate(dory_dma_channel);
}
