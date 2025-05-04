/*
 * network.h
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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "pulp.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "mem_controller.h"

typedef struct {
    unsigned int L3_input;
    unsigned int L3_output;
    unsigned int L3_after_weights;
    unsigned int L2_input;
    unsigned int bypass;
    unsigned int L2_output;
    unsigned int L2_weights;
    unsigned int L1_buffer;
    unsigned int ram;
    unsigned int out_mult;
    unsigned int out_shift;
    unsigned int layer_id;
} layer_args_t;

void network_free(struct pi_device ram);
void network_alloc(struct pi_device fs, struct pi_device ram);
void network_run(char *L2_memory_buffer, int L2_memory_dimension, char *L2_output_to_pass, int begin_end, struct pi_device ram);
void execute_layer_fork(void *arg);
void execute_layer(void *arg);

#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
const char * L3_weights_files[] = {
  "BNReluConvolution0_weights.hex", "BNReluConvolution2_weights.hex", "BNReluConvolution3_weights.hex", "BNReluConvolution4_weights.hex", "BNReluConvolution5_weights.hex", "BNReluConvolution6_weights.hex", "BNReluConvolution7_weights.hex", "FullyConnected8_weights.hex"
};
int L3_weights_size[8];
static int L3_weights;
static int L3_input;
static int L3_output;
static int layers_pointers[9];

static char * Layers_name[9] = {"BNReluConvolution0", "Pooling1", "BNReluConvolution2", "BNReluConvolution3", "BNReluConvolution4", "BNReluConvolution5", "BNReluConvolution6", "BNReluConvolution7", "FullyConnected8"};
static int L3_input_layers[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[9] = {1, 0, 1, 1, 1, 1, 1, 1, 1};
static int branch_input[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int check_weights[9] = {137822, 0, 1243157, 1242993, 2479657, 4939901, 9178282, 20239089, 706149};
static int check_weights_dimension[9] = {1312, 0, 9728, 9728, 19456, 37888, 75776, 149504, 7696};
static int cumulative_weights_dimension[9] = {0, 1312, 1312, 11040, 20768, 40224, 78112, 153888, 303392};
static int check_activations[9] = {1956945, 190544, 78451, 21981, 32374, 17120, 20789, 4775, 8937};
static int check_activations_dimension[9] = {15360, 122880, 30720, 7680, 7680, 3840, 3840, 1920, 1920};
static int out_mult_vector[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
static int out_shift_vector[9] = {24, 0, 23, 23, 23, 23, 24, 22, 0};
static int check_activations_out[9] = {190544, 78451, 21981, 32374, 17120, 20789, 4775, 8937, -30178};
static int check_activations_out_dimension[9] = {122880, 30720, 7680, 7680, 3840, 3840, 1920, 1920, 16};
static int layer_with_weights[9] = {1, 0, 1, 1, 1, 1, 1, 1, 1};
#endif

#endif  // __NETWORK_H__
