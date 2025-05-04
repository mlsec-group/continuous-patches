/*
 * test_template.c
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



#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/camera.h"
#include "bsp/camera/himax.h"

#include "network.h"

PI_L2 char *buff;

PI_L2 char* L2_memory_buffer;
PI_L2 char* L2_output;

PI_L2 uint8_t crc;

static struct pi_device camera;

#define WIDTH 162
#define HEIGHT 162

#define BUFF_SIZE (WIDTH*HEIGHT)

static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
static int activations_input;

//uart
static struct pi_device uart;
static PI_L2 uint8_t magic = 0xBC;
static PI_L2 uint8_t length = sizeof(int32_t)*4;

static void set_register(uint32_t reg_addr, uint8_t value)
{
  uint8_t set_value = value;
  pi_camera_reg_set(&camera, reg_addr, &set_value);
}

// filesystem management functions
void open_filesystem_and_ram(struct pi_device *flash, struct pi_device *fs)
{
  struct pi_readfs_conf conf;
  struct pi_hyperflash_conf flash_conf;

  /* Init & open flash. */
  pi_hyperflash_conf_init(&flash_conf);
  pi_open_from_conf(flash, &flash_conf);
  if (pi_flash_open(flash))
  {
      // cpxPrintToConsole(LOG_TO_CRTP, "Error flash open !\n");
      // printf("Error flash open !\n");
      pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&conf);
  conf.fs.flash = flash;
  pi_open_from_conf(fs, &conf);
  if (pi_fs_mount(fs))
  {
      // cpxPrintToConsole(LOG_TO_CRTP, "Error FS mounting !\n");
      // printf("Error FS mounting !\n");
      pmsis_exit(-2);
  }
  pi_task_t task = {0};
  pi_task_block(&task);
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
}

void cropImage(char *imgBuff, char *L2_input_buffer)
{
  // set initial offset to pixel imgBuff[34][1] 
  uint32_t offset = 34*162;

  char *curr_adr = L2_input_buffer;

  for (uint8_t idx_h = 0; idx_h <97; idx_h++)
  {
    // copy 160 bytes to the current row of the image
    memcpy(curr_adr, imgBuff + (offset * sizeof(char)), 160*sizeof(char));
    // jump to the next row in imgBuff, skip the 2 dead pixels on the right and left
    offset += 162;
    // jump to the next row in L2_input_buffer
    curr_adr += 160 * sizeof(char);
  }
}


static int open_camera(struct pi_device *device)
{
    // printf("Opening Himax camera\n");
    struct pi_himax_conf cam_conf;
    pi_himax_conf_init(&cam_conf);

    cam_conf.format = PI_CAMERA_QQVGA;

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
        return -1;

    // Rotate camera orientation
    pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
    uint8_t set_value = 3;
    uint8_t reg_value;

    pi_camera_reg_set(&camera, IMG_ORIENTATION, &set_value);
    pi_time_wait_us(1000000);
    pi_camera_reg_get(&camera, IMG_ORIENTATION, &reg_value);
    if (set_value!=reg_value)
    {
        // printf("Failed to rotate camera image\n");
        return -1;
    }
    pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

    pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

  uint8_t aeg = 0x00;
  uint8_t aGain = 4;
  uint8_t dGain = 1;
  uint16_t exposure = 100;


   set_register(0x2100, aeg);  // AE_CTRL

      switch(aGain) {
        case 8:
          set_register(0x0205, 0x30);
          break;
        case 4:
          set_register(0x0205, 0x20);
          break;
        case 2:
          set_register(0x0205, 0x10);
          break;
        case 1:
        default:
          set_register(0x0205, 0x00);
          break;
      }

      set_register(0x020E, (dGain >> 6)); // 2.6 int part
      set_register(0x020F, dGain & 0x3F); // 2.6 float part

      
      if (exposure < 2) {
        exposure = 2;
      }
      if (exposure > 0x0216 - 2) {
        exposure = 0x0216 - 2;
      }
      set_register(0x0202, (exposure >> 8) & 0xFF);    // INTEGRATION_H
      set_register(0x0203, exposure & 0xFF);    // INTEGRATION_L

    return 0;
}

static int open_uart(struct pi_device *device)
{
    struct pi_uart_conf conf;
    pi_uart_conf_init(&conf);
    conf.baudrate_bps = 115200;

    pi_open_from_conf(&uart, &conf);
    if (pi_uart_open(&uart))
    {
        // printf("[UART] open failed !\n");
        pmsis_exit(-1);
    }

    return 0;

}


int prediction_task(void)
{
    // Open the Himax camera
    if (open_camera(&camera))
    {
        // printf("Failed to open camera\n");
        pmsis_exit(-1);
    }

    // Open UART
    if (open_uart(&uart))
    {
        // printf("Failed to open uart\n");
        pmsis_exit(-1);
    }else
    {
        // printf("UART initialized!\n");
    }

    /*
    Opening of Filesystem and Ram
    */
    struct pi_device fs;
    struct pi_device flash;
    open_filesystem_and_ram(&flash, &fs);
    pi_ram_alloc(&ram, &activations_input, (uint32_t) 500000);

    /*
    Allocate buffers
    */
    char* L2_input;
    L2_memory_buffer = pi_l2_malloc((uint32_t) 380000);
    int begin_end = 1;
    L2_input = L2_memory_buffer;
    L2_output = pi_l2_malloc((int32_t) 4);

    buff = pmsis_l2_malloc(WIDTH*HEIGHT*sizeof(char));
    if (buff == NULL){ return -1;}
    
    /*
    Allocation of the network
    */
    pi_ram_read(&ram, activations_input, L2_input, 15360);
    network_alloc(fs, ram);

    #ifdef VERBOSE
    // printf("Initialized buffers\n");
    #endif

    while(1)
    {
        #ifdef VERBOSE
        // printf("Entering loop...\n");
        // printf("Start camera...\n");
        #endif
        // Start the camera
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
        #ifdef VERBOSE
        // printf("Capturing image...\n");
        #endif
        // Capture an image
        pi_camera_capture(&camera, buff, BUFF_SIZE);
        #ifdef VERBOSE
        // printf("Stopping camera...\n");
        #endif

        // Stop the camera
        pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

        #ifdef VERBOSE
        // printf("Cropping image...\n");
        #endif
        
        // Crop image to 96x160 pixels, 
        // only keep the middle part of the image and skip the two dead pixels on the left and right
        cropImage(buff, L2_input);

        // Get prediction from the network
        begin_end = 1;
        network_run(L2_memory_buffer, 380000, L2_output, begin_end, ram);

        // Send the output of the network via UART
        pi_uart_write(&uart, &magic, 1);
        pi_uart_write(&uart, &length, 1);
        pi_uart_write(&uart, L2_output, 4*sizeof(int32_t));
        
        // compute crc
        crc = 0;
        for (const uint8_t* p = (uint8_t*)L2_output; p < (uint8_t*)L2_output + length; p++) {
            crc ^= *p;
        }
        pi_uart_write(&uart, crc, 1);
   
    }

    // printf("Exiting loop!\n");
    pi_ram_free(&ram, activations_input, 500000);
    network_free(ram);  

    pmsis_exit(0);

}

int main(void)
{
    // printf("\n\t*** PMSIS FRONTNET ***\n\n");
    PMU_set_voltage(1000, 0);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
    pi_time_wait_us(10000);
    return pmsis_kickoff((void *) prediction_task);
}