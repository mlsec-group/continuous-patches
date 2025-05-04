#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "commander.h"
#include "peer_localization.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "log.h"
#include "param.h"
#include "math3d.h"

#include "uart1.h"
#include "debug.h"

#define DEBUG_MODULE "APP"


const float distance = 1.0;  // distance between UAV and target in m
const float angle = 0.0;   // angle between UAV and target
const float tau = 1/100.0; // update rate

bool start_main = false;
uint8_t peer_id = 5;
float max_velo_v = 1.0f;

struct vec target_pos;
float target_yaw;

float x_d = 0;
float y_d = 0;
float z_d = 0;
float phi_d = 0;

const uint32_t baudrate_esp32 = 115200;

// helper for calculating the angle between two vectors
float calcAngleBetweenVectors(struct vec vec1, struct vec vec2)
{
  struct vec vec1_norm = vnormalize(vec1);
  struct vec vec2_norm = vnormalize(vec2);
  float dot = vdot(vec1_norm, vec2_norm);
  return acosf(dot);
}

// calculate the vector of the heading, angle in radians
struct vec calcHeadingVec(float radius, float angle)
{
  float x = (float) 0.0f + radius * cosf(angle);
  float y = (float) 0.0f + radius * sinf(angle);
  return mkvec(x, y, 0.0f);
}

void appMain()
{

  static setpoint_t setpoint;
  setpoint.mode.x = modeAbs;
  setpoint.mode.y = modeAbs;
  setpoint.mode.z = modeAbs;
  setpoint.mode.yaw = modeAbs;
  
  struct vec p_D = vzero();  // current position
  struct vec p_D_prime = vzero(); // next position

  struct vec e_H_delta = vzero(); // heading vector of target multiplied by safety distance
  
  float estYawRad;  // estimate of current yaw angle


  logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");

  logVarId_t idXEstimate = logGetVarId("stateEstimate", "x");
  logVarId_t idYEstimate = logGetVarId("stateEstimate", "y");
  logVarId_t idZEstimate = logGetVarId("stateEstimate", "z");

  // init UART
  DEBUG_PRINT("[DEBUG] Init UART...\n");
  uart1Init(baudrate_esp32);
  DEBUG_PRINT("[DEBUG] done!\n");

  while(1) {
    vTaskDelay(M2T(10));

    // Receive UART packages from the AI Deck 
    uint8_t dummy = 0x00;
    uint8_t uart_buffer[16];
    
    // wait until magic byte is received
    while (dummy != 0xBC)
    {
      uart1Getchar((uint8_t*)&dummy);
    }
    
    uart1Getchar((uint8_t*)&dummy);
    uint8_t length = dummy;
    // if the next byte is transmitting the correct length, proceed reading the message
    if (length == sizeof(uart_buffer))
    {
      for (uint8_t i = 0; i < length+1; i++)
      {
        uart1Getchar((uint8_t*)&uart_buffer[i]);
      }
    }

    //TODO: implement crc check    

    // get the current 3D coordinate of the UAV
    p_D.x = logGetFloat(idXEstimate);
    p_D.y = logGetFloat(idYEstimate);
    p_D.z = logGetFloat(idZEstimate);

    // calculate the correct float values of the target pose received via UART
    int32_t x = *(int32_t *)(uart_buffer + 0);
    int32_t y = *(int32_t *)(uart_buffer + 4);
    int32_t z = *(int32_t *)(uart_buffer + 8);
    int32_t phi = *(int32_t *)(uart_buffer + 12);

    x_d = (float)x * 2.46902e-05f + 1.02329e+00f;
    y_d = (float)y * 2.46902e-05f + 7.05523e-04f;
    z_d = (float)z * 2.46902e-05f + 2.68245e-01f;
    phi_d = (float)phi * 2.46902e-05f + 5.60173e-04f;

    target_pos = mkvec(x_d, y_d, z_d);   // target pose in UAV frame

    // translate the target pose in world frame
    estYawRad = radians(logGetFloat(idStabilizerYaw));    // get the current yaw in degrees
    struct quat q = rpy2quat(mkvec(0,0,estYawRad));
    target_pos = vadd(p_D, qvrot(q, target_pos));

    // calculate the UAV's yaw angle
    struct vec target_drone_global = vsub(target_pos, p_D);
    target_yaw = atan2f(target_drone_global.y, target_drone_global.x);

    
    setpoint.attitude.yaw = degrees(target_yaw);
    // z is kept at same height as target
    // setpoint.position.z = target_pos.z;

    // eq 6
    e_H_delta = calcHeadingVec(1.0f*distance, target_yaw-M_PI_F);  // target_yaw was shifted by 180°, correction through -pi
    // radius is 1 * distance
    // angle is set to the current yaw angle (rad) of the target

    p_D_prime = vadd(target_pos, e_H_delta);
    setpoint.position.x = p_D_prime.x;
    setpoint.position.y = p_D_prime.y;
    setpoint.position.z = 1.0f;           // keep z fixed an low for now, crashes from higher up damage the AI deck

    setpoint.velocity_body = false;  // world frame

    // clamp the values to prevent the UAV from flying into the savety net
    setpoint.position.x = clamp(setpoint.position.x, -0.8f, 0.8f);
    setpoint.position.y = clamp(setpoint.position.y, -0.8f, 0.8f);
    setpoint.position.z = clamp(setpoint.position.z, 0.0f, 2.0f);

    // only update the setpoint as soons as start_main is set to true
    if (start_main) {
    commanderSetSetpoint(&setpoint, 3);
    }//end if
  }//end while
}//end main


/**
 * Parameters to set the start flag, peer id and max velocity
 * for the frontnet-like controller.
 */
PARAM_GROUP_START(frontnet)
/**
 * @brief Estimator type Any(0), complementary(1), kalman(2) (Default: 0)
 */
PARAM_ADD_CORE(PARAM_UINT8, start, &start_main)
PARAM_ADD_CORE(PARAM_UINT8, cfid, &peer_id)
PARAM_ADD_CORE(PARAM_FLOAT, maxvelo, &max_velo_v)


PARAM_GROUP_STOP(frontnet)

// log group for local variables
LOG_GROUP_START(frontnet)
LOG_ADD(LOG_FLOAT, targetx, &target_pos.x)
LOG_ADD(LOG_FLOAT, targety, &target_pos.y)
LOG_ADD(LOG_FLOAT, targetz, &target_pos.z)
LOG_ADD(LOG_FLOAT, targetyaw, &target_yaw)

LOG_ADD(LOG_FLOAT, x_uart, &x_d)
LOG_ADD(LOG_FLOAT, y_uart, &y_d)
LOG_ADD(LOG_FLOAT, z_uart, &z_d)
LOG_ADD(LOG_FLOAT, phi_uart, &phi_d)
LOG_GROUP_STOP(frontnet)
