Sensors + Camera (Drone)
        ↓
Ngrok Tunnel (Upstream)
        ↓
GPU System (ROS + LSTM + YOLO)
        ↓
Desired State Output:
- Roll rate
- Pitch rate
- Yaw rate
- Throttle
- Mode (takeoff / land / cruise)
- Constants (altitude, heading, etc.)
        ↓
Ngrok Tunnel (Downstream)
        ↓
Embedded System:
- Apply constants
- Convert rates → PWM
- Maintain stability



[ CM4/CM5 Module ]
   - Camera (CSI)
   - Quectel modem (internet)
   - ngrok tunnel
   - UART link to carrier board (data in/out)

         |
         | UART
         v

[ Carrier Board (all components) ]
   - STM32 MCU (real-time controller)
   - Shift register + UART mux logic
   - Sensors (IMU, GPS, baro, etc.)
   - PWM outputs to ESCs
   - Battery monitor + failsafe
   - Mounting holes for CM module





                         +----------------------+
                         |     EDGE SERVER      |
                         |  (YOLOv8 + LSTM)     |
                         |----------------------|
                         | - Receives sensor    |
                         |   + camera data      |
                         | - Runs inference     |
                         | - Sends control cmds |
                         |   (rates + throttle) |
                         +----------+-----------+
                                    |
                            ngrok tunnel (bi-dir)
                                    |
                                    v

+-----------------------------+     +----------------------------------+
|       CM4/CM5 Module        |     |        Carrier Board (PCB)       |
|  (Plug-in module)           |     |----------------------------------|
|-----------------------------|     | - STM32 / RP2350 MCU             |
| - Camera (CSI)              |     | - Shift register + UART mux      |
| - Quectel modem (Internet)  |     | - Sensors (IMU, GPS, baro, etc.)|
| - Streams sensor + camera   |     | - PWM outputs to ESCs (4 motors)|
|   data upstream             |     | - Battery monitoring + failsafe  |
| - Receives rates + throttle |     | - Physical UART link to CM       |
|   from edge                 |     | - Mounting holes for CM module   |
+-------------+---------------+     +----------------+-----------------+
              | UART line (physical header)              |
              |                                          |
              v                                          v
        (Data In/Out)                            (All sensors & MCU)







CM4/CM5 SODIMM Pinout (selected)

| Pin | Signal         | Use Case                             |
|-----|----------------|--------------------------------------|
| 1   | 3V3            | Power to CM module                   |
| 2   | 5V             | Power to carrier board / sensors     |
| 3   | GND            | Common ground                        |
| 4   | UART0_TXD      | CM -> MCU (data out)                 |
| 5   | UART0_RXD      | MCU -> CM (data in)                  |
| 6   | UART1_TXD      | Optional extra UART (if needed)      |
| 7   | UART1_RXD      | Optional extra UART (if needed)      |
| 8   | I2C0_SDA       | Optional (not needed in your design) |
| 9   | I2C0_SCL       | Optional                             |
| 10  | SPI0_MOSI      | Optional (not needed)                |
| 11  | SPI0_MISO      | Optional                             |
| 12  | SPI0_SCLK      | Optional                             |
| 13  | GPIO (CE)      | Shift register control (latch)       |
| 14  | GPIO (CLK)     | Shift register control (clock)       |
| 15  | GPIO (DATA)    | Shift register control (data)        |
| 16  | CSI0_D0        | Camera data lane 0                   |
| 17  | CSI0_D1        | Camera data lane 1                   |
| 18  | CSI0_D2        | Camera data lane 2                   |
| 19  | CSI0_D3        | Camera data lane 3                   |
| 20  | CSI0_CLK       | Camera clock                         |
| 21  | CSI0_HSYNC     | Camera HSync                         |
| 22  | CSI0_VSYNC     | Camera VSync                         |
| 23  | GPIO (RESET)   | Optional camera reset                 |
| 24  | GPIO (PWR)     | Optional camera power control        |
| 25  | GND            | Ground                               |





STM32 UART (MCU) pins:
- UART1_TX  -> CM UART0_RXD (CM input)
- UART1_RX  -> CM UART0_TXD (CM output)

STM32 GPIO pins:
- GPIOA0 -> SHIFT_REG_CLK
- GPIOA1 -> SHIFT_REG_LATCH
- GPIOA2 -> SHIFT_REG_DATA
- GPIOA3 -> SENSOR_SELECT_0
- GPIOA4 -> SENSOR_SELECT_1
- GPIOA5 -> SENSOR_SELECT_2
- GPIOA6 -> SENSOR_SELECT_3

STM32 PWM outputs:
- TIM1_CH1 -> ESC1
- TIM1_CH2 -> ESC2
- TIM1_CH3 -> ESC3
- TIM1_CH4 -> ESC4

STM32 ADC inputs:
- ADC1_IN0 -> Battery voltage
- ADC1_IN1 -> Battery current



74HC595 pins:
- SER (data)   -> STM32 GPIOA2
- SRCLK (clk)  -> STM32 GPIOA0
- RCLK (latch) -> STM32 GPIOA1
- OE           -> GND
- MR           -> VCC (pull-up)
- Q0..Q7       -> UART mux select lines




or




74HC4067 Control Pins:
- S0 -> STM32 GPIOA3
- S1 -> STM32 GPIOA4
- S2 -> STM32 GPIOA5
- S3 -> STM32 GPIOA6
- EN -> STM32 GPIOA7 (enable)

UART mux output:
- UART_MUX_TX -> sensor TX
- UART_MUX_RX -> sensor RX

