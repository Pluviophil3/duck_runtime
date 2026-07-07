import serial
import time

# 配置参数 - 根据实际情况修改
SERIAL_PORT = '/dev/ttyS0'    # 串口号
BAUD_RATE = 9600             # 波特率
TIMEOUT = 1                  # 读取超时时间(秒)
MAX_RETRIES = 3              # 最大重试次数

def main():
    print(f"串口通信测试程序 - 端口: {SERIAL_PORT}, 波特率: {BAUD_RATE}")
    print("按 Ctrl+C 终止程序\n")
    
    ser = None
    retries = 0
    
    try:
        # 初始化串口连接
        while retries < MAX_RETRIES:
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
                print(f"成功连接到串口 {SERIAL_PORT}")
                break
            except serial.SerialException as e:
                retries += 1
                print(f"连接失败 ({retries}/{MAX_RETRIES}): {e}")
                time.sleep(1)
        
        if not ser or not ser.is_open:
            print(f"无法连接到串口 {SERIAL_PORT}，程序退出")
            return
        
        # 主循环 - 读取并打印数据
        while True:
            try:
                # 检查是否有数据可读
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8', errors='replace').strip()
                    if data:
                        print(f"收到数据: {data}")
                else:
                    # 没有数据时短暂休眠，减少CPU占用
                    time.sleep(0.01)
                    
            except UnicodeDecodeError as e:
                print(f"解码错误: {e}")
                # 清空输入缓冲区，防止错误数据累积
                ser.reset_input_buffer()
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except serial.SerialException as e:
        print(f"串口通信错误: {e}")
    finally:
        # 确保串口始终被关闭
        if ser and ser.is_open:
            ser.close()
            print("已关闭串口连接")

if __name__ == "__main__":
    main()