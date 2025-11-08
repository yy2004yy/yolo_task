from maix import app, uart, pinmap, time, camera, display, image, nn
from maix.image import ApriltagFamilies
import struct, math

# ports = uart.list_devices()
# pinmap.set_pin_function("A16", "UART0_TX")
# pinmap.set_pin_function("A17", "UART0_RX")
device = "/dev/ttyS0" 

serial0 = uart.UART(device, 115200) # 串口0  波特率 9600

#-------------------------------------------------------------#
# 数据打包变量
class target_check(object):
    type1=0          #int16_t  动物类型1
    type2=0          #int16_t  动物类型2
    pixel=0      #uint16_t
    flag=0       #uint8_t
    state=0      #uint8_t
    angle=0      #int16_t
    distance=0   #uint16_t
    apriltag_id=0#uint16_t
    img_width=0  #uint16_t
    img_height=0 #uint16_t
    reserved1=0  #uint8_t
    reserved2=0  #uint8_t
    reserved3=0  #uint8_t
    reserved4=0  #uint8_t
    fps=0        #uint8_t
    range_sensor1=0
    range_sensor2=0
    range_sensor3=0
    range_sensor4=0
    camera_id=0 
    loc_1_x=0 # int32 # 动物1 坐标x
    loc_1_y=0 # int32 # 动物1 坐标y
    loc_2_x=0 # int32 # 动物2 坐标x
    loc_2_y=0 # int32 # 动物2 坐标y

#串口数据
class uart_buf_prase(object):
    uart_buf = []
    _data_len = 0
    _data_cnt = 0
    state = 0

#摄像头模式
class mode_ctrl(object):
    work_mode = 0x01 #工作模式.默认是点检测，可以通过串口设置成其他模式
    check_show = 1   #开显示，在线调试时可以打开，离线使用请关闭，可提高计算速度
#-------------------------------------------------------------#

ctr=mode_ctrl()
R=uart_buf_prase()
target=target_check()
target.camera_id=0x01
target.loc_1_x=65536
target.loc_1_y=2025
target.loc_2_x=65537
target.loc_2_y=2025
    
# 数据帧头
HEADER=[0xFF,0xFC]
MODE=[0xF1,0xF2,0xF3]

#数据打包封装
def package_blobs_data(mode):
    data=bytearray([HEADER[0],HEADER[1],0xA0+mode,
                   0x00, # 后面替换为数据长度 
                   target.type1>>8&0xff, target.type1&0xff,        #将整形数据拆分成两个8位
                   target.type2>>8&0xff, target.type2&0xff,        #将整形数据拆分成两个8位
                   target.pixel>>8&0xff, target.pixel&0xff,#将整形数据拆分成两个8位
                   target.flag&0xff,                 #数据有效标志位
                   target.state&0xff,                #数据有效标志位
                   target.angle>>8&0xff, target.angle&0xff,#将整形数据拆分成两个8位
                   target.distance>>8&0xff, target.distance&0xff,#将整形数据拆分成两个8位
                   target.apriltag_id>>8&0xff, target.apriltag_id&0xff,#将整形数据拆分成两个8位
                   target.img_width>>8&0xff, target.img_width&0xff,    #将整形数据拆分成两个8位
                   target.img_height>>8&0xff, target.img_height&0xff,  #将整形数据拆分成两个8位
                   target.fps&0xff,      #数据有效标志位
                   target.reserved1&0xff,#数据有效标志位
                   target.reserved2&0xff,#数据有效标志位
                   target.reserved3&0xff,#数据有效标志位
                   target.reserved4&0xff,#数据有效标志位
                   target.range_sensor1>>8&0xff, target.range_sensor1&0xff,
                   target.range_sensor2>>8&0xff, target.range_sensor2&0xff,
                   target.range_sensor3>>8&0xff, target.range_sensor3&0xff,
                   target.range_sensor4>>8&0xff, target.range_sensor4&0xff,
                   target.camera_id&0xff,
                   target.loc_1_x>>24&0xff,target.loc_1_x>>16&0xff,
                   target.loc_1_x>>8&0xff, target.loc_1_x&0xff,
                   target.loc_1_y>>24&0xff,target.loc_1_y>>16&0xff,
                   target.loc_1_y>>8&0xff, target.loc_1_y&0xff,
                   target.loc_2_x>>24&0xff,target.loc_2_x>>16&0xff,
                   target.loc_2_x>>8&0xff, target.loc_2_x&0xff,
                   target.loc_2_y>>24&0xff,target.loc_2_y>>16&0xff,
                   target.loc_2_y>>8&0xff, target.loc_2_y&0xff,
                   0x00])
    #数据包的长度
    data_len=len(data)
    data[3]=data_len-5#有效数据的长度
    #和校验
    sum=0
    for i in range(0,data_len-1):
        sum=sum+data[i]
    data[data_len-1] = sum & 0xFF
    #返回打包好的数据
    return data

#和校验
def Receive_Anl(data_buf,num):
    sum = 0
    i = 0
    while i<(num-1):
        sum = sum + data_buf[i]
        i = i + 1
    sum = sum%256 #求余
    if sum != data_buf[num-1]:
        return
    #和校验通过
    if data_buf[2]==0xA0:
        #设置模块工作模式
        ctr.work_mode = data_buf[4]
        print(ctr.work_mode)
        print("Set work mode success!")

#串口数据处理
def uart_data_prase(buf):
    if R.state==0 and buf==0xFF:#帧头1
        R.state=1
        R.uart_buf.append(buf)
    elif R.state==1 and buf==0xFE:#帧头2
        R.state=2
        R.uart_buf.append(buf)
    elif R.state==2 and buf<0xFF:#功能字
        R.state=3
        R.uart_buf.append(buf)
    elif R.state==3 and buf<50:#数据长度小于50
        R.state=4
        R._data_len=buf  #有效数据长度
        R._data_cnt=buf+5#总数据长度
        R.uart_buf.append(buf)
    elif R.state==4 and R._data_len>0:#存储对应长度数据
        R._data_len=R._data_len-1
        R.uart_buf.append(buf)
        if R._data_len==0:
            R.state=5
    elif R.state==5:
        R.uart_buf.append(buf)
        R.state=0
        Receive_Anl(R.uart_buf,R.uart_buf[3]+5)
#        print(R.uart_buf)
        R.uart_buf=[]#清空缓冲区，准备下次接收数据
    else:
        R.state=0
        R.uart_buf=[]#清空缓冲区，准备下次接收数据

def UartReadBuffer():       #调用的读取函数
    i = 0
    if serial0.available():
        receive_data = bytearray(serial0.read())
        print(receive_data)
        Buffer_size = len(receive_data)
        while i<Buffer_size:
            uart_data_prase(receive_data[i])    #读取单个字符，这里可以，只接受命令
            i = i + 1

# YOLOv8物体检测
def maix_yolo_detection():
    target.flag = 0
    target.type1 = 0  # 重置第一个目标类型
    target.type2 = 0  # 重置第二个目标类型
    img = cam.read()
    
    # 设置摄像头参数
    cam.luma(50)
    cam.constrast(50)
    cam.saturation(100)
    
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
    
    detected_count = 0  # 已检测到的目标计数
    
    for obj in objs:
        if obj.score <= 0.8:
            continue
            
        # 绘制检测框
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED, thickness=8)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED, scale=3)
        
        # 存储前两个检测目标的信息
        if detected_count == 0:
            target.type1 = obj.class_id
            target.loc_1_x = obj.x + obj.w // 2
            target.loc_1_y = obj.y + obj.h // 2
            print("type:", target.type1)
            print("locx:", target.loc_1_x)
            print("locy:", target.loc_1_y)
            target.pixel = obj.w * obj.h  # 只记录第一个目标的面积
            detected_count += 1
        elif detected_count == 1:
            target.type2 = obj.class_id
            target.loc_2_x = obj.x + obj.w // 2
            target.loc_2_y = obj.y + obj.h // 2
            print("type:", target.type2)
            print("locx:", target.loc_2_x)
            print("locy:", target.loc_2_y)
            detected_count += 1
            break  # 只取前两个目标
        
    target.flag = 1 if detected_count > 0 else 0  # 设置标志位
    disp.show(img)


# yolo输入必须为640*640
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
IMAGE_DIS_MAX = IMAGE_DIS_MAX=(int)(math.sqrt(IMAGE_WIDTH*IMAGE_WIDTH+IMAGE_HEIGHT*IMAGE_HEIGHT)/2)

# 初始化YOLOv8检测器
detector = nn.YOLOv8(model="/root/models/best_int8.mud", dual_buff=True)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

set_flag = 0 
ctr.work_mode=0x05  # 修改此处更改使用模式
last_ticks=0
ticks=0
ticks_delta=0

while not app.need_exit():
    if ctr.work_mode==0x00: # 空闲模式
        img = cam.read()
        disp.show(img)
        
    elif ctr.work_mode==0x05: # YOLOv8检测模式
        maix_yolo_detection()

    #数据发送
    target.img_width = IMAGE_WIDTH
    target.img_height = IMAGE_HEIGHT
    data = package_blobs_data(ctr.work_mode)
    serial0.write(bytes(data))
    print("sent:", data)
    

    # 数据接收
    UartReadBuffer()
    time.sleep_ms(100) # sleep 1ms to make CPU free

    last_ticks=ticks
    ticks=time.ticks_ms()#ticks=time.ticks_ms()
                      #新版本OPENMV固件使用time.ticks_ms()
                      #旧版本OPENMV固件使用time.ticks()
    ticks_delta=ticks-last_ticks
    if ticks_delta<1:
        ticks_delta=1
    target.fps=(int)(1000/ticks_delta)
    # print(target.reserved3_int32)