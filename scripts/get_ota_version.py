import json
import requests
import logging

OTA_VERSION_URL = 'https://api.tenclass.net/xiaozhi/ota/'
# 替换为你的实际 MAC 地址
MAC_ADDR = 'dc:a6:32:9a:60:19'
mqtt_info = {}

def get_ota_version():
    global mqtt_info
    header = {
        'Device-Id': MAC_ADDR,
        'Content-Type': 'application/json'
    }
    post_data = {"flash_size": 16777216, "minimum_free_heap_size": 8318916, "mac_address": f"{MAC_ADDR}",
                 "chip_model_name": "esp32s3", "chip_info": {"model": 9, "cores": 2, "revision": 2, "features": 18},
                 "application": {"name": "xiaozhi", "version": "0.9.9", "compile_time": "Jan 22 2025T20:40:23Z",
                                 "idf_version": "v5.3.2-dirty",
                                 "elf_sha256": "22986216df095587c42f8aeb06b239781c68ad8df80321e260556da7fcf5f522"},
                 "partition_table": [{"label": "nvs", "type": 1, "subtype": 2, "address": 36864, "size": 16384},
                                     {"label": "otadata", "type": 1, "subtype": 0, "address": 53248, "size": 8192},
                                     {"label": "phy_init", "type": 1, "subtype": 1, "address": 61440, "size": 4096},
                                     {"label": "model", "type": 1, "subtype": 130, "address": 65536, "size": 983040},
                                     {"label": "storage", "type": 1, "subtype": 130, "address": 1048576,
                                      "size": 1048576},
                                     {"label": "factory", "type": 0, "subtype": 0, "address": 2097152, "size": 4194304},
                                     {"label": "ota_0", "type": 0, "subtype": 16, "address": 6291456, "size": 4194304},
                                     {"label": "ota_1", "type": 0, "subtype": 17, "address": 10485760,
                                      "size": 4194304}],
                 "ota": {"label": "factory"},
                 "board": {"type": "bread-compact-wifi", "ssid": "mzy", "rssi": -58, "channel": 6,
                           "ip": "192.168.31.239", "mac": "dc:a6:32:9a:60:19"}}

    response = requests.post(OTA_VERSION_URL, headers=header, data=json.dumps(post_data))
    print('=========================')
    print(response.text)
    logging.info(f"get version: {response}")
    mqtt_info = response.json()['mqtt']
    print(mqtt_info)

# 调用函数获取 MQTT 信息
get_ota_version()

# 使用获取到的 MQTT 信息
print("获取到的 MQTT 信息:", mqtt_info)
