import configparser

import argparse

argparser = argparse.ArgumentParser(
        description='Agent Simulation')
argparser.add_argument(
    '-t', '--token',
    help='your personal github access token')
args = argparser.parse_args()

# 替换为你的个人访问令牌（可选，但推荐）
TOKEN = args.token

config = configparser.ConfigParser()
# 使用UTF-8编码读取带有中文的配置文件config.ini
config.read('config.ini', encoding="utf-8")

# 获取配置文件中的值
carla_home = config.get('carla', 'home')

print("Test passed!")

