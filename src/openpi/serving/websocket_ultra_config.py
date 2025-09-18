
HEAD_IMG_SIZE = (3, 224, 224)   # 假设头部图像尺寸为 224x224x3
LEFT_IMG_SIZE = (3, 224, 224)   # 假设左手图像尺寸为 224x224x3
RIGHT_IMG_SIZE = (3, 224, 224)  # 假设右手图像尺寸为 224x224x3
STATE_SIZE = (31, )             # 假设状态尺寸为 (31, 1)
ACTION_SIZE = (50, 31)          # 假设动作尺寸为 (50, 31)
USE_PROMPT = True
PROMPT_SIZE = 128               # prompt长度