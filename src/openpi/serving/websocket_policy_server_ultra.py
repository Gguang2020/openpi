import time
import asyncio
import websockets
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from .websocket_ultra_config import *

mp_lock = mp.Lock()

class SharedDataBuffer:
    """共享内存缓冲区"""
    def __init__(self, data_shape: tuple, dtype: np.dtype):
        self.data_shape = data_shape
        self.dtype = dtype
        self.size = np.prod(data_shape) * np.dtype(dtype).itemsize

        # 创建两个共享内存块（双缓冲）
        self.shm = SharedMemory(create=True, size=self.size)
        self.buffer = np.ndarray(data_shape, dtype=dtype, buffer=self.shm.buf)

    def write(self, data: np.ndarray):
        """写入数据"""
        np.copyto(self.buffer, data)

    def read(self) -> np.ndarray:
        """读取数据"""
        return self.buffer.copy()  # 返回拷贝以避免外部修改

    def close(self):
        """释放共享内存"""
        self.shm.close()
        self.shm.unlink()

class SubAndPubWorker(mp.Process):
    def __init__(self, host: str, port: int,
                 input_buffer_head: SharedDataBuffer, input_buffer_left: SharedDataBuffer,
                 input_buffer_right: SharedDataBuffer, input_buffer_state: SharedDataBuffer,
                 input_buffer_prompt: SharedDataBuffer, input_buffer_timestamp: SharedDataBuffer,
                 output_buffer_timestamp: SharedDataBuffer, output_buffer_action: SharedDataBuffer):
        """初始化推理工作进程"""
        super().__init__()
        self.clients = set()  # 存储所有连接的客户端
        self.sender_clients = set()  # 存储发送者客户端
        self.receiver_clients = set()  # 存储接收者客户端
        self._host = host
        self._port = port
        self.input_buffer_head = input_buffer_head
        self.input_buffer_left = input_buffer_left
        self.input_buffer_right = input_buffer_right
        self.input_buffer_state = input_buffer_state
        self.input_buffer_prompt = input_buffer_prompt
        self.input_buffer_timestamp = input_buffer_timestamp
        self.output_buffer_timestamp = output_buffer_timestamp
        self.output_buffer_action = output_buffer_action

    def run(self):
        asyncio.run(self.run_self())

    async def run_self(self):
        await self.start_self()
        await asyncio.Future()  # run forever

    async def start_self(self):
        asyncio.create_task(self._process_messages())
        server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=100 * 1024 * 1024,  # 设置为100MB
            max_queue=10  # 同时增加队列大小
        )
        return server

    async def _process_messages(self):
        """从共享内存获取推理结果并广播给接收者"""
        last_timestamp = 0.0
        while True:
            # 读取结果
            timestamp = self.output_buffer_timestamp.read()
            action = self.output_buffer_action.read()

            if timestamp is not None and action is not None:
                if timestamp[0] != last_timestamp:
                    # 准备响应数据
                    response = {
                        "obs_timestamp": float(timestamp[0]),   # 转为Python float
                        "actions": action                        # 转为Python list
                    }
                    packed_response = msgpack_numpy.packb(response)

                    for receiver in self.receiver_clients:
                        try:
                            await receiver.send(packed_response)
                        except:
                            continue
                    last_timestamp = timestamp[0]  # 更新最后发送的时间戳

            await asyncio.sleep(0.001)  # 避免空转

    async def _handler(self, websocket):
        # 为新连接分配唯一ID
        client_id = id(websocket)
        self.clients.add(websocket)
        print(f"New client connected: {client_id}")

        try:
            # 接收客户端初始消息以确定角色
            initial_msg = await websocket.recv()
            role_msg = msgpack_numpy.unpackb(initial_msg)

            if role_msg.get("role") == "sender":
                self.sender_clients.add(websocket)
                print(f"Client {client_id} registered as sender")

                # 处理发送者的消息
                while True:
                    try:
                        obs_recv = await websocket.recv()
                        obs = msgpack_numpy.unpackb(obs_recv)

                        if np.isnan(obs["state"][0]):
                            continue

                        # 将数据写入共享内存（带锁）
                        with mp_lock:
                            self.input_buffer_state.write(np.array(obs["state"], dtype=np.float64))
                            self.input_buffer_head.write(np.array(obs["images"]["cam_high"], dtype=np.float32))
                            self.input_buffer_left.write(np.array(obs["images"]["cam_left_wrist"], dtype=np.float32))
                            self.input_buffer_right.write(np.array(obs["images"]["cam_right_wrist"], dtype=np.float32))
                            self.input_buffer_timestamp.write(obs["obs_timestamp"])
                            if USE_PROMPT:
                                # 将字符串转换为numpy数组
                                prompt_str = obs["prompt"]
                                prompt_bytes = prompt_str.encode('utf-8')[:PROMPT_SIZE-1]  # 留一个位置给\0
                                prompt_arr = np.zeros(PROMPT_SIZE, dtype=np.uint8)
                                prompt_arr[:len(prompt_bytes)] = np.frombuffer(prompt_bytes, dtype=np.uint8)
                                prompt_arr[len(prompt_bytes)] = 0
                                self.input_buffer_prompt.write(prompt_arr)

                    except websockets.ConnectionClosed:
                        print(f"Sender {client_id} disconnected")
                        break
                        # await asyncio.sleep(0.005)
                    except Exception as e:
                        print(f"Error with sender {client_id}: ", e)
                        await asyncio.sleep(0.005)

            elif role_msg.get("role") == "receiver":
                self.receiver_clients.add(websocket)
                print(f"Client {client_id} registered as receiver")

                # 保持连接，等待接收消息
                while True:
                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error with client {client_id}: {str(e)}")

        finally:
            self.clients.discard(websocket)
            self.sender_clients.discard(websocket)
            self.receiver_clients.discard(websocket)


class WebsocketPolicyServerAsynchronous:

    def __init__(self, policy: _base_policy.BasePolicy, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._policy = policy

        # 共享内存缓冲区（假设图像尺寸为 640x480x3 的 uint8 数组）
        self.input_buffer_head = SharedDataBuffer(HEAD_IMG_SIZE, np.float32)
        self.input_buffer_left = SharedDataBuffer(LEFT_IMG_SIZE, np.float32)
        self.input_buffer_right = SharedDataBuffer(RIGHT_IMG_SIZE, np.float32)
        self.input_buffer_state = SharedDataBuffer(STATE_SIZE, np.float64)
        self.input_buffer_prompt = SharedDataBuffer((PROMPT_SIZE,), np.uint8)
        self.input_buffer_timestamp = SharedDataBuffer((1, ), np.float64)
        self.output_buffer_timestamp = SharedDataBuffer((1, ), np.float64)
        self.output_buffer_action = SharedDataBuffer(ACTION_SIZE, np.float64)

        # 启动子进程
        self.worker = SubAndPubWorker(
            host=host,
            port=port,
            input_buffer_head=self.input_buffer_head,
            input_buffer_left=self.input_buffer_left,
            input_buffer_right=self.input_buffer_right,
            input_buffer_state=self.input_buffer_state,
            input_buffer_prompt=self.input_buffer_prompt,
            input_buffer_timestamp=self.input_buffer_timestamp,
            output_buffer_timestamp=self.output_buffer_timestamp,
            output_buffer_action=self.output_buffer_action
        )
        self.worker.start()
        self.run_infer()

    def run_infer(self):
        last_obs_timestamp = None
        obs_dict = {
            "images": {
                "cam_high": None,
                "cam_left_wrist": None,
                "cam_right_wrist": None
            },
            "obs_timestamp": None
        }
        while True:
            with mp_lock:
                obs_head, obs_left, obs_right, obs_state, obs_timestamp =\
                    self.input_buffer_head.read(), self.input_buffer_left.read(),\
                    self.input_buffer_right.read(), self.input_buffer_state.read(),\
                    self.input_buffer_timestamp.read()
                if USE_PROMPT:
                    obs_prompt = self.input_buffer_prompt.read()

            if any(x is None for x in [obs_head, obs_left, obs_right, obs_timestamp]):
                time.sleep(0.001)  # 避免空转
                continue

            if last_obs_timestamp is None:
                last_obs_timestamp = obs_timestamp[0]

            if obs_timestamp[0] != last_obs_timestamp:
                last_obs_timestamp = obs_timestamp[0]
                # obs_dict["obs_timestamp"] = obs_timestamp[0]
                obs_dict["state"] = obs_state.reshape(-1)  # 假设状态是一个一维数组
                obs_dict["images"]["cam_high"] = obs_head.reshape(HEAD_IMG_SIZE)
                obs_dict["images"]["cam_left_wrist"] = obs_left.reshape(LEFT_IMG_SIZE)
                obs_dict["images"]["cam_right_wrist"] = obs_right.reshape(RIGHT_IMG_SIZE)

                if USE_PROMPT:
                    prompt_str = bytes(obs_prompt).decode('utf-8').split('\0')[0]
                    obs_dict["prompt"] = prompt_str

                action = self._policy.infer(obs_dict)    # CPU 密集型任务

                self.output_buffer_timestamp.write(obs_timestamp[0])
                self.output_buffer_action.write(action["actions"])
            else:
                time.sleep(0.001)

class BasePolicy(_base_policy.BasePolicy):
    def __init__(self):
        pass

    def infer(self, obs_dict):
        """模拟推理方法，返回一个动作"""
        time.sleep(0.042)
        action = np.random.rand(50, 31).astype(np.float64)
        return {"actions": action}

if __name__ == "__main__":
    policy = BasePolicy()  # 假设你有一个 BasePolicy 类
    print("Starting server...")
    server = WebsocketPolicyServerAsynchronous(policy=policy, host="0.0.0.0", port=8000)
