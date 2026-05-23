#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Docker 容器版 OpenClaw Sandbox
使用 Docker SDK 替代 Rock API，直接管理本地容器
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import docker
from docker.models.containers import Container
import aiohttp


class LocalDockerSandboxError(Exception):
    """LocalDockerSandbox 自定义异常"""
    pass


class LocalDockerSandbox:
    """
    基于本地 Docker 容器的 OpenClaw 客户端。

    流程：
    1. 启动本地 Docker 容器（映射随机端口）
    2. 等待容器健康检查通过
    3. 上传 openclaw.json
    4. 调用 /reload-config
    5. 等待 /health = healthy
    6. 通过 /prompt /sessions 等接口操作 OpenClaw
    """

    def __init__(
        self,
        local_config_path: str = "./openclaw.json",
        image: str = "openclaw-cluster:v7",
        timeout: int = 60,
        max_retries: int = 60,
        retry_interval: int = 2,
        container_api_port: int = 9000,
        log_level: int = logging.INFO,
    ):
        self.local_config_path = str(local_config_path)
        self.image = image
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.container_api_port = container_api_port

        self.container: Optional[Container] = None
        self.container_id: Optional[str] = None
        self.host_port: Optional[int] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

        self.docker_client = docker.from_env()
        self.logger = self._build_logger(log_level)

    def _build_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        logger.setLevel(log_level)

        if not logger.handlers:
            log_file = f"/tmp/local_docker_sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.propagate = False
            logger.info(f"日志文件: {log_file}")

        return logger

    async def __aenter__(self):
        await self._ensure_http_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def _ensure_http_session(self):
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    def _find_available_port(self, start: int = 9001, end: int = 9100) -> int:
        """查找可用的主机端口"""
        import socket
        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise LocalDockerSandboxError(f"无法找到可用端口 ({start}-{end})")

    async def start_container(self) -> str:
        """启动本地 Docker 容器"""
        try:
            # 查找可用端口
            self.host_port = self._find_available_port()
            self.logger.info(f"使用主机端口: {self.host_port}")

            # 启动容器
            self.container = self.docker_client.containers.run(
                self.image,
                detach=True,
                ports={f'{self.container_api_port}/tcp': self.host_port},
                remove=False,  # 不自动删除，需要手动清理
                auto_remove=False,
                network_mode='bridge',
                mem_limit='10g',
                cpu_count=8,
            )

            self.container_id = self.container.id[:12]
            self.logger.info(f"容器已启动: {self.container_id}, 端口映射: {self.host_port} -> {self.container_api_port}")
            return self.container_id

        except Exception as e:
            raise LocalDockerSandboxError(f"启动容器失败: {e}")

    async def wait_for_container_ready(self) -> bool:
        """等待容器就绪"""
        if not self.container:
            raise LocalDockerSandboxError("容器未启动")

        for attempt in range(1, self.max_retries + 1):
            self.logger.info(f"等待容器就绪: 第 {attempt}/{self.max_retries} 次检查")

            try:
                self.container.reload()
                status = self.container.status

                if status == 'running':
                    self.logger.info("容器已就绪")
                    return True
                elif status in ['exited', 'dead']:
                    self.logger.error(f"容器启动失败: {status}")
                    return False

            except Exception as e:
                self.logger.warning(f"检查容器状态异常: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_interval)

        self.logger.error("容器启动超时")
        return False

    async def execute_command(self, command: str) -> dict:
        """在容器内执行命令"""
        if not self.container:
            raise LocalDockerSandboxError("容器未启动")

        try:
            exit_code, output = self.container.exec_run(
                cmd=['bash', '-c', command],
                demux=True
            )

            stdout = output[0].decode('utf-8') if output[0] else ''
            stderr = output[1].decode('utf-8') if output[1] else ''

            result = {
                'result': {
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': exit_code
                }
            }

            self.logger.debug(f"命令执行: {command}, exit_code={exit_code}")
            return result

        except Exception as e:
            raise LocalDockerSandboxError(f"执行命令失败: {e}")

    async def upload_file(self, local_file_path: str, target_path: str) -> dict:
        """上传文件到容器"""
        if not self.container:
            raise LocalDockerSandboxError("容器未启动")

        file_path = Path(local_file_path)
        if not file_path.exists():
            raise LocalDockerSandboxError(f"本地文件不存在: {local_file_path}")

        try:
            import tarfile
            import io

            # 创建 tar 归档
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(str(file_path), arcname=Path(target_path).name)

            tar_stream.seek(0)

            # 上传到容器
            target_dir = str(Path(target_path).parent)
            self.container.put_archive(target_dir, tar_stream)

            self.logger.info(f"文件已上传: {local_file_path} -> {target_path}")
            return {'success': True}

        except Exception as e:
            raise LocalDockerSandboxError(f"文件上传失败: {e}")

    async def stop_container(self) -> bool:
        """停止并删除容器"""
        if not self.container:
            self.logger.info("没有容器，无需停止")
            return True

        try:
            self.logger.info(f"停止容器: {self.container_id}")
            self.container.stop(timeout=10)
            self.container.remove(force=True)
            self.logger.info("容器已删除")
            return True
        except Exception as e:
            self.logger.warning(f"停止容器失败: {e}")
            return False

    def _extract_execute_result(self, data: dict) -> tuple[str, str, int]:
        result = data.get("result", {})
        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        exit_code = result.get("exit_code", 1)
        return stdout, stderr, exit_code

    async def _call_container_api(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[dict] = None,
        timeout: int = 60,
    ) -> dict:
        """
        通过 HTTP 调用容器内的 FastAPI 服务
        """
        await self._ensure_http_session()
        url = f"http://localhost:{self.host_port}{endpoint}"

        try:
            if method.upper() == "GET":
                async with self.http_session.get(url, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"GET {endpoint} 失败: {response.status} - {text}")
                    return json.loads(text)

            elif method.upper() == "POST":
                headers = {"Content-Type": "application/json"}
                async with self.http_session.post(url, json=json_data, headers=headers, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"POST {endpoint} 失败: {response.status} - {text}")
                    return json.loads(text)

            elif method.upper() == "DELETE":
                async with self.http_session.delete(url, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"DELETE {endpoint} 失败: {response.status} - {text}")
                    return json.loads(text)
            else:
                raise LocalDockerSandboxError(f"不支持的 HTTP 方法: {method}")

        except asyncio.TimeoutError:
            raise LocalDockerSandboxError(f"API 调用超时: {endpoint}")
        except json.JSONDecodeError as e:
            raise LocalDockerSandboxError(f"API 返回非 JSON: {endpoint}, error={e}")

    async def _call_container_api_text(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[dict] = None,
        timeout: int = 60,
    ) -> str:
        """
        通过 HTTP 调用容器内的 FastAPI 服务（返回纯文本）
        用于 /sessions/{id}/export 等返回 JSONL 的端点
        """
        await self._ensure_http_session()
        url = f"http://localhost:{self.host_port}{endpoint}"

        try:
            if method.upper() == "GET":
                async with self.http_session.get(url, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"GET {endpoint} 失败: {response.status} - {text}")
                    return text

            elif method.upper() == "POST":
                headers = {"Content-Type": "application/json"}
                async with self.http_session.post(url, json=json_data, headers=headers, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"POST {endpoint} 失败: {response.status} - {text}")
                    return text

            else:
                raise LocalDockerSandboxError(f"不支持的 HTTP 方法: {method}")

        except asyncio.TimeoutError:
            raise LocalDockerSandboxError(f"API 调用超时: {endpoint}")

    async def _wait_fastapi_up(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if "status" not in data:
            raise LocalDockerSandboxError(f"/health 返回异常: {data}")
        return data

    async def _wait_service_healthy(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if data.get("status") != "healthy":
            raise LocalDockerSandboxError(f"服务尚未 healthy: {data}")
        return data

    async def _retry(
        self,
        func,
        action_name: str,
        retries: Optional[int] = None,
        interval: Optional[int] = None,
    ) -> Any:
        retries = retries if retries is not None else self.max_retries
        interval = interval if interval is not None else self.retry_interval
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                self.logger.info(f"{action_name} - 尝试 {attempt}/{retries}")
                return await func()
            except Exception as e:
                last_error = e
                self.logger.warning(f"{action_name} 失败: {e}")
                if attempt < retries:
                    await asyncio.sleep(interval)

        raise LocalDockerSandboxError(f"{action_name} 最终失败: {last_error}")

    async def initialize(self):
        """
        初始化流程：
        1. 启动容器
        2. 等待容器就绪
        3. 等待 FastAPI 服务可访问
        4. 上传 openclaw.json
        5. 调用 /reload-config
        6. 等待 /health = healthy
        """
        await self._ensure_http_session()

        if self._initialized:
            self.logger.info("已经初始化过，跳过 initialize")
            return

        self.logger.info("开始初始化本地 Docker OpenClaw sandbox")

        # 1. 启动容器
        await self._retry(self.start_container, "启动容器")

        # 2. 等待容器就绪
        ready = await self.wait_for_container_ready()
        if not ready:
            raise LocalDockerSandboxError("容器未能成功启动")

        # 3. 等待 FastAPI 服务可访问
        await self._retry(
            self._wait_fastapi_up,
            "等待 FastAPI 服务启动",
            retries=30,
            interval=3,
        )

        # 4. 准备配置目录
        await self._retry(
            lambda: self.execute_command("mkdir -p /root/.openclaw"),
            "准备配置目录",
            retries=3,
            interval=1,
        )

        # 5. 上传配置文件
        await self._retry(
            lambda: self.upload_file(self.local_config_path, "/root/.openclaw/openclaw.json"),
            "上传 openclaw.json",
            retries=5,
            interval=2,
        )

        # 6. 验证配置文件
        await self._retry(
            lambda: self.execute_command("ls -l /root/.openclaw/openclaw.json"),
            "验证 openclaw.json",
            retries=3,
            interval=1,
        )

        # 7. reload 配置
        reload_result = await self._retry(
            lambda: self._call_container_api("/reload-config", "POST", timeout=30),
            "重载 OpenClaw 配置",
            retries=5,
            interval=3,
        )
        self.logger.info(f"配置重载响应: {json.dumps(reload_result, ensure_ascii=False)}")

        # 8. 等待服务完全 healthy
        health_data = await self._retry(
            self._wait_service_healthy,
            "等待 OpenClaw 服务 healthy",
            retries=10,
            interval=2,
        )
        self.logger.info(f"健康检查通过: {json.dumps(health_data, ensure_ascii=False)}")

        self._initialized = True
        self.logger.info("本地 Docker OpenClaw sandbox 初始化完成")

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        timeout: int = 600
    ) -> dict:
        """通过 FastAPI /prompt 发送消息"""
        if not self._initialized:
            raise LocalDockerSandboxError("请先调用 initialize()")

        message = (message or "").strip()
        if not message:
            raise LocalDockerSandboxError("消息不能为空")

        payload = {
            "prompt": message,
            "timeout": timeout,
        }
        if session_id:
            payload["session_id"] = session_id

        self.logger.info(f"发送消息到 OpenClaw: {message[:200]}")

        result = await self._retry(
            lambda: self._call_container_api("/prompt", "POST", payload, timeout=timeout + 30),
            "发送消息到 OpenClaw",
            retries=3,
            interval=2,
        )

        # 兼容服务端 response 内嵌 JSON 字符串的情况
        raw_response = result.get("response")
        if isinstance(raw_response, str):
            try:
                result["response_json"] = json.loads(raw_response)
            except Exception:
                result["response_json"] = None

        self.logger.info(
            f"消息发送完成: session_id={result.get('session_id')}, status={result.get('status')}"
        )
        return result

    async def list_sessions(self) -> dict:
        """列出所有会话"""
        if not self._initialized:
            raise LocalDockerSandboxError("请先调用 initialize()")

        return await self._retry(
            lambda: self._call_container_api("/sessions", "GET", timeout=20),
            "列出所有会话",
            retries=3,
            interval=2,
        )

    async def export_session_jsonl(self, session_id: str, local_dir: str = "./exports_local") -> str:
        """导出指定 session 的 jsonl 到本地"""
        if not self._initialized:
            raise LocalDockerSandboxError("请先调用 initialize()")

        if not session_id:
            raise LocalDockerSandboxError("session_id 不能为空")

        output_dir = Path(local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        endpoint = f"/sessions/{session_id}/export"

        async def _export():
            # 使用 _call_container_api_text 因为返回的是 JSONL 文本
            return await self._call_container_api_text(endpoint, "GET", timeout=60)

        # 调用 API 获取 jsonl 内容（纯文本）
        file_content = await self._retry(
            _export,
            f"导出会话 {session_id}",
            retries=3,
            interval=2,
        )

        # 直接写入本地文件
        local_path = output_dir / f"session_{session_id}.jsonl"
        local_path.write_text(file_content, encoding="utf-8")
        self.logger.info(f"已导出到本地: {local_path}")
        return str(local_path)

    async def delete_session(self, session_id: str) -> dict:
        """删除指定会话"""
        if not self._initialized:
            raise LocalDockerSandboxError("请先调用 initialize()")

        if not session_id:
            raise LocalDockerSandboxError("session_id 不能为空")

        return await self._retry(
            lambda: self._call_container_api(f"/sessions/{session_id}", "DELETE", timeout=20),
            f"删除会话 {session_id}",
            retries=3,
            interval=2,
        )

    async def stop(self):
        """停止容器并释放资源"""
        if self.container_id:
            await self._retry(
                self.stop_container,
                "停止容器",
                retries=3,
                interval=2,
            )

        self.container = None
        self.container_id = None
        self.host_port = None
        self._initialized = False
        self.logger.info("资源已释放")
