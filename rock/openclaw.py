#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any

import aiohttp


class OpenClawSandboxError(Exception):
    """OpenClaw Sandbox 自定义异常"""
    pass


class OpenClawSandbox:
    """
    基于 ROCK Sandbox + 容器内 FastAPI 的 OpenClaw 客户端。

    流程：
    1. 启动 sandbox
    2. 等待 sandbox ready
    3. 通过 ROCK execute 在容器内 curl localhost:9000 检查 FastAPI
    4. 上传 openclaw.json
    5. 调用 /reload-config
    6. 通过 /prompt /sessions 等接口操作 OpenClaw
    """

    def __init__(
        self,
        api_key: str,
        user_id: str,
        experiment_id: str,
        local_config_path: str = "./openclaw.json",
        api_base_url: str = "http://xrl.alibaba-inc.com",
        image: str = "rex-registry.cn-hangzhou.cr.aliyuncs.com/chatos/openclaw-cluster:v6",
        auto_clear_time_minutes: int = 30,
        cpus: str = "8",
        memory: str = "10g",
        timeout: int = 60,
        max_retries: int = 60,
        retry_interval: int = 2,
        container_api_port: int = 9000,
        log_level: int = logging.INFO,
    ):
        self.api_key = api_key
        self.user_id = user_id
        self.experiment_id = experiment_id
        self.local_config_path = str(local_config_path)
        self.api_base_url = api_base_url.rstrip("/")
        self.image = image
        self.auto_clear_time_minutes = auto_clear_time_minutes
        self.cpus = cpus
        self.memory = memory
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.container_api_port = container_api_port

        self.sandbox_id: Optional[str] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

        self.logger = self._build_logger(log_level)

    def _build_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        logger.setLevel(log_level)

        if not logger.handlers:
            log_file = f"/tmp/openclaw_sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def _headers_json(self) -> dict:
        return {
            "Content-Type": "application/json",
            "XRL-Authorization": f"Bearer {self.api_key}",
            "x-user-id": self.user_id,
            "x-experiment-id": self.experiment_id,
        }

    def _headers_form(self) -> dict:
        return {
            "XRL-Authorization": f"Bearer {self.api_key}",
            "x-user-id": self.user_id,
            "x-experiment-id": self.experiment_id,
        }

    async def _retry(
        self,
        func: Callable[[], Awaitable[Any]],
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

        raise OpenClawSandboxError(f"{action_name} 最终失败: {last_error}")

    async def _post_json(self, path: str, payload: dict) -> dict:
        await self._ensure_http_session()
        url = f"{self.api_base_url}{path}"

        async with self.http_session.post(
            url,
            headers=self._headers_json(),
            json=payload
        ) as response:
            text = await response.text()
            if response.status != 200:
                raise OpenClawSandboxError(f"POST {path} 失败: {response.status} - {text}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                raise OpenClawSandboxError(f"POST {path} 返回非 JSON: {text}")

    async def _get_json(self, path: str, params: dict) -> dict:
        await self._ensure_http_session()
        url = f"{self.api_base_url}{path}"

        async with self.http_session.get(
            url,
            headers=self._headers_json(),
            params=params
        ) as response:
            text = await response.text()
            if response.status != 200:
                raise OpenClawSandboxError(f"GET {path} 失败: {response.status} - {text}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                raise OpenClawSandboxError(f"GET {path} 返回非 JSON: {text}")

    async def start_sandbox(self) -> str:
        payload = {
            "image": self.image,
            "auto_clear_time_minutes": self.auto_clear_time_minutes,
            "cpus": self.cpus,
            "memory": self.memory,
        }
        data = await self._post_json("/apis/envs/sandbox/v1/start_async", payload)
        self.logger.info(f"启动响应: {json.dumps(data, ensure_ascii=False)}")

        sandbox_id = data.get("result", {}).get("sandbox_id")
        if not sandbox_id:
            raise OpenClawSandboxError("启动成功但未获取到 sandbox_id")

        self.sandbox_id = sandbox_id
        self.logger.info(f"获取到 sandbox_id: {sandbox_id}")
        return sandbox_id

    async def check_sandbox_status(self, sandbox_id: Optional[str] = None) -> dict:
        sandbox_id = sandbox_id or self.sandbox_id
        if not sandbox_id:
            raise OpenClawSandboxError("check_sandbox_status 时 sandbox_id 为空")

        data = await self._get_json(
            "/apis/envs/sandbox/v1/get_status",
            {"sandbox_id": sandbox_id},
        )
        self.logger.info(f"状态响应: {json.dumps(data, ensure_ascii=False)}")
        return data

    async def wait_for_sandbox_ready(self, sandbox_id: Optional[str] = None) -> bool:
        sandbox_id = sandbox_id or self.sandbox_id
        if not sandbox_id:
            raise OpenClawSandboxError("wait_for_sandbox_ready 时 sandbox_id 为空")

        # 增加重试次数：120 次 × 3 秒 = 最多 6 分钟等待
        max_wait_retries = 120
        wait_interval = 3  # 固定 3 秒间隔，线性等待

        for attempt in range(1, max_wait_retries + 1):
            self.logger.info(f"等待沙箱就绪: 第 {attempt}/{max_wait_retries} 次检查 (每 {wait_interval}s 检查一次)")

            try:
                data = await self.check_sandbox_status(sandbox_id)
                result = data.get("result", {})
                state = result.get("state", "unknown")
                is_alive = result.get("is_alive", False)

                self.logger.info(f"沙箱状态解析: state={state}, is_alive={is_alive}")

                if is_alive or state in ("running", "ready"):
                    self.logger.info(f"沙箱已就绪 (耗时约 {attempt * wait_interval} 秒)")
                    return True

                if state == "failed":
                    self.logger.error("沙箱启动失败（state=failed）")
                    return False

            except Exception as e:
                self.logger.warning(f"检查沙箱状态异常: {e}")

            if attempt < max_wait_retries:
                await asyncio.sleep(wait_interval)  # 使用 asyncio.sleep 而不是 time.sleep

        self.logger.error(f"沙箱启动超时 (等待了 {max_wait_retries * wait_interval} 秒)")
        return False

    async def execute_command(
        self,
        command: str,
        sandbox_id: Optional[str] = None,
        shell: bool = True
    ) -> dict:
        sandbox_id = sandbox_id or self.sandbox_id
        if not sandbox_id:
            raise OpenClawSandboxError("execute_command 时 sandbox_id 为空")

        payload = {
            "sandbox_id": sandbox_id,
            "command": command,
            "shell": shell,
        }
        data = await self._post_json("/apis/envs/sandbox/v1/execute", payload)
        self.logger.info(f"命令执行响应: {json.dumps(data, ensure_ascii=False)}")
        return data

    async def upload_file(
        self,
        local_file_path: str,
        target_path: str,
        sandbox_id: Optional[str] = None
    ) -> dict:
        sandbox_id = sandbox_id or self.sandbox_id
        if not sandbox_id:
            raise OpenClawSandboxError("upload_file 时 sandbox_id 为空")

        file_path = Path(local_file_path)
        if not file_path.exists():
            raise OpenClawSandboxError(f"本地文件不存在: {local_file_path}")

        await self._ensure_http_session()
        url = f"{self.api_base_url}/apis/envs/sandbox/v1/upload"

        with open(file_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=file_path.name)
            form.add_field("target_path", target_path)
            form.add_field("sandbox_id", sandbox_id)

            async with self.http_session.post(
                url,
                headers=self._headers_form(),
                data=form
            ) as response:
                text = await response.text()
                if response.status != 200:
                    raise OpenClawSandboxError(f"文件上传失败: {response.status} - {text}")
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    raise OpenClawSandboxError(f"上传返回非 JSON: {text}")

        self.logger.info(f"文件上传响应: {json.dumps(data, ensure_ascii=False)}")
        return data

    async def stop_sandbox(self, sandbox_id: Optional[str] = None) -> bool:
        sandbox_id = sandbox_id or self.sandbox_id
        if not sandbox_id:
            self.logger.info("没有 sandbox_id，无需停止")
            return True

        try:
            payload = {"sandbox_id": sandbox_id}
            data = await self._post_json("/apis/envs/sandbox/v1/stop", payload)
            self.logger.info(f"停止沙箱响应: {json.dumps(data, ensure_ascii=False)}")
            return True
        except Exception as e:
            self.logger.warning(f"停止沙箱失败: {e}")
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
        通过 ROCK execute 在容器内调用 localhost FastAPI。
        """
        url = f"http://127.0.0.1:{self.container_api_port}{endpoint}"
        timeout += 30
        if method.upper() == "GET":
            cmd = f"curl -sS --max-time {timeout} -f '{url}'"
        elif method.upper() == "POST":
            if json_data is not None:
                json_str = json.dumps(json_data, ensure_ascii=False).replace("'", "'\\''")
                cmd = (
                    f"curl -sS --max-time {timeout} -f -X POST '{url}' "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{json_str}'"
                )
            else:
                cmd = f"curl -sS --max-time {timeout} -f -X POST '{url}'"
        elif method.upper() == "DELETE":
            cmd = f"curl -sS --max-time {timeout} -f -X DELETE '{url}'"
        else:
            raise OpenClawSandboxError(f"不支持的 HTTP 方法: {method}")

        exec_data = await self.execute_command(cmd)
        stdout, stderr, exit_code = self._extract_execute_result(exec_data)

        self.logger.info(f"[容器API] {method} {endpoint}")
        self.logger.info(f"[容器API] cmd={cmd}")
        self.logger.info(f"[容器API] exit_code={exit_code}")
        if stdout:
            self.logger.info(f"[容器API] stdout={stdout[:1000]}")
        if stderr:
            self.logger.warning(f"[容器API] stderr={stderr[:1000]}")

        if exit_code != 0:
            raise OpenClawSandboxError(
                f"容器 API 调用失败: endpoint={endpoint}, exit_code={exit_code}, stderr={stderr}, stdout={stdout}"
            )

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            raise OpenClawSandboxError(
                f"容器 API 返回非 JSON: endpoint={endpoint}, stdout={stdout}, error={e}"
            )

    async def _wait_fastapi_up(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if "status" not in data:
            raise OpenClawSandboxError(f"/health 返回异常: {data}")
        return data

    async def _wait_service_healthy(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if data.get("status") != "healthy":
            raise OpenClawSandboxError(f"服务尚未 healthy: {data}")
        return data

    async def _start_container_fastapi_server(self) -> dict:
        """
        在 ROCK 容器内手动后台启动 FastAPI 服务。
        不依赖 Docker ENTRYPOINT。
        """
        cmd = (
            "mkdir -p /tmp && "
            "nohup python3 /app/cluster-api-server.py "
            ">/tmp/cluster-api.log 2>&1 & "
            "echo FASTAPI_SERVER_STARTED"
        )

        exec_data = await self.execute_command(cmd)
        stdout, stderr, exit_code = self._extract_execute_result(exec_data)

        self.logger.info("[手动启动 FastAPI] cmd=%s", cmd)
        self.logger.info("[手动启动 FastAPI] exit_code=%s", exit_code)
        if stdout:
            self.logger.info("[手动启动 FastAPI] stdout=%s", stdout[:1000])
        if stderr:
            self.logger.warning("[手动启动 FastAPI] stderr=%s", stderr[:1000])

        if exit_code != 0:
            raise OpenClawSandboxError(
                f"手动启动 FastAPI 服务失败: exit_code={exit_code}, stderr={stderr}, stdout={stdout}"
            )

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }

    async def _read_container_fastapi_log(self) -> str:
        """
        读取容器内 FastAPI 服务日志，便于排查启动失败原因。
        """
        exec_data = await self.execute_command("cat /tmp/cluster-api.log || true")
        stdout, _, _ = self._extract_execute_result(exec_data)
        return stdout


    async def initialize(self):
        """
        初始化流程：
        1. 启动 sandbox
        2. 等待 sandbox ready
        3. 在容器内手动启动 FastAPI 服务（不依赖 Docker ENTRYPOINT）
        4. 等待 FastAPI /health 可访问
        5. 上传 openclaw.json
        6. 调用 /reload-config
        7. 等待 /health = healthy
        """
        await self._ensure_http_session()

        if self._initialized:
            self.logger.info("已经初始化过，跳过 initialize")
            return

        self.logger.info("开始初始化 OpenClaw sandbox (FastAPI 模式)")

        # 1. 启动 sandbox
        await self._retry(self.start_sandbox, "启动沙箱")

        # 2. 等待 sandbox ready
        ready = await self.wait_for_sandbox_ready(self.sandbox_id)
        if not ready:
            raise OpenClawSandboxError("沙箱未能成功 ready，初始化失败")

        # 3. 先确认关键文件是否存在
        await self._retry(
            lambda: self.execute_command("ls -lah /app && ls -lah /app/cluster-api-server.py"),
            "检查 FastAPI 服务脚本是否存在",
            retries=3,
            interval=1,
        )

        # 4. 手动启动 FastAPI 服务
        await self._retry(
            self._start_container_fastapi_server,
            "手动启动容器内 FastAPI 服务",
            retries=3,
            interval=2,
        )

        # 5. 等待 FastAPI 可访问（增加重试次数和间隔）
        try:
            await self._retry(
                self._wait_fastapi_up,
                "等待容器内 FastAPI 服务启动",
                retries=40,  # 40 次 × 5 秒 = 最多 3 分钟多
                interval=5,  # 线性等待，每 5 秒检查一次
            )
        except Exception as e:
            fastapi_log = await self._read_container_fastapi_log()
            self.logger.error("FastAPI 服务启动失败，容器内日志如下：\n%s", fastapi_log)
            raise OpenClawSandboxError(f"FastAPI 服务启动失败: {e}")

        # 6. 准备配置目录
        await self._retry(
            lambda: self.execute_command("mkdir -p /root/.openclaw && ls -lah /root/.openclaw"),
            "准备配置目录",
            retries=3,
            interval=1,
        )

        # 7. 上传配置文件（增加重试次数）
        await self._retry(
            lambda: self.upload_file(self.local_config_path, "/root/.openclaw/openclaw.json"),
            "上传 openclaw.json",
            retries=8,
            interval=3,
        )

        # 8. 验证配置文件
        await self._retry(
            lambda: self.execute_command("ls -l /root/.openclaw/openclaw.json && head -n 5 /root/.openclaw/openclaw.json"),
            "验证 openclaw.json",
            retries=3,
            interval=1,
        )

        # 9. reload 配置（增加重试和超时）
        reload_result = await self._retry(
            lambda: self._call_container_api("/reload-config", "POST", timeout=60),
            "重载 OpenClaw 配置",
            retries=8,
            interval=4,
        )
        self.logger.info(f"配置重载响应: {json.dumps(reload_result, ensure_ascii=False)}")

        # 10. 等待服务完全 healthy（增加重试次数）
        health_data = await self._retry(
            self._wait_service_healthy,
            "等待 OpenClaw 服务 healthy",
            retries=20,  # 20 次 × 3 秒 = 最多 1 分钟
            interval=3,
        )
        self.logger.info(f"健康检查通过: {json.dumps(health_data, ensure_ascii=False)}")

        self._initialized = True
        self.logger.info("OpenClaw sandbox 初始化完成")

        """
        初始化流程：
        1. 启动 sandbox
        2. 等待 sandbox ready
        3. 等待容器内 FastAPI 服务可访问
        4. 上传 openclaw.json
        5. 调用 /reload-config
        6. 等待 /health = healthy
        """
        await self._ensure_http_session()

        if self._initialized:
            self.logger.info("已经初始化过，跳过 initialize")
            return

        self.logger.info("开始初始化 OpenClaw sandbox (FastAPI 模式)")

        await self._retry(self.start_sandbox, "启动沙箱")

        ready = await self.wait_for_sandbox_ready(self.sandbox_id)
        if not ready:
            raise OpenClawSandboxError("沙箱未能成功 ready，初始化失败")

        # 等待 FastAPI 至少启动可访问
        await self._retry(
            self._wait_fastapi_up,
            "等待容器内 FastAPI 服务启动",
            retries=20,
            interval=3,
        )

        # 可选调试：看一下目录
        await self._retry(
            lambda: self.execute_command("mkdir -p /root/.openclaw && ls -lah /root/.openclaw"),
            "准备配置目录",
            retries=3,
            interval=1,
        )

        # 上传配置文件
        await self._retry(
            lambda: self.upload_file(self.local_config_path, "/root/.openclaw/openclaw.json"),
            "上传 openclaw.json",
            retries=5,
            interval=2,
        )

        # 可选调试：确认文件存在
        await self._retry(
            lambda: self.execute_command("ls -l /root/.openclaw/openclaw.json && head -n 5 /root/.openclaw/openclaw.json"),
            "验证 openclaw.json",
            retries=3,
            interval=1,
        )

        # reload 配置
        reload_result = await self._retry(
            lambda: self._call_container_api("/reload-config", "POST", timeout=30),
            "重载 OpenClaw 配置",
            retries=5,
            interval=3,
        )
        self.logger.info(f"配置重载响应: {json.dumps(reload_result, ensure_ascii=False)}")

        # 等到健康
        health_data = await self._retry(
            self._wait_service_healthy,
            "等待 OpenClaw 服务 healthy",
            retries=10,
            interval=2,
        )
        self.logger.info(f"健康检查通过: {json.dumps(health_data, ensure_ascii=False)}")

        self._initialized = True
        self.logger.info("OpenClaw sandbox 初始化完成")

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        timeout: int = 600
    ) -> dict:
        """
        通过 FastAPI /prompt 向 OpenClaw 发送消息。
        """
        if not self._initialized:
            raise OpenClawSandboxError("请先调用 initialize()")

        message = (message or "").strip()
        if not message:
            raise OpenClawSandboxError("消息不能为空")

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
            retries=5,  # 增加重试次数
            interval=3,
        )

        # 兼容服务端 response 内嵌 JSON 字符串的情况
        raw_response = result.get("response")
        if isinstance(raw_response, str):
            try:
                result["response_json"] = json.loads(raw_response)
            except Exception:
                result["response_json"] = None

        self.logger.info(
            f"消息发送完成: session_id={result.get('session_id')}, status={result.get('status')}, error={result.get('error')}"
        )
        return result

    async def list_sessions(self) -> dict:
        """
        列出所有会话。
        """
        if not self._initialized:
            raise OpenClawSandboxError("请先调用 initialize()")

        return await self._retry(
            lambda: self._call_container_api("/sessions", "GET", timeout=20),
            "列出所有会话",
            retries=3,
            interval=2,
        )

    async def export_session_jsonl(self, session_id: str, local_dir: str = "./exports") -> str:
        """
        导出指定 session 的 jsonl 到本地。
        这里依然走容器内 curl + cat，适合中小文件。
        """
        if not self._initialized:
            raise OpenClawSandboxError("请先调用 initialize()")

        if not session_id:
            raise OpenClawSandboxError("session_id 不能为空")

        output_dir = Path(local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        endpoint = f"/sessions/{session_id}/export"
        remote_tmp_path = f"/tmp/session_{session_id}.jsonl"

        async def _export():
            download_cmd = (
                f"curl -sS --max-time 60 -f "
                f"http://127.0.0.1:{self.container_api_port}{endpoint} "
                f"-o {remote_tmp_path}"
            )
            download_result = await self.execute_command(download_cmd)
            _, stderr1, exit_code1 = self._extract_execute_result(download_result)
            if exit_code1 != 0:
                raise OpenClawSandboxError(f"下载会话文件失败: {stderr1}")

            cat_result = await self.execute_command(f"cat {remote_tmp_path}")
            content, stderr2, exit_code2 = self._extract_execute_result(cat_result)
            if exit_code2 != 0:
                raise OpenClawSandboxError(f"读取会话文件失败: {stderr2}")

            if not content:
                raise OpenClawSandboxError(f"导出的文件为空: session_id={session_id}")

            return content

        file_content = await self._retry(
            _export,
            f"导出会话 {session_id}",
            retries=5,  # 增加重试次数
            interval=3,
        )

        local_path = output_dir / f"session_{session_id}.jsonl"
        local_path.write_text(file_content, encoding="utf-8")
        self.logger.info(f"已导出到本地: {local_path}")
        return str(local_path)

    async def delete_session(self, session_id: str) -> dict:
        """
        删除指定会话。
        """
        if not self._initialized:
            raise OpenClawSandboxError("请先调用 initialize()")

        if not session_id:
            raise OpenClawSandboxError("session_id 不能为空")

        return await self._retry(
            lambda: self._call_container_api(f"/sessions/{session_id}", "DELETE", timeout=20),
            f"删除会话 {session_id}",
            retries=3,
            interval=2,
        )

    async def stop(self):
        """
        停止 sandbox 并释放资源。
        """
        if self.sandbox_id:
            await self._retry(
                lambda: self.stop_sandbox(self.sandbox_id),
                "停止 sandbox",
                retries=5,
                interval=2,
            )

        self.sandbox_id = None
        self._initialized = False
        self.logger.info("资源已释放")
