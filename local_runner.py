"""批量运行本地 Docker OpenClaw 容器任务的主控制器（rock_runner 本地版）"""
import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from rock.local_docker import LocalDockerSandbox
from utils import LLMJudge, KeyPool, ConfigGenerator, async_retry

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f'local_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalTaskRunner:
    """本地 Docker 任务运行器 - 使用 openclaw-cluster:v6 镜像，流程与 rock_runner 一致"""

    DATASET_CHOICES = {
        "agenthazard_strongest": "data/dataset.json",
    }

    def __init__(self, config_path: str = "config/config_local.json", dataset: str = None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 确定数据集路径和对应的输出目录名
        if dataset and dataset in self.DATASET_CHOICES:
            dataset_path = self.DATASET_CHOICES[dataset]
            self.dataset_name = dataset
        else:
            dataset_path = self.config['paths']['dataset']
            self.dataset_name = Path(dataset_path).stem

        self.config['paths']['dataset'] = dataset_path

        # 初始化 Key Pool（支持多 key 轮询）
        self.key_pool = KeyPool(self.config['agent']['api_keys'])

        # 初始化 Judge Key Pool（也支持多 key）
        judge_keys = self.config['agent']['api_keys']  # Judge 和 Agent 共用 keys
        self.judge_key_pool = KeyPool(judge_keys)

        # 初始化配置生成器
        self.config_generator = ConfigGenerator(
            self.config['paths']['openclaw_config']
        )

        # 确保导出目录存在
        Path(self.config['paths']['exports_dir']).mkdir(parents=True, exist_ok=True)

        # 加载数据集
        with open(self.config['paths']['dataset'], 'r', encoding='utf-8') as f:
            self.dataset: List[Dict] = json.load(f)

        # 加载已完成的任务ID（断点续传）
        self.completed_ids = self._load_completed_ids()

        # 并发控制
        self.semaphore = asyncio.Semaphore(self.config['execution']['concurrency'])
        self.results_lock = asyncio.Lock()

    def _load_completed_ids(self) -> set:
        """从结果CSV加载已完成的任务ID"""
        csv_path = Path(self.config['paths']['results_csv'])
        if not csv_path.exists():
            return set()

        completed = set()
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add(int(row['id']))
            logger.info(f"加载 {len(completed)} 个已完成任务")
        except Exception as e:
            logger.warning(f"加载已完成任务失败: {e}")

        return completed

    async def _save_result(self, result: Dict[str, Any]):
        """保存单个结果到CSV（线程安全）"""
        csv_path = Path(self.config['paths']['results_csv'])

        async with self.results_lock:
            file_exists = csv_path.exists()

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'harmful', 'score', 'reason', 'target',
                             'category', 'comment', 'jailbreak_method', 'error']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(result)

    async def _create_judge(self) -> LLMJudge:
        """创建 Judge 实例（使用轮询的 key）"""
        api_key = await self.judge_key_pool.get_key()
        return LLMJudge(
            api_key=api_key,
            base_url=self.config['agent']['base_url'],
            model_name=self.config['judge']['model'],
            max_retries=self.config['execution']['max_retries']
        )

    async def _upload_mock_secrets(self, client: LocalDockerSandbox, item_id: int):
        """上传 mock secrets 到容器中（模拟真实环境的敏感数据）"""
        secrets_dir = Path("mock_secrets")
        if not secrets_dir.exists():
            logger.warning(f"[Item {item_id}] mock_secrets 目录不存在，跳过上传")
            return

        try:
            # 创建必要的目录
            await client.execute_command("mkdir -p /root/.ssh")
            await client.execute_command("mkdir -p /root/project")

            # 上传 .env 文件到多个位置
            env_file = secrets_dir / ".env"
            if env_file.exists():
                await client.upload_file(str(env_file), "/root/.env")
                await client.upload_file(str(env_file), "/root/project/.env")
                logger.info(f"[Item {item_id}] 上传 .env 文件")

            # 上传 SSH 私钥（带正确权限）
            ssh_key = secrets_dir / "id_rsa"
            if ssh_key.exists():
                await client.upload_file(str(ssh_key), "/root/.ssh/id_rsa")
                await client.execute_command("chmod 600 /root/.ssh/id_rsa")
                logger.info(f"[Item {item_id}] 上传 SSH 私钥")

            # 上传配置文件
            config_json = secrets_dir / "config.json"
            if config_json.exists():
                await client.upload_file(str(config_json), "/root/config.json")
                await client.upload_file(str(config_json), "/root/project/config.json")
                logger.info(f"[Item {item_id}] 上传 config.json")

            # 上传凭据文件
            credentials_txt = secrets_dir / "credentials.txt"
            if credentials_txt.exists():
                await client.upload_file(str(credentials_txt), "/root/credentials.txt")
                await client.upload_file(str(credentials_txt), "/root/project/credentials.txt")
                logger.info(f"[Item {item_id}] 上传 credentials.txt")

            logger.info(f"[Item {item_id}] Mock secrets 上传完成")

        except Exception as e:
            logger.warning(f"[Item {item_id}] Mock secrets 上传失败（不影响继续执行）: {e}")

    @async_retry(max_retries=3, delay=10.0, backoff=1.0)
    async def _run_single_item(self, item: Dict) -> Dict[str, Any]:
        """
        运行单个数据项

        关键设计：
        1. 每个任务使用独立的临时配置文件（避免并发覆盖）
        2. 从 KeyPool 轮询获取 API key
        3. 多层容器释放保险（async with + finally + 幂等 stop）
        4. 临时配置文件清理（确保不残留）
        5. 上传 mock secrets 到容器（模拟真实敏感数据）
        """
        item_id = item['id']
        logger.info(f"[Item {item_id}] 开始处理")

        # 初始化结果
        result = {
            'id': item_id,
            'target': item['query'],
            'category': item['category'],
            'comment': item['comment'],
            'jailbreak_method': item['jailbreak_method'],
            'harmful': False,
            'score': 0,
            'reason': '',
            'error': ''
        }

        # 临时配置文件路径
        temp_config_path = f"config_local/openclaw_temp_{item_id}.json"
        sandbox = None

        try:
            # 1. 从 KeyPool 获取 API key
            api_key = await self.key_pool.get_key()
            logger.info(f"[Item {item_id}] 分配 API key: {api_key[:4]}...{api_key[-4:]}")

            # 2. 生成独立的临时配置文件
            self.config_generator.generate_config(
                api_key=api_key,
                base_url=self.config['agent']['base_url'],
                model=self.config['agent']['model'],
                output_path=temp_config_path
            )
            logger.info(f"[Item {item_id}] 临时配置: {temp_config_path}")

            # 3. 创建本地 Docker 容器（使用 async with 确保 __aexit__ 调用）
            async with LocalDockerSandbox(
                local_config_path=temp_config_path,  # 使用独立配置
                image=self.config['local_docker']['image'],
                timeout=self.config['execution']['timeout'],
                log_level=logging.WARNING
            ) as client:
                sandbox = client

                # 4. 初始化容器
                logger.info(f"[Item {item_id}] 初始化容器")
                await client.initialize()
                logger.info(f"[Item {item_id}] 容器 ID: {client.container_id}")

                # 5. 上传 mock secrets（容器就绪后）
                await self._upload_mock_secrets(client, item_id)

                # 6. 逐 turn 发送消息
                session_id = f"item-{item_id}"
                decomposed_queries = item['decomposed_query']

                for turn_idx, query in enumerate(decomposed_queries, 1):
                    logger.info(
                        f"[Item {item_id}] Turn {turn_idx}/{len(decomposed_queries)}"
                    )
                    resp = await client.send_message(
                        query,
                        session_id=session_id,
                        timeout=self.config['execution']['timeout']
                    )

                    if resp.get('status') == 'error':
                        logger.warning(
                            f"[Item {item_id}] Turn {turn_idx} 失败: {resp.get('error')}"
                        )

                # 7. 导出会话记录
                logger.info(f"[Item {item_id}] 导出会话记录")
                export_path = await client.export_session_jsonl(
                    session_id,
                    self.config['paths']['exports_dir']
                )
                logger.info(f"[Item {item_id}] 导出成功: {export_path}")

                # 8. 读取会话记录
                with open(export_path, 'r', encoding='utf-8') as f:
                    session_records = [json.loads(line) for line in f]

                # 9. LLM Judge 评估
                logger.info(f"[Item {item_id}] 开始 Judge 评估")
                judge = await self._create_judge()
                judge_result = await judge.evaluate_trajectory(
                    session_records=session_records,
                    target=item['query'],
                    comment=item['comment'],
                    jailbreak_method=item['jailbreak_method'],
                    category=item['category'],
                    record_id=item_id
                )

                result.update({
                    'harmful': judge_result['harmful'],
                    'score': judge_result['score'],
                    'reason': judge_result['reason']
                })

                logger.info(
                    f"[Item {item_id}] 完成 - harmful={result['harmful']}, "
                    f"score={result['score']}"
                )

                # async with 会自动调用 client.__aexit__() -> client.stop()
                # 这里会释放容器资源

        except Exception as e:
            error_msg = str(e)[:200]
            logger.error(f"[Item {item_id}] 失败: {error_msg}", exc_info=True)
            result['error'] = error_msg

        finally:
            # === 多层容器释放保险 ===

            # 1. 如果 sandbox 对象存在，显式调用 stop()（幂等的）
            if sandbox is not None:
                try:
                    logger.info(f"[Item {item_id}] 释放容器资源 (container_id: {getattr(sandbox, 'container_id', 'N/A')})")
                    await sandbox.stop()
                    logger.info(f"[Item {item_id}] 容器已释放")
                except Exception as stop_error:
                    logger.error(f"[Item {item_id}] 容器释放失败: {stop_error}")

            # 2. 清理临时配置文件
            try:
                temp_config = Path(temp_config_path)
                if temp_config.exists():
                    temp_config.unlink()
                    logger.debug(f"[Item {item_id}] 临时配置已清理")
            except Exception as cleanup_error:
                logger.warning(f"[Item {item_id}] 清理临时配置失败: {cleanup_error}")

        return result

    async def _process_item_with_semaphore(self, item: Dict):
        """带信号量控制的任务处理（高并发控制）"""
        async with self.semaphore:
            result = await self._run_single_item(item)
            await self._save_result(result)

    async def run_all(self):
        """运行所有任务（高并发 + 断点续传 + 异常容错）"""
        pending_items = [
            item for item in self.dataset
            if item['id'] not in self.completed_ids
        ]

        logger.info("=" * 60)
        logger.info(f"总任务数: {len(self.dataset)}")
        logger.info(f"已完成: {len(self.completed_ids)}")
        logger.info(f"待处理: {len(pending_items)}")
        logger.info(f"并发数: {self.config['execution']['concurrency']}")
        logger.info(f"API Keys: {len(self.config['agent']['api_keys'])} 个")
        logger.info(f"Docker 镜像: {self.config['local_docker']['image']}")
        logger.info("=" * 60)

        if not pending_items:
            logger.info("所有任务已完成")
            return

        # 创建所有任务（Semaphore 自动控制并发）
        tasks = [
            self._process_item_with_semaphore(item)
            for item in pending_items
        ]

        # 并发执行（return_exceptions=True 确保单任务失败不影响整体）
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("=" * 60)
        logger.info("所有任务处理完成")
        logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="本地 Docker 批量评测运行器")
    parser.add_argument(
        "--dataset", "-d",
        choices=list(LocalTaskRunner.DATASET_CHOICES.keys()),
        default=None,
        help="选择数据集: agenthazard_strongest 或 atbench_trans（默认使用配置文件中的路径）"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config_local.json",
        help="配置文件路径（默认: config/config_local.json）"
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    runner = LocalTaskRunner(config_path=args.config, dataset=args.dataset)
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
