# 本地运行

## Python API

```bash
cd python-impl
conda run -n agent uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

默认 `SMART_CS_MODEL_MODE=rules`，不需要模型接口；在该模式中图片会保守地进入转人工
确认流程。文本售后和转人工动作仍需要用户确认。

## Milvus 知识检索

`docker-compose.yml` 提供 `etcd`、`minio`、`standalone` 和 `python-agent` 服务。
当前用户需要具有 Docker socket 权限才能启动它们。

```bash
docker compose up -d etcd minio standalone
cd python-impl
conda run -n agent python scripts/index_knowledge.py
conda run -n agent pytest tests/integration/test_milvus_hybrid.py -q
conda run -n agent python scripts/evaluate_rag.py
```

没有 Milvus 服务时，只能运行明确标注的离线报告流程：

```bash
cd python-impl
conda run -n agent python scripts/evaluate_rag.py --offline
```
