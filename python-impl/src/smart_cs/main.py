# 创建 FastAPI 应用，并装配数据库、模型、运行时和路由。

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.engine import make_url

from smart_cs.api.routers.conversations import router as conversations_router
from smart_cs.agents.knowledge import KnowledgeService
from smart_cs.agents.vision import LangChainVisionModel, VisionAgent
from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.application.context_builder import RuntimeContextBuilder
from smart_cs.application.conversation_service import ConversationService
from smart_cs.application.memory import ConversationSummarizer, MemoryWriteback, SqlMemoryStoreAdapter
from smart_cs.application.session_facts import SessionFactsExtractor
from smart_cs.config import Settings
from smart_cs.domain.errors import (
    ConversationBusyError,
    ConversationLeaseLostError,
    InvalidActionState,
    ToolPermissionError,
)
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import configured_model_profiles
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.infrastructure.assets import LocalAssetStorage
from smart_cs.tools.executor import AuthorizedToolExecutor


@dataclass(frozen=True)
class RuntimeBundle:
    """FastAPI 应用启动时创建、关闭时统一清理的一组运行资源。"""

    database: Database
    repository: SqlRepository
    runtime: AgentRuntime


class LazyKnowledgeService:
    """Defer heavy RAG setup until the first knowledge request."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._service: KnowledgeService | None = None

    def answer(self, query: str):
        return self._get_service().answer(query)

    def _get_service(self) -> KnowledgeService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    from smart_cs.rag.embeddings import LocalSentenceEmbeddings
                    from smart_cs.rag.retrieval import RuleBasedQueryRewriter
                    from smart_cs.rag.vector_store import connect_hybrid_store

                    embeddings = LocalSentenceEmbeddings(self._settings.embedding_model)
                    self._service = KnowledgeService(
                        connect_hybrid_store(self._settings, embeddings),
                        RuleBasedQueryRewriter(),
                    )
        return self._service


def build_runtime(
    settings: Settings, knowledge_service: KnowledgeService | None = None
) -> RuntimeBundle:
    # 确保 SQLite 数据库文件和 LangGraph checkpoint 文件所在目录已存在。
    _ensure_sqlite_parent(settings.database_url)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建数据库访问对象、仓库层，并初始化表结构和演示数据。
    database = Database(settings.database_url)
    repository = SqlRepository(database)
    repository.create_schema()
    repository.seed_demo_data()

    if settings.model_mode.lower() == "rules":
        raise ValueError("rules mode has been removed; configure SMART_CS_MODEL_MODE=llm")
    profiles = configured_model_profiles(settings)

    # 如果调用方没有注入 KnowledgeService，并且启用了 RAG，就延迟创建知识问答服务。
    if knowledge_service is None and settings.rag_enabled:
        knowledge_service = LazyKnowledgeService(settings)

    # 组装多 Agent 运行时：Supervisor 只调用 sub-agent tools；底层工具仍由 executor 强制鉴权。
    memory_store = SqlMemoryStoreAdapter(repository)
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        checkpoint_path=settings.checkpoint_path,
        model_profiles=profiles,
        knowledge_service=knowledge_service,
        memory_writeback=MemoryWriteback(
            repository=repository,
            summarizer=ConversationSummarizer(summarizer=profiles.summary),
        ),
        context_builder=RuntimeContextBuilder(
            repository,
            memory_store,
            session_facts_extractor=SessionFactsExtractor(profiles.extraction),
        ),
        memory_store=memory_store,
    )

    # 返回统一资源包，供 FastAPI app 生命周期持有并在关闭时清理。
    return RuntimeBundle(database=database, repository=repository, runtime=runtime)


def create_app(
    settings: Settings | None = None,
    knowledge_service: KnowledgeService | None = None,
    vision_agent: VisionAgent | None = None,
    asset_storage: LocalAssetStorage | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    bundle = build_runtime(app_settings, knowledge_service)
    profiles = configured_model_profiles(app_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            bundle.runtime.close()
            bundle.database.dispose()

    app = FastAPI(title="smart-cs-agent", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.database = bundle.database
    app.state.repository = bundle.repository
    app.state.runtime = bundle.runtime
    if vision_agent is None:
        vision_agent = VisionAgent(LangChainVisionModel(profiles.vision))
    if asset_storage is None:
        asset_storage = LocalAssetStorage(app_settings.asset_root)
    app.state.service = ConversationService(
        repository=bundle.repository,
        runtime=bundle.runtime,
        vision_agent=vision_agent,
        asset_storage=asset_storage,
    )
    _register_error_handlers(app)
    app.include_router(conversations_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "smart-cs-agent",
            "phase": "foundation",
        }

    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ToolPermissionError)
    async def permission_error(_request: Request, exc: ToolPermissionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ConversationBusyError)
    async def busy_error(_request: Request, exc: ConversationBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ConversationLeaseLostError)
    async def lease_lost_error(
        _request: Request, exc: ConversationLeaseLostError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": f"{exc}. Please retry the request."},
        )

    @app.exception_handler(InvalidActionState)
    async def invalid_action_error(_request: Request, exc: InvalidActionState) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )


def _ensure_sqlite_parent(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return
    Path(url.database).parent.mkdir(parents=True, exist_ok=True)


app = create_app()
