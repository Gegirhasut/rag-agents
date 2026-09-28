from fastapi import APIRouter

from rag_agents.api.v1 import agents, documents, keys, query

router = APIRouter(prefix="/api/v1")
router.include_router(agents.router)
router.include_router(documents.router)
router.include_router(query.router)
router.include_router(keys.router)
