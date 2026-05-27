from fastapi import Request

from smart_cs.application.conversation_service import ConversationService


def get_service(request: Request) -> ConversationService:
    return request.app.state.service
