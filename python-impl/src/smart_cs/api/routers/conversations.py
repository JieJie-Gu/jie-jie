from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from smart_cs.api.dependencies import get_service
from smart_cs.api.schemas import (
    ConfirmRequest,
    ConversationCreateRequest,
    ConversationResponse,
    ConversationWorkflowResponse,
    MessageRequest,
    ToolCallsResponse,
)
from smart_cs.application.conversation_service import ConversationService


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest,
    service: ConversationService = Depends(get_service),
) -> dict[str, str]:
    return service.create_conversation(request.customer_id)


@router.post("/{conversation_id}/messages", response_model=ConversationWorkflowResponse)
def send_message(
    conversation_id: str,
    request: MessageRequest,
    service: ConversationService = Depends(get_service),
) -> dict:
    return service.send_message(conversation_id, request.customer_id, request.content)


@router.post("/{conversation_id}/messages-with-image", response_model=ConversationWorkflowResponse)
async def send_message_with_image(
    conversation_id: str,
    customer_id: str = Form(min_length=1),
    content: str = Form(min_length=1),
    image: UploadFile = File(...),
    service: ConversationService = Depends(get_service),
) -> dict:
    return service.send_message_with_image(
        conversation_id,
        customer_id,
        content,
        image.filename or "image",
        image.content_type or "application/octet-stream",
        await image.read(),
    )


@router.post("/{conversation_id}/actions/confirm", response_model=ConversationWorkflowResponse)
def confirm_action(
    conversation_id: str,
    request: ConfirmRequest,
    service: ConversationService = Depends(get_service),
) -> dict:
    return service.confirm(
        conversation_id,
        request.customer_id,
        request.action_id,
        approved=request.approved,
    )


@router.get("/{conversation_id}/tool-calls", response_model=ToolCallsResponse)
def list_tool_calls(
    conversation_id: str,
    customer_id: str = Query(min_length=1),
    service: ConversationService = Depends(get_service),
) -> dict:
    return {"tool_calls": service.list_tool_calls(conversation_id, customer_id)}
