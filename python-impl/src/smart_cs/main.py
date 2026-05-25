from fastapi import FastAPI


app = FastAPI(title="smart-cs-agent")


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "smart-cs-agent",
        "phase": "foundation",
    }
