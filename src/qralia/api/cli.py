"""Point d'entrée local de l'API."""


def main() -> None:
    import uvicorn

    uvicorn.run("qralia.api.main:app", host="127.0.0.1", port=8000, reload=False)

