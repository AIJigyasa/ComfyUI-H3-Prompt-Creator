from .h3_prompt_creator import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _ollama_models,
)
from .h3_use_case_prompt_creator import H3UseCasePromptCreator

NODE_CLASS_MAPPINGS = dict(NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS["H3UseCasePromptCreator"] = H3UseCasePromptCreator
NODE_DISPLAY_NAME_MAPPINGS["H3UseCasePromptCreator"] = "H3 Use Case Prompt Creator"

WEB_DIRECTORY = "./js"

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/h3_prompt_creator/ollama_models")
    async def h3_ollama_models(request):
        base_url = request.query.get("url", "http://127.0.0.1:11434")
        return web.json_response({"models": _ollama_models(base_url)})

    @PromptServer.instance.routes.get("/h3_prompt_creator/use_case_fields")
    async def h3_use_case_fields(request):
        """Which widgets each use case actually needs.

        Served from Python so the UI cannot drift out of sync with
        USE_CASE_GUIDES when a use case gains or loses a field.
        """
        from .h3_use_case_prompt_creator import COMMON_FIELDS, USE_CASE_GUIDES

        return web.json_response({
            "always": sorted(COMMON_FIELDS),
            "by_use_case": {
                name: {
                    "fields": guide.get("fields", []),
                    "required": guide.get("required", []),
                }
                for name, guide in USE_CASE_GUIDES.items()
            },
        })
except Exception:
    # Allows importing/linting outside a running ComfyUI server.
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
