from typing import Any, Callable
import anyio


async def run_in_thread(fn: Callable[..., Any], *args, **kwargs) -> Any:
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))

