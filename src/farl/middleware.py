from typing import cast

from starlette.datastructures import MutableHeaders

from farl.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY, STATE_KEY
from farl.types import AsyncFarlProtocol, FarlProtocol, FarlState


class FarlMiddleware:
    def __init__(
        self,
        app,
        farl: FarlProtocol | AsyncFarlProtocol | None = None,
    ) -> None:
        self.app = app
        self.farl = farl

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        farl_state = (
            FarlState(policy=[], state=[], violated=[])
            if self.farl is None
            else FarlState(farl=self.farl, policy=[], state=[], violated=[])
        )
        scope.setdefault("state", {}).setdefault(STATE_KEY, farl_state)

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start" and (
                farl_state := scope.get("state", {}).get(STATE_KEY)
            ):
                farl_state = cast(FarlState, farl_state)

                best_none = None
                best = None
                for i in farl_state["state"]:
                    if i.reset_timestamp is None:
                        if (best_none is None) or (i.remaining < best_none.remaining):
                            best_none = i
                    elif (best is None) or (
                        i.remaining,
                        i.reset_timestamp,
                    ) < (
                        best.remaining,
                        best.reset_timestamp,
                    ):
                        best = i

                headers = MutableHeaders(scope=message)

                if farl_state["policy"]:
                    ratelimit_policy = ",".join(str(i) for i in farl_state["policy"])
                    headers.setdefault(HEADER_RATELIMIT_POLICY, ratelimit_policy)

                if best or best_none:
                    ratelimit = str(best or best_none)
                    headers.setdefault(HEADER_RATELIMIT, ratelimit)

                await send(message)
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)
