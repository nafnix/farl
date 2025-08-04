from typing import cast

from starlette.datastructures import MutableHeaders

from navio.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY, STATE_KEY
from navio.utils import NavioState

from .types import AsyncNavioProtocol, NavioProtocol


class NavioMiddleware:
    def __init__(
        self,
        app,
        navio: NavioProtocol | AsyncNavioProtocol | None = None,
    ) -> None:
        self.app = app
        self.navio = navio

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        navio_state = (
            NavioState(policy=[], state=[], violated=[])
            if self.navio is None
            else NavioState(navio=self.navio, policy=[], state=[], violated=[])
        )
        scope.setdefault("state", {}).setdefault(STATE_KEY, navio_state)

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start" and (
                navio_state := scope.get("state", {}).get(STATE_KEY)
            ):
                navio_state = cast(NavioState, navio_state)

                best_none = None
                best = None
                for i in navio_state["state"]:
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

                if navio_state["policy"]:
                    ratelimit_policy = ",".join(str(i) for i in navio_state["policy"])
                    headers.setdefault(HEADER_RATELIMIT_POLICY, ratelimit_policy)

                if best or best_none:
                    ratelimit = str(best or best_none)
                    headers.setdefault(HEADER_RATELIMIT, ratelimit)

                await send(message)
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)
