from collections.abc import Awaitable, Callable


class Reservation:
    """Something set aside for one request (tokens, money), to be settled against what it used.

    Settling refunds whatever was reserved but not used, or charges for any excess. It happens
    at most once, so calling it again (a cleanup path after a normal finish, say) is harmless.
    """

    def __init__(
        self,
        reserved: int,
        prompt_tokens: int,
        adjust: Callable[[int], Awaitable[int | None]] | None,
        balance: int | None = None,
    ) -> None:
        self.reserved = reserved
        self.prompt_tokens = prompt_tokens  # token reservations only: our estimate of the prompt
        self._adjust = adjust
        self.balance = balance  # what the allowance had left, as of the last time we looked
        self._settled = False

    async def settle(self, actual: int) -> None:
        if self._settled or self._adjust is None:
            self._settled = True
            return
        self._settled = True
        delta = self.reserved - max(0, actual)
        if delta != 0:
            balance = await self._adjust(delta)
            if balance is not None:
                self.balance = balance

    async def release(self) -> None:
        """The request used nothing (it failed before producing output): refund everything."""
        await self.settle(0)


NO_RESERVATION = Reservation(0, 0, None)
