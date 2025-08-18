import asyncio


async def main():
    async def a():
        await asyncio.sleep(1)
        print("a")

    async def b():
        await asyncio.sleep(1.1)
        raise ValueError("invalid function b")

    print(await asyncio.gather(a(), b(), return_exceptions=True))


asyncio.run(main())
