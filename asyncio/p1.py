# import time

# def task(name):
#     print(f"start {name}")
#     time.sleep(1) # blocking
#     print(f"end {name}")

# def main():
#     task("A")
#     task("B")

# main()

import asyncio

async def task(name):
    print(f"start {name}")
    await asyncio.sleep(1) 
    print(f"end {name}")

async def main():
    await asyncio.gather(task("A"), task("B"))

asyncio.run(main())