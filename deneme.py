import time
import asyncio
async def factorial(number):
    for i in range(number):
        await asyncio.sleep(i)


async def main():
	# gather bizim için bir program dizisi hazırlayıp çalıştırıyor.
    
	await asyncio.gather(
	factorial(1),
	factorial(2),
	factorial(3),
    )
    

print(f"started at {time.strftime('%X')}")
asyncio.run(main())
print(f"finished at {time.strftime('%X')}")