import asyncio
from aiohttp import web


async def echo_handler(request):
    try:
        data = await request.json()
    except Exception:
        data = {"error": "invalid json"}
    
    await asyncio.sleep(0.05)
    
    return web.json_response({"echoed": data, "status": "ok"})


async def start_server(port=9000):
    app = web.Application()
    app.router.add_post('/echo', echo_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', port)
    await site.start()
    
    print(f"Echo server started on http://127.0.0.1:{port}/echo")
    print("Press Ctrl+C to stop.")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(start_server())
