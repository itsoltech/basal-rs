import asyncio, json, time, httpx
L = [json.loads(l) for l in open("tools/bench/mixed.jsonl")]
long = next(r for r in L if r["class"] == "doc-16k")
shorts = [r for r in L if r["class"] == "short-1q"][:20]
async def main():
    async with httpx.AsyncClient(timeout=600) as c:
        url = "http://127.0.0.1:8100/v1/systemone"
        t0 = time.perf_counter()
        lt = asyncio.create_task(c.post(url, json=long["request"]))
        await asyncio.sleep(0.5)
        res = []
        for s in shorts:
            t = time.perf_counter(); r = await c.post(url, json=s["request"]); res.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200, r.text
        lr = await lt
        print("long", lr.status_code, "%.0f ms" % ((time.perf_counter() - t0) * 1000), "shorts ms", [round(x) for x in res])
        out = {"long": lr.json()["answers"], "shorts": [ (await c.post(url, json=s["request"])).json()["answers"] for s in shorts[:3]]}
        json.dump(out, open("/tmp/smoke_" + __import__("sys").argv[1] + ".json", "w"))
asyncio.run(main())
