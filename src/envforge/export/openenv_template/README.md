# envforge procedural environment

Run the server from this directory:

    pip install -r server/requirements.txt
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Connect:

    from openenv.core import GenericEnvClient
    with GenericEnvClient(base_url="http://localhost:8000").sync() as env:
        r = env.reset(seed=3)
        print(r.observation["prompt"])
        print(env.step({"answer": "42"}).reward)

Or build the image: `docker build -f server/Dockerfile -t envforge-env .`
