
from flask import request, jsonify
from solvepnp_func import solver

def solvePnp(params):

    w = params.get("x", [])
    p = params.get("u", [])
    size = params.get("size", [])
    f = params.get("f", None)
    s = params.get("s", None)
    i = params.get("i", None)
    solve = solver(w, p, size, f, s, i)
    return solve

def solvePnpFromHttpGet():
    params = {}

    for key in request.args:
        params[key] = request.args.get(key)

    return solvePnp(params)

def solvePnpFromHttpPost():
    params = request.json

    return solvePnp(params)
