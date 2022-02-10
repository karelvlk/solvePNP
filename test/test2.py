
import sys

sys.path.append("/app/src")

from colorama import Fore

from version import VERSION
from callService import callServiceInternal

def test2():
    print("Test HTTP headers in response")

    response = callServiceInternal({})
    if ("X-SOLVEPNP-VERSION" in response.headers):
        version = response.headers["X-SOLVEPNP-VERSION"]
        if (version == VERSION):
            print("  OK")
        else:
            print("  FAILED: Version contains bad value")
            print("    result: ", version)
            print("    required: ", VERSION)
    else:
        print("  FAILED: HTTP headers does not contain X-SOLVEPNP-VERSION")
