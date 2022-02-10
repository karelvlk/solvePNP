# NAKI-JESENIKY-SOLVEPNP

## Installation

1. Requirements
2. Run in development mode
3. Build, release and deploy

## Requirements

None

## Run in development mode

Run service in development (available on host "http://localhost:5000")
```
.\run.ps1 -type service
```

Run tests
```PowerShell
.\run.ps1 -type test
```
```Bash
sh run.sh -t test
```

## Build, release and deploy

Build image and publish to repository production version
(login to Azure and Docker is required only if you are not logged previously)

- update version to X.Y.Z in files
    `./package.json`
    `./src/version.py`

```
az login
docker login
.\run.ps1 -type release -version X.Y.Z
```

Run service from production image (available on host "http://localhost:5000")
```
.\run.ps1 -type production -version X.Y.Z
```

## Using

It is describe in manual.txt
