
param (
    [Parameter(Mandatory=$true)][string]$type="",
    [Parameter(Mandatory=$false)][string]$version=""
)

Function BuildAndPushImage
{
    if ($version) {
        iex "docker build --no-cache -f .\Dockerfile-prod -t mavlk/naki-jeseniky-solvepnp:$version ."
        iex "docker push mavlk/naki-jeseniky-solvepnp:$version"
    } else {
        Write-Host "Version must be specified."
    }
}

Function RunService
{
    $volumes =
        " -v ${PWD}:/app"

    $env =
        " -e FLASK_DEBUG=1"

    Try {
        iex "docker build -f .\Dockerfile  . -t naki-jeseniky-solvepnp"
        iex "docker run $volumes $env --name naki-jeseniky-solvepnp --rm -p 5000:5000 naki-jeseniky-solvepnp"
    } Finally {
        iex "docker stop naki-jeseniky-solvepnp"
    }
}

Function RunServiceFromProductionImage
{
    if ($version) {
        Try {
            iex "docker run --name naki-jeseniky-solvepnp --rm -p 5000:5000 mavlk/naki-jeseniky-solvepnp:$version"
        } Finally {
            iex "docker stop naki-jeseniky-solvepnp"
        }
    } else {
        Write-Host "Version must be specified."
    }
}

Function RunTest
{
    $volumes =
        " -v ${PWD}:/app"

    $env =
        ""

    Try {
        iex "docker build -f .\Dockerfile  . -t naki-jeseniky-solvepnp"
        iex "docker build -f .\Dockerfile-test  . -t naki-jeseniky-solvepnp-test"
        iex "docker run $volumes $env -d --name service --rm naki-jeseniky-solvepnp"
        iex "docker run $volumes $env --link service:service --rm naki-jeseniky-solvepnp-test"
    } Finally {
        iex "docker logs service"
        iex "docker stop service"
    }
}

Write-Host "Running task:", $type

switch ($type) {
    # run development standalone application/module
    "service"       { RunService }
    "test"          { RunTest }

    # run production image
    "production"    { RunServiceFromProductionImage}

    # run build, release or deploy production tools
    "release"       { BuildAndPushImage }

}
